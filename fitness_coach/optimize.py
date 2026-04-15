"""GEPA optimization of PlanSignature instructions.

Demonstrates the upside of stripping a hand-tuned signature docstring back
to a terse contract and letting GEPA rebuild the elaboration empirically
against a methodology-grounded scoring function.

The production `signatures.PlanSignature` has a multi-paragraph docstring
encoding methodology rules, behavioral defaults, halted-movement handling,
and population-specific guidance. That's hand-engineered prompt — the kind
of work that should belong to an optimizer. This file:

  1. Defines `PlanSignatureMinimal` — same I/O contract, terse one-line
     docstring (the actual "signature as contract" pattern)
  2. Builds 6 hand-picked eval cases drawn from the synthetic athletes
     (powerlifter clean progression / plateau / shoulder pain;
     runner clean week / ITB warning / returning from skipped week)
  3. Scores plans by `methodology.validate_plan` — pass/fail signal that
     also respects `halted_movements` constraints
  4. Runs GEPA against PlanSignatureMinimal's instructions
  5. Prints baseline pass rate → optimized instructions → optimized pass rate

Cost: ~$0.30 per run on gpt-4o-mini (task LM) + gpt-4o (reflection LM).

This file is run independently of the main demo:
    uv run python -m fitness_coach.optimize
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dspy
import gepa
from gepa import EvaluationBatch, GEPAAdapter
from pydantic_ai import Agent

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toy import agent_from_signature  # noqa: E402, F401  (kept for parity with rest of codebase)

from fitness_coach.methodology import get_methodology  # noqa: E402
from fitness_coach.schemas import (  # noqa: E402
    AthleteState,
    HandoffDoc,
    ObservationOutput,
    PlanProposal,
)
from fitness_coach.synthetic import ATHLETES  # noqa: E402


# ====================================================================
# PlanSignatureMinimal — same I/O contract, no hand-tuned elaboration
# ====================================================================

class PlanSignatureMinimal(dspy.Signature):
    """Propose the next session's prescription for the athlete."""

    athlete_state: AthleteState = dspy.InputField(
        desc="Current longitudinal state",
    )
    observation: ObservationOutput = dspy.InputField(
        desc="Coded observation from the just-completed session",
    )
    prior_handoff: HandoffDoc | None = dspy.InputField(
        desc="Handoff document from the previous session, or None for the first session",
    )
    plan: PlanProposal = dspy.OutputField(
        desc="Prescription for the next session — activities + rationale",
    )


# ====================================================================
# Eval cases — representative situations sampled from synthetic library
# ====================================================================

@dataclass
class PlanEvalCase:
    name: str
    state: AthleteState
    observation: ObservationOutput
    prior_handoff: HandoffDoc | None
    halted_movements: frozenset[str]


def build_eval_cases() -> list[PlanEvalCase]:
    """Six hand-picked cases covering the architectural surface area."""
    pl_001_state, _ = ATHLETES["pl_001"]
    pl_002_state, _ = ATHLETES["pl_002"]
    rn_001_state, _ = ATHLETES["rn_001"]
    rn_002_state, _ = ATHLETES["rn_002"]

    return [
        PlanEvalCase(
            name="pl_001 healthy progression",
            state=pl_001_state.model_copy(deep=True),
            observation=ObservationOutput(
                progression_signals=["all sets at target RPE 7", "bar speed consistent"],
                concern_signals=[],
                evidence_quality_assessment="High-trust self-report; no objective concerns",
            ),
            prior_handoff=None,
            halted_movements=frozenset(),
        ),
        PlanEvalCase(
            name="pl_002 bench plateau (3 failures triggers deload rule)",
            state=pl_002_state.model_copy(deep=True, update={
                "consecutive_failed_sessions": {"bench": 3},
            }),
            observation=ObservationOutput(
                progression_signals=[],
                concern_signals=["bench plateau confirmed across 3 sessions", "RPE peaked at 10"],
                evidence_quality_assessment="High-trust evidence; technique check confirmed plateau is strength, not form",
            ),
            prior_handoff=None,
            halted_movements=frozenset(),
        ),
        PlanEvalCase(
            name="pl_002 shoulder pain on bench (movement halted)",
            state=pl_002_state.model_copy(deep=True),
            observation=ObservationOutput(
                progression_signals=[],
                concern_signals=["sharp pain in left front delt during bench, ~5/10"],
                evidence_quality_assessment="High-trust self-report on pain",
            ),
            prior_handoff=None,
            halted_movements=frozenset({"bench"}),
        ),
        PlanEvalCase(
            name="rn_001 healthy build week",
            state=rn_001_state.model_copy(deep=True),
            observation=ObservationOutput(
                progression_signals=["all easy paces felt easy", "long run smooth"],
                concern_signals=[],
                evidence_quality_assessment="Solid week, on-track for half-marathon plan",
            ),
            prior_handoff=None,
            halted_movements=frozenset(),
        ),
        PlanEvalCase(
            name="rn_002 ITB warning (injury history present)",
            state=rn_002_state.model_copy(deep=True, update={
                "current_weekly_mileage": 14.0,
                "weeks_at_current_mileage": 0,
            }),
            observation=ObservationOutput(
                progression_signals=[],
                concern_signals=["ITB tightness returning", "preventive halt of last run"],
                evidence_quality_assessment="High-trust self-report; consistent with prior ITB pattern",
            ),
            prior_handoff=None,
            halted_movements=frozenset(),
        ),
        PlanEvalCase(
            name="rn_001 returning from skipped week (baseline preserved)",
            state=rn_001_state.model_copy(deep=True),
            observation=ObservationOutput(
                progression_signals=["legs felt fresh from unintended rest"],
                concern_signals=["only 1 run completed last week — work travel"],
                evidence_quality_assessment="Self-report consistent; baseline mileage intact",
            ),
            prior_handoff=None,
            halted_movements=frozenset(),
        ),
    ]


# ====================================================================
# Prompt rendering (parallel to graph._render_prompt)
# ====================================================================

def _render(sig: type[dspy.Signature], **inputs: Any) -> str:
    lines = []
    for name, fi in sig.input_fields.items():
        val = inputs.get(name)
        if val is None:
            val = "(none)"
        elif hasattr(val, "model_dump_json"):
            val = val.model_dump_json(indent=2)
        prefix = fi.json_schema_extra.get("prefix", f"{name}:")
        desc = fi.json_schema_extra.get("desc", "")
        lines.append(f"{prefix} {desc}\n{val}")
    return "\n\n".join(lines)


# ====================================================================
# GEPA Adapter — bridges {plan: <instructions>} candidates to the agent
# ====================================================================

class PlanGEPAAdapter(GEPAAdapter):
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        # Build base agent once; instructions overridden per candidate.
        # CRITICAL: temperature=0 makes the eval deterministic. Without it the
        # same prompt produces different plans across runs and GEPA's score
        # signal is dominated by noise — measured 5/6 in optimization but 1/6
        # in re-eval on identical inputs.
        output_field = next(iter(PlanSignatureMinimal.output_fields.values()))
        self.base_agent = Agent(
            model=model,
            output_type=output_field.annotation,
            instructions=PlanSignatureMinimal.instructions,
            output_retries=3,
            model_settings={"temperature": 0.0},
        )

    def evaluate(
        self,
        batch: list[PlanEvalCase],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        outputs: list[PlanProposal | None] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] = []

        for case in batch:
            halted_hint = ""
            if case.halted_movements:
                halted_hint = (
                    f"\n\nHALTED MOVEMENTS THIS SESSION (do NOT include in the plan): "
                    f"{sorted(case.halted_movements)}\n"
                    "These have moderate+ pain signals and must be omitted or substituted."
                )
            prompt = _render(
                PlanSignatureMinimal,
                athlete_state=case.state,
                observation=case.observation,
                prior_handoff=case.prior_handoff,
            ) + halted_hint

            with self.base_agent.override(instructions=candidate["plan"]):
                try:
                    result = self.base_agent.run_sync(prompt)
                    plan: PlanProposal | None = result.output
                except Exception as e:  # noqa: BLE001
                    plan = None
                    score = 0.0
                    violations = [f"agent run failed: {type(e).__name__}: {str(e)[:120]}"]
                else:
                    methodology = get_methodology(case.state)
                    vr = methodology.validate_plan(
                        case.state, plan, halted_movements=case.halted_movements,
                    )
                    score = 1.0 if vr.is_valid else 0.0
                    violations = vr.violations

            outputs.append(plan)
            scores.append(score)
            if capture_traces:
                trajectories.append({
                    "case_name": case.name,
                    "score": score,
                    "violations": violations,
                    "plan_summary": (
                        f"{len(plan.activities)} activities" if plan else "FAILED"
                    ),
                })

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        records = []
        for traj in (eval_batch.trajectories or []):
            records.append({
                "Inputs": {"case": traj["case_name"]},
                "Generated Output": {"plan_summary": traj["plan_summary"]},
                "Feedback": (
                    "PASSED methodology validation"
                    if traj["score"] == 1.0
                    else f"FAILED. Violations: {traj['violations']}"
                ),
            })
        return {comp: records for comp in components_to_update}


# ====================================================================
# Main
# ====================================================================

def main():
    print("=" * 72)
    print("GEPA optimization of PlanSignature — minimal seed → optimized")
    print("=" * 72)

    cases = build_eval_cases()
    adapter = PlanGEPAAdapter(model="openai:gpt-4o-mini")

    seed = {"plan": "Propose the next session's prescription for the athlete."}

    print(f"\n--- BASELINE (terse seed: {len(seed['plan'])} chars) ---")
    print(f"  Seed: {seed['plan']!r}")
    baseline = adapter.evaluate(cases, seed, capture_traces=True)
    baseline_score = sum(baseline.scores) / len(baseline.scores)
    print(f"\n  Baseline pass rate: {baseline_score:.0%} "
          f"({int(sum(baseline.scores))}/{len(baseline.scores)} cases)")
    for traj in baseline.trajectories:
        marker = "✓" if traj["score"] == 1.0 else "✗"
        print(f"    {marker} {traj['case_name']:55s}  ({traj['plan_summary']})")
        if traj["score"] < 1.0:
            for v in traj["violations"][:2]:
                print(f"        ! {v[:110]}")

    print(f"\n{'=' * 72}")
    print("GEPA OPTIMIZATION (max 30 metric calls, gpt-4o reflection)")
    print(f"{'=' * 72}")
    result = gepa.optimize(
        seed_candidate=seed,
        trainset=cases,
        valset=cases,
        adapter=adapter,
        reflection_lm="openai/gpt-4o",
        max_metric_calls=60,  # 30 was too few — only 2 candidates explored
        reflection_minibatch_size=3,
        display_progress_bar=False,
    )

    # Tie-break on recency: later candidates evolved from prior ones, so
    # when aggregate scores tie they are at least as good and often
    # broader on per-instance Pareto coverage. Picking the seed on a tie
    # would obscure GEPA's actual contribution.
    best_idx = max(
        range(len(result.candidates)),
        key=lambda i: (result.val_aggregate_scores[i], i),
    )
    best = result.candidates[best_idx]

    print(f"\n{'=' * 72}")
    print(f"OPTIMIZED instructions ({len(best['plan'])} chars)")
    print(f"{'=' * 72}")
    # Word-wrap the instructions for readability
    import textwrap
    print(textwrap.fill(best["plan"], width=72))

    optimized = adapter.evaluate(cases, best, capture_traces=True)
    optimized_score = sum(optimized.scores) / len(optimized.scores)
    print(f"\n--- POST-OPTIMIZATION ---")
    print(f"  Pass rate: {optimized_score:.0%} "
          f"({int(sum(optimized.scores))}/{len(optimized.scores)} cases)")
    for traj in optimized.trajectories:
        marker = "✓" if traj["score"] == 1.0 else "✗"
        print(f"    {marker} {traj['case_name']:55s}  ({traj['plan_summary']})")
        if traj["score"] < 1.0:
            for v in traj["violations"][:2]:
                print(f"        ! {v[:110]}")

    print(f"\n{'=' * 72}")
    print(f"Lift: baseline {baseline_score:.0%} → optimized {optimized_score:.0%}  "
          f"({(optimized_score - baseline_score) * 100:+.0f} pp)")
    print(f"Candidates explored: {len(result.candidates)}")
    print(f"Total metric calls: {result.total_metric_calls}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
