"""Methodology-grounded LLM judge — scores coaching outputs on 4 axes.

Used to compare straw_coach and agentic-graph outputs side-by-side. The
judge is grounded in the published methodology rules (encoded as a
context string) and produces axis-by-axis scores with rationales.

The "judge grounded in published materials" pattern matters because no
real coach is in the loop here — published Rippetoe Starting Strength
and Daniels' Running Formula text supplies the rubric the LLM judge
scores against.

Limitations to keep in mind:
  - The judge is "at best equivalent to a freshly trained coach who has
    read the manual but never seen an athlete"
  - LLM judges have known biases (verbosity, authority, capability)
  - The judge cannot evaluate domain knowledge it does not have
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import dspy
from pydantic import BaseModel, Field

# Reuse the canonical Signature → Agent bridge from the root toy module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toy import agent_from_signature  # noqa: E402

from fitness_coach.schemas import (  # noqa: E402
    AthleteState,
    HaltedSession,
    SessionLog,
    SessionResult,
)
from fitness_coach.straw_coach import StrawCoachOutput  # noqa: E402


# ====================================================================
# Methodology context strings — what the judge knows
# ====================================================================

LINEAR_PROGRESSION_CONTEXT = """\
LINEAR PROGRESSION (Rippetoe-style, novice powerlifters)

Standard prescriptions:
  - Squat 3x5, +5 lb per session when target reached
  - Bench 3x5, +2.5 lb per session when target reached
  - Deadlift 1x5, +5 lb per session when target reached
  - OHP 3x5, +2.5 lb per session when target reached

Key principles:
  - Progress when last session hit all reps at target RPE (≤8)
  - Hold load on first failure; deload 10% after 3 consecutive failures
  - NEVER progress through pain — pain on a movement halts that movement
  - Bad sleep / acute fatigue → hold loads, do not progress

What "good" looks like:
  - Plan respects current load ± standard increment
  - Coaching cues are technical and movement-specific
  - State transition reflects evidence (RPE, completed reps, evidence quality)
  - Safety: pain reports halt the offending movement
"""

DANIELS_RUNNING_CONTEXT = """\
DANIELS' RUNNING FORMULA (recreational distance runners)

Key rules:
  - Weekly mileage progresses ≤+10% per week, only after 2+ healthy weeks at current
  - 80/20 distribution: ≥80% of weekly mileage should be easy/recovery
  - VDOT updates only after races, ≤±0.5 per update
  - Quality work (tempo, intervals) ≤20% of weekly mileage

What "good" looks like:
  - Plan respects mileage cap and 80/20 ratio
  - Coaching cues are specific to easy-pace discipline and form
  - State transition reflects readiness (weeks at current, injury history)
  - Safety: pain or injury recurrence halts running, not just slows it
  - Life-event interruption (skipped week) → hold mileage, do not "catch up"
  - Form regression → conservative volume, address gait
"""


def _methodology_context(state: AthleteState) -> str:
    if state.population == "powerlifter":
        return LINEAR_PROGRESSION_CONTEXT
    if state.population == "runner":
        return DANIELS_RUNNING_CONTEXT
    return ""


# ====================================================================
# Output models
# ====================================================================

class AxisScore(BaseModel):
    """One dimension of the evaluator's report."""
    score: int = Field(ge=1, le=5,
                       description="1 = methodology violated; 5 = methodology executed exactly")
    rationale: str = Field(min_length=1,
                           description="One-sentence justification, citing methodology rule or evidence")


class EvaluatorReport(BaseModel):
    """4-axis methodology-grounded score for a coaching output on one session."""
    plan_quality: AxisScore = Field(
        description="Does the prescribed next session match what methodology would recommend given current state and observed evidence?"
    )
    coaching_specificity: AxisScore = Field(
        description="Are the cues actionable and athlete-specific, or generic platitudes?"
    )
    adaptation_appropriateness: AxisScore = Field(
        description="Is the proposed state transition justified by accumulated evidence and respectful of methodology thresholds?"
    )
    safety_adherence: AxisScore = Field(
        description="Were safety signals (pain, injury recurrence, recovery deficit) properly handled — no progression through pain, deload when warranted?"
    )

    @property
    def aggregate(self) -> float:
        return (
            self.plan_quality.score
            + self.coaching_specificity.score
            + self.adaptation_appropriateness.score
            + self.safety_adherence.score
        ) / 4.0


# ====================================================================
# DSPy signature
# ====================================================================

class EvaluateSignature(dspy.Signature):
    """Score a coaching output on 4 dimensions against published methodology.

    You are a methodology expert. Given the session that just happened
    (athlete log + evidence + safety signals) and the coaching output
    produced (plan + cues + state transition + handoff), grade each
    dimension 1-5 with a one-sentence rationale.

    Score 1 = clear methodology violation (e.g., progressing load on a
    pain report). Score 5 = methodology executed precisely (e.g., held
    load on poor sleep evidence; deloaded after third consecutive
    failure; halted bench movement on shoulder pain).

    Cite specific methodology rules or session evidence in your
    rationales. Do not be lenient — if a coaching output ignored a
    visible safety signal, safety_adherence should be 1. If cues are
    generic platitudes ("focus", "stay relaxed"), coaching_specificity
    should be ≤2.

    Be calibrated: most coaching outputs should land 2-4 on most axes.
    Reserve 5 for outputs that visibly addressed a non-trivial signal."""

    methodology_context: str = dspy.InputField(
        desc="The published methodology rules this athlete follows",
    )
    athlete_state: AthleteState = dspy.InputField(
        desc="Current longitudinal state of the athlete",
    )
    session_log: SessionLog = dspy.InputField(
        desc="The session that just happened",
    )
    coaching_output_json: str = dspy.InputField(
        desc="The coach's output for this session — plan, cues, transition, handoff "
             "summary — serialized as JSON",
    )
    report: EvaluatorReport = dspy.OutputField(
        desc="4-axis score with rationales",
    )


# ====================================================================
# Public API
# ====================================================================

def _normalize_output_to_dict(
    output: SessionResult | HaltedSession | StrawCoachOutput,
) -> dict[str, Any]:
    """Project either a SessionResult, HaltedSession, or StrawCoachOutput into
    a common dict shape so the judge can score them uniformly."""
    if isinstance(output, SessionResult):
        return {
            "kind": "session_result",
            "plan": output.handoff.next_session_focus,  # the actual plan was applied to next session
            "coaching_cues": output.coaching.cues,
            "coaching_watch_for": output.coaching.watch_for,
            "coaching_rationale": output.coaching.rationale,
            "proposed_transition": (
                output.proposed_state_transition.model_dump(mode="json")
                if output.proposed_state_transition else None
            ),
            "handoff_summary": output.handoff.next_session_focus,
            "handoff_areas": output.handoff.areas_to_work_on,
            "handoff_what_worked": output.handoff.what_worked,
            "handoff_watch_for": output.handoff.what_to_watch_for,
            "used_fallback_plan": output.used_fallback_plan,
        }
    if isinstance(output, HaltedSession):
        return {
            "kind": "halted_session",
            "halted_at": output.halted_at_node,
            "reason": output.reason,
            "triggering_signals": [s.model_dump(mode="json") for s in output.triggering_signals],
        }
    if isinstance(output, StrawCoachOutput):
        return {
            "kind": "straw_coach_output",
            "plan_activities": [a.model_dump(mode="json") for a in output.plan.activities],
            "plan_rationale": output.plan.rationale,
            "coaching_cues": output.coaching.cues,
            "coaching_watch_for": output.coaching.watch_for,
            "coaching_rationale": output.coaching.rationale,
            "proposed_transition": (
                output.proposed_state_transition.model_dump(mode="json")
                if output.proposed_state_transition else None
            ),
            "handoff_summary": output.handoff_summary,
        }
    raise TypeError(f"Unsupported output type: {type(output).__name__}")


async def evaluate_session_output(
    athlete_state: AthleteState,
    session_log: SessionLog,
    output: SessionResult | HaltedSession | StrawCoachOutput,
    model: str = "openai:gpt-4o-mini",
) -> EvaluatorReport:
    """Run the methodology-grounded judge on one session output.

    Returns an `EvaluatorReport` with 4 axes scored 1-5 plus rationales.
    """
    agent = agent_from_signature(EvaluateSignature, model=model, output_retries=3)
    output_dict = _normalize_output_to_dict(output)

    prompt_parts = []
    for name, field_info in EvaluateSignature.input_fields.items():
        prefix = field_info.json_schema_extra.get("prefix", f"{name}:")
        desc = field_info.json_schema_extra.get("desc", "")

        if name == "methodology_context":
            value = _methodology_context(athlete_state)
        elif name == "athlete_state":
            value = athlete_state.model_dump_json(indent=2)
        elif name == "session_log":
            value = session_log.model_dump_json(indent=2)
        elif name == "coaching_output_json":
            value = json.dumps(output_dict, indent=2, default=str)
        else:
            continue

        header = f"{prefix} {desc}" if desc else prefix
        prompt_parts.append(f"{header}\n{value}")

    prompt = "\n\n".join(prompt_parts)
    result = await agent.run(prompt)
    return result.output
