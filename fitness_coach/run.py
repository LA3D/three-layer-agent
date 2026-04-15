"""Side-by-side demo: agentic cognitive core vs. rigid straw coach.

For each athlete:
  - Iterates through their 6 sessions
  - Per session: runs the straw coach + the agentic graph + the methodology
    judge on both
  - Threads longitudinal state forward (state evolves based on what actually
    happened in each session log)
  - Persists per-session traces as JSON to traces/{athlete_id}/session_{n}.json
    via FileStatePersistence — these are the GEPA-ready trajectories
  - Tracks the prior session's handoff document and feeds it into the next
    session's PlanNode

Output: per-session 4-axis evaluator scores side-by-side, then a summary
showing the gap on routine vs. surprise sessions.

Cost: ~$0.30 per full run on gpt-4o-mini (24 sessions × 2 systems × 1
evaluator call + 24 × ~5 graph node calls).
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_graph import FullStatePersistence
from pydantic_graph.persistence.file import FileStatePersistence

from fitness_coach.evaluator import EvaluatorReport, evaluate_session_output
from fitness_coach.graph import (
    IngestNode,
    SessionState,
    build_graph,
)
from fitness_coach.schemas import (
    AthleteState,
    HaltedSession,
    LiftActivity,
    PowerlifterState,
    RunActivity,
    RunnerState,
    SessionLog,
    SessionResult,
)
from fitness_coach.straw_coach import StrawCoachOutput, run_straw_session
from fitness_coach.synthetic import ATHLETES


TRACES_ROOT = Path(__file__).parent / "traces"


# ====================================================================
# State evolution — derive longitudinal state from prior session log
# ====================================================================

# Threshold below which a runner's session log is treated as a "disruption"
# (skipped week, illness) rather than a real reduction. Sparse logs do not
# overwrite current_weekly_mileage — they carry no information about the
# athlete's sustainable mileage.
SPARSE_RUN_LOG_THRESHOLD = 3


def evolve_state(prior_state: AthleteState, prior_log: SessionLog) -> AthleteState:
    """Update longitudinal state to reflect what happened in `prior_log`.

    Idempotent: short-circuits if `prior_log.session_index` has already been
    applied (matches `prior_state.last_evolved_session_index`).

    Both straw and agentic coaches see the same evolved state — comparison is
    on their reactions to identical inputs, not cumulative impact.
    """
    # Idempotency guard — repeated calls with the same log return the same state
    if (
        prior_state.last_evolved_session_index is not None
        and prior_log.session_index <= prior_state.last_evolved_session_index
    ):
        return prior_state

    if isinstance(prior_state, PowerlifterState):
        new = prior_state.model_copy(deep=True)
        for activity in prior_log.activity_log:
            if not isinstance(activity, LiftActivity):
                continue
            new.current_lifts[activity.exercise] = activity.load_lb
            failed = bool(activity.reps_per_set) and any(r < 5 for r in activity.reps_per_set)
            if failed:
                new.consecutive_failed_sessions[activity.exercise] = (
                    new.consecutive_failed_sessions.get(activity.exercise, 0) + 1
                )
            else:
                new.consecutive_failed_sessions[activity.exercise] = 0
        new.last_evolved_session_index = prior_log.session_index
        return new

    if isinstance(prior_state, RunnerState):
        new = prior_state.model_copy(deep=True)
        runs = [a for a in prior_log.activity_log if isinstance(a, RunActivity)]

        # Sparse log = disruption (skipped week, illness, travel). It carries
        # no information about sustainable mileage. Do not overwrite baseline
        # and do not increment the weeks-at-current counter.
        if len(runs) >= SPARSE_RUN_LOG_THRESHOLD:
            new.current_weekly_mileage = sum(r.distance_mi for r in runs)
            if not new.injury_history:
                new.weeks_at_current_mileage += 1
            else:
                # Injury triggers a hold — counter resets so progression
                # readiness is re-earned after the injury clears.
                new.weeks_at_current_mileage = 0

        new.last_evolved_session_index = prior_log.session_index
        return new

    raise TypeError(f"Unknown state type: {type(prior_state).__name__}")


# ====================================================================
# Per-session orchestration
# ====================================================================

@dataclass
class SessionComparison:
    """One session's worth of data for the comparison report."""
    athlete_id: str
    session_index: int
    is_surprise: bool
    surprise_summary: str
    straw_report: EvaluatorReport
    agentic_report: EvaluatorReport
    agentic_halted: bool
    agentic_used_fallback: bool


def _surprise_summary(log: SessionLog) -> tuple[bool, str]:
    """Heuristic: a session is a 'surprise' if it has a safety signal OR
    its evidence mentions a deviation from routine training."""
    if log.safety_signals:
        descs = [f"{s.severity} {s.signal_type} on {s.location_or_movement}" for s in log.safety_signals]
        return True, " / ".join(descs)
    surprise_keywords = ["sleep", "skipped", "travel", "plateau", "stuck", "couldn't",
                         "PR", "unexpected", "form", "compensat", "missed", "stalling",
                         "heavy", "twinge", "tight"]
    for ev in log.evidence:
        content_lower = ev.content.lower()
        for kw in surprise_keywords:
            if kw in content_lower:
                return True, f"evidence: '{ev.content[:80]}...'" if len(ev.content) > 80 else f"evidence: '{ev.content}'"
    return False, "routine"


async def run_one_session(
    state: AthleteState,
    log: SessionLog,
    prior_handoff: str,
) -> tuple[SessionResult | HaltedSession, StrawCoachOutput, EvaluatorReport, EvaluatorReport]:
    """Run both systems on one (state, log) pair and evaluate both."""
    graph = build_graph()

    # ----- agentic graph with file persistence -----
    trace_dir = TRACES_ROOT / state.athlete_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_file = trace_dir / f"session_{log.session_index}.json"
    persistence = FileStatePersistence(trace_file)
    persistence.set_graph_types(graph)

    session_state = SessionState(
        athlete_state=state,
        session_log=log,
        prior_handoff=prior_handoff,
    )
    async with graph.iter(
        IngestNode(),
        state=session_state,
        persistence=persistence,
    ) as run:
        async for _node in run:
            pass
        agentic_output = run.result.output

    # ----- straw coach (synchronous, deterministic) -----
    straw_output = run_straw_session(state, log)

    # ----- evaluator on both -----
    agentic_report, straw_report = await asyncio.gather(
        evaluate_session_output(state, log, agentic_output),
        evaluate_session_output(state, log, straw_output),
    )

    return agentic_output, straw_output, agentic_report, straw_report


async def run_athlete_arc(
    athlete_id: str,
    initial_state: AthleteState,
    sessions: list[SessionLog],
) -> list[SessionComparison]:
    """Run the full 6-session arc for one athlete."""
    state = initial_state
    prior_handoff = ""
    comparisons: list[SessionComparison] = []

    for i, log in enumerate(sessions):
        if i > 0:
            state = evolve_state(state, sessions[i - 1])

        is_surprise, surprise_summary = _surprise_summary(log)
        agentic_output, _straw_output, agentic_report, straw_report = await run_one_session(
            state, log, prior_handoff,
        )

        agentic_halted = isinstance(agentic_output, HaltedSession)
        agentic_used_fallback = (
            isinstance(agentic_output, SessionResult) and agentic_output.used_fallback_plan
        )

        comparisons.append(SessionComparison(
            athlete_id=athlete_id,
            session_index=log.session_index,
            is_surprise=is_surprise,
            surprise_summary=surprise_summary,
            straw_report=straw_report,
            agentic_report=agentic_report,
            agentic_halted=agentic_halted,
            agentic_used_fallback=agentic_used_fallback,
        ))

        # Update prior_handoff for next session
        if isinstance(agentic_output, SessionResult):
            prior_handoff = agentic_output.handoff.model_dump_json(indent=2)
        else:
            prior_handoff = f"Previous session was halted: {agentic_output.reason}"

    return comparisons


# ====================================================================
# Reporting
# ====================================================================

def _format_axis_line(label: str, straw_score: int, agentic_score: int) -> str:
    delta = agentic_score - straw_score
    delta_str = f"+{delta}" if delta > 0 else str(delta)
    return f"  {label:30s}  straw {straw_score}  agentic {agentic_score}  ({delta_str})"


def print_athlete_section(athlete_id: str, name: str, comparisons: list[SessionComparison]):
    print(f"\n{'=' * 78}")
    print(f"ATHLETE: {athlete_id} ({name})")
    print(f"{'=' * 78}")
    for cmp in comparisons:
        marker = "⚠" if cmp.is_surprise else " "
        flag_parts = []
        if cmp.agentic_halted:
            flag_parts.append("AGENTIC HALTED")
        if cmp.agentic_used_fallback:
            flag_parts.append("AGENTIC FALLBACK")
        flag = f"  [{', '.join(flag_parts)}]" if flag_parts else ""
        print(f"\n  {marker} Session {cmp.session_index}: {cmp.surprise_summary}{flag}")
        print(_format_axis_line("Plan Quality",
                                cmp.straw_report.plan_quality.score,
                                cmp.agentic_report.plan_quality.score))
        print(_format_axis_line("Coaching Specificity",
                                cmp.straw_report.coaching_specificity.score,
                                cmp.agentic_report.coaching_specificity.score))
        print(_format_axis_line("Adaptation Appropriateness",
                                cmp.straw_report.adaptation_appropriateness.score,
                                cmp.agentic_report.adaptation_appropriateness.score))
        print(_format_axis_line("Safety Adherence",
                                cmp.straw_report.safety_adherence.score,
                                cmp.agentic_report.safety_adherence.score))
        print(f"  {'Aggregate':30s}  straw {cmp.straw_report.aggregate:.2f}  "
              f"agentic {cmp.agentic_report.aggregate:.2f}  "
              f"({cmp.agentic_report.aggregate - cmp.straw_report.aggregate:+.2f})")


def print_summary(all_comparisons: list[SessionComparison]):
    n_total = len(all_comparisons)
    n_surprise = sum(1 for c in all_comparisons if c.is_surprise)
    n_routine = n_total - n_surprise

    straw_total = sum(c.straw_report.aggregate for c in all_comparisons) / n_total
    agentic_total = sum(c.agentic_report.aggregate for c in all_comparisons) / n_total

    surprises = [c for c in all_comparisons if c.is_surprise]
    straw_surprise = sum(c.straw_report.aggregate for c in surprises) / max(1, n_surprise)
    agentic_surprise = sum(c.agentic_report.aggregate for c in surprises) / max(1, n_surprise)

    routines = [c for c in all_comparisons if not c.is_surprise]
    straw_routine = sum(c.straw_report.aggregate for c in routines) / max(1, n_routine)
    agentic_routine = sum(c.agentic_report.aggregate for c in routines) / max(1, n_routine)

    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    print(f"All sessions ({n_total}):")
    print(f"  straw   aggregate: {straw_total:.2f} / 5.0")
    print(f"  agentic aggregate: {agentic_total:.2f} / 5.0  ({agentic_total - straw_total:+.2f})")
    print()
    print(f"Surprise sessions ({n_surprise}/{n_total}) — where the difference matters most:")
    print(f"  straw   aggregate: {straw_surprise:.2f} / 5.0")
    print(f"  agentic aggregate: {agentic_surprise:.2f} / 5.0  ({agentic_surprise - straw_surprise:+.2f})")
    print()
    print(f"Routine sessions ({n_routine}/{n_total}):")
    print(f"  straw   aggregate: {straw_routine:.2f} / 5.0")
    print(f"  agentic aggregate: {agentic_routine:.2f} / 5.0  ({agentic_routine - straw_routine:+.2f})")
    print()
    print(f"Traces persisted to: {TRACES_ROOT}")


# ====================================================================
# Entry point
# ====================================================================

async def main_async(
    athlete_filter: list[str] | None = None,
    clean_traces: bool = True,
):
    print("=" * 78)
    print("FITNESS COACH TOY — agentic cognitive core vs. rigid straw coach")
    print("=" * 78)
    print()
    print("Architecture: pydantic-graph FSM + DSPy signatures + PydanticAI agents")
    print("              + deterministic methodology validators + safety overrides")
    print()

    if clean_traces and TRACES_ROOT.exists():
        # Wipe trace dir before run — FileStatePersistence appends, so stale
        # snapshots from earlier runs would otherwise pollute the JSON.
        shutil.rmtree(TRACES_ROOT, ignore_errors=True)
        TRACES_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"Cleaned traces directory: {TRACES_ROOT}")
        print()

    print("State graph:")
    print(build_graph().mermaid_code())
    print()

    targets = athlete_filter or list(ATHLETES.keys())
    all_comparisons: list[SessionComparison] = []

    for athlete_id in targets:
        if athlete_id not in ATHLETES:
            print(f"Skipping unknown athlete: {athlete_id}")
            continue
        initial_state, sessions = ATHLETES[athlete_id]
        comparisons = await run_athlete_arc(athlete_id, initial_state, sessions)
        print_athlete_section(athlete_id, initial_state.name, comparisons)
        all_comparisons.extend(comparisons)

    print_summary(all_comparisons)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--athletes", nargs="*", default=None,
                        help="Restrict to specific athlete IDs (e.g. pl_002 rn_001). "
                             "Default: all athletes.")
    parser.add_argument("--no-clean-traces", action="store_true",
                        help="Skip wiping fitness_coach/traces/ before the run. "
                             "Default behavior cleans the traces dir first to avoid "
                             "FileStatePersistence appending to stale files.")
    args = parser.parse_args()
    asyncio.run(main_async(args.athletes, clean_traces=not args.no_clean_traces))


if __name__ == "__main__":
    main()
