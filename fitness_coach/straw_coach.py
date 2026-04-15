"""Rigid expert-system "straw coach" — the comparison baseline.

What it does:
  - Mechanically advances the methodology's prescribed loads/mileage
  - Has no awareness of evidence beyond a session counter and (for lifts)
    a missed-rep counter
  - Ignores pain reports, sleep quality, life events, form regression — by
    design

Why this design: the demo's whole point is to show the agentic architecture
handling situations that rule-based logic cannot. A fail-graceful straw coach
that "kind of" handles surprises would muddy the comparison. So this one
fails dramatically — it will prescribe a load increase on a session where
the athlete reported moderate joint pain, and it will keep ramping mileage
through ITB warnings.

This file is pure deterministic Python — no LLM calls, no I/O.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from fitness_coach.methodology import (
    NOVICE_INCREMENT_LB,
    MAX_MILEAGE_PROGRESSION_PCT_PER_WEEK,
)
from fitness_coach.schemas import (
    AthleteState,
    CoachingOutput,
    LiftActivity,
    PlanProposal,
    PowerlifterState,
    RunActivity,
    RunnerState,
    SessionLog,
    StateTransitionProposal,
)


# Number of consecutive failed sessions on a lift before straw coach deloads.
DELOAD_AFTER_N_FAILURES = 3
# Deload amount as a fraction of current load.
DELOAD_FRACTION = 0.10


class StrawCoachOutput(BaseModel):
    """What the straw coach produces for one session — parallel structure to
    the agentic SessionResult for evaluator comparison."""

    athlete_id: str
    session_index: int
    plan: PlanProposal
    coaching: CoachingOutput
    proposed_state_transition: StateTransitionProposal | None = Field(
        default=None,
        description="The straw coach proposes one transition per session — usually load/mileage progression",
    )
    handoff_summary: str = Field(
        description="Brief stand-in for HandoffDoc — straw coach has no handoff concept",
    )
    generated_at: datetime


# ====================================================================
# Powerlifter rules
# ====================================================================

def _last_set_target_reps(activity: LiftActivity) -> bool:
    """Did the athlete hit all reps in their last set?

    Used as a 'success' check. Straw coach interpretation: target was 5 reps
    per set; failure if final set was below 5.
    """
    if not activity.reps_per_set:
        return False
    return activity.reps_per_set[-1] >= 5


def _powerlifter_session(
    state: PowerlifterState, last_log: SessionLog,
) -> StrawCoachOutput:
    next_index = last_log.session_index + 1
    activities: list = []
    transitions: list[StateTransitionProposal] = []

    for prior in last_log.activity_log:
        if not isinstance(prior, LiftActivity):
            continue

        current = state.current_lifts.get(prior.exercise, prior.load_lb)
        increment = NOVICE_INCREMENT_LB.get(prior.exercise, 5.0)

        # Was last session a "failure"?
        failed = not _last_set_target_reps(prior)
        consecutive_fails = state.consecutive_failed_sessions.get(prior.exercise, 0)
        if failed:
            consecutive_fails += 1

        if consecutive_fails >= DELOAD_AFTER_N_FAILURES:
            # Mechanical deload — only triggered after threshold
            new_load = round(current * (1 - DELOAD_FRACTION) / 2.5) * 2.5
            rationale = f"Deload after {consecutive_fails} consecutive failures"
        elif failed:
            # Hold load on failure
            new_load = current
            rationale = f"Hold load (failure {consecutive_fails}/{DELOAD_AFTER_N_FAILURES})"
        else:
            # Standard progression
            new_load = current + increment
            rationale = f"Standard +{increment} progression"

        activities.append(LiftActivity(
            exercise=prior.exercise,
            sets=3, reps_per_set=[5, 5, 5],
            load_lb=new_load,
        ))

        if abs(new_load - current) > 0.01:
            transitions.append(StateTransitionProposal(
                dimension=f"{prior.exercise}_load_lb",
                from_value=current, to_value=new_load,
                rationale=rationale,
                confidence="high",  # rule-based, always certain
            ))

    plan = PlanProposal(
        next_session_index=next_index,
        activities=activities,
        rationale="Linear progression: +increment per lift, deload after 3 consecutive failures",
    )

    coaching = CoachingOutput(
        cues=[
            "Brace hard before each rep.",
            "Keep bar path consistent.",
            "Hit your numbers — execute the prescribed loads.",
        ],
        watch_for=["Missed reps"],
        rationale="Generic novice-progression cues",
    )

    # Straw coach proposes the single most-impactful transition
    proposed = transitions[0] if transitions else None

    return StrawCoachOutput(
        athlete_id=state.athlete_id,
        session_index=next_index,
        plan=plan,
        coaching=coaching,
        proposed_state_transition=proposed,
        handoff_summary=f"Session {next_index}: progress per linear progression rules.",
        generated_at=datetime.now(timezone.utc),
    )


# ====================================================================
# Runner rules
# ====================================================================

def _runner_session(
    state: RunnerState, last_log: SessionLog,
) -> StrawCoachOutput:
    next_index = last_log.session_index + 1
    prior_runs = [a for a in last_log.activity_log if isinstance(a, RunActivity)]
    prior_total = sum(r.distance_mi for r in prior_runs) if prior_runs else state.current_weekly_mileage

    # Rigid +10% mileage progression — applied uniformly
    new_total = prior_total * (1 + MAX_MILEAGE_PROGRESSION_PCT_PER_WEEK / 100.0)
    scale = new_total / prior_total if prior_total > 0 else 1.0

    activities: list = []
    if prior_runs:
        for r in prior_runs:
            activities.append(RunActivity(
                run_type=r.run_type,
                distance_mi=round(r.distance_mi * scale, 1),
                duration_min=round(r.duration_min * scale, 1),
            ))
    else:
        # No prior runs to model — just prescribe one easy run at current mileage
        activities.append(RunActivity(
            run_type="easy", distance_mi=state.current_weekly_mileage / 7.0,
            duration_min=(state.current_weekly_mileage / 7.0) * 9.5,
        ))

    plan = PlanProposal(
        next_session_index=next_index,
        activities=activities,
        rationale=f"Daniels +10%/week mileage progression: {prior_total:.1f} → {new_total:.1f} mi",
    )

    coaching = CoachingOutput(
        cues=[
            "Hit your easy paces — don't run them too hard.",
            "Maintain cadence around 175-180.",
            "Stay relaxed in the upper body.",
        ],
        watch_for=["Pace creeping up on easy runs"],
        rationale="Generic Daniels distance-running cues",
    )

    proposed = StateTransitionProposal(
        dimension="weekly_mileage",
        from_value=state.current_weekly_mileage,
        to_value=round(new_total, 1),
        rationale="Standard +10% weekly progression",
        confidence="high",
    )

    return StrawCoachOutput(
        athlete_id=state.athlete_id,
        session_index=next_index,
        plan=plan,
        coaching=coaching,
        proposed_state_transition=proposed,
        handoff_summary=f"Week {next_index}: +10% mileage per Daniels rule.",
        generated_at=datetime.now(timezone.utc),
    )


# ====================================================================
# Public dispatch
# ====================================================================

def run_straw_session(
    state: AthleteState, last_log: SessionLog,
) -> StrawCoachOutput:
    """Apply the rigid coach to one (state, last_session_log) pair.

    NOTE: the straw coach makes no use of:
      - last_log.evidence (sleep, RPE narrative, form check observations)
      - last_log.safety_signals (pain reports, injury recurrences)
    By design — that's what makes it the comparison baseline.
    """
    if isinstance(state, PowerlifterState):
        return _powerlifter_session(state, last_log)
    if isinstance(state, RunnerState):
        return _runner_session(state, last_log)
    raise TypeError(f"Unsupported state type: {type(state).__name__}")
