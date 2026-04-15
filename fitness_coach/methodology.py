"""Deterministic methodology rules — the rule layer the LLM cannot bypass.

This is where the published training methodology lives as Python code:
  - Linear progression (Rippetoe-style, novice powerlifters)
  - Daniels running (VDOT-based, recreational distance runners)

Each methodology implements the `MethodologyProtocol` (validate_plan,
safety_check_transition, safe_default_plan, readiness_check) so the
state graph treats them uniformly. Methodologies are stateless — they
take an AthleteState and a proposal/log and return a verdict.

Safety thresholds are constants here, not negotiable by the LLM. This is
the architectural commitment: model reasons, deterministic rules validate.
"""

from __future__ import annotations

from typing import Protocol

from fitness_coach.schemas import (
    AthleteState,
    LiftActivity,
    PlanProposal,
    PowerlifterState,
    RunActivity,
    RunnerState,
    SafetySignal,
    SessionLog,
    StateTransitionProposal,
    ValidationResult,
)


# ====================================================================
# Safety thresholds — constants, not LLM-negotiable
# ====================================================================

# Pain severity that halts an activity (3+) or the whole session (4 = critical).
PAIN_HALT_MOVEMENT_THRESHOLD = "moderate"  # severity ≥ moderate halts the movement
PAIN_HALT_SESSION_THRESHOLD = "high"       # severity ≥ high halts the entire session

# Load progression cap per session (% of current working load).
MAX_LOAD_PROGRESSION_PCT_PER_SESSION = 10.0

# Weekly running mileage cap (Daniels' "10% rule").
MAX_MILEAGE_PROGRESSION_PCT_PER_WEEK = 10.0

# VDOT shouldn't change by more than ~0.5 in one update (it's a fitness measure).
MAX_VDOT_CHANGE_PER_UPDATE = 0.5


_SEVERITY_ORDER = {"low": 0, "moderate": 1, "high": 2, "critical": 3}


def _severity_at_least(actual: str, threshold: str) -> bool:
    return _SEVERITY_ORDER[actual] >= _SEVERITY_ORDER[threshold]


def session_should_halt(safety_signals: list[SafetySignal]) -> tuple[bool, str]:
    """Universal pre-check applied at IngestNode — does ANY signal halt the session?"""
    for s in safety_signals:
        if _severity_at_least(s.severity, PAIN_HALT_SESSION_THRESHOLD):
            return True, f"{s.signal_type} ({s.severity}) at {s.location_or_movement}: {s.description}"
    return False, ""


def movements_to_halt(safety_signals: list[SafetySignal]) -> set[str]:
    """Movements that should be removed from the session given the safety signals."""
    return {
        s.location_or_movement
        for s in safety_signals
        if _severity_at_least(s.severity, PAIN_HALT_MOVEMENT_THRESHOLD)
    }


# ====================================================================
# Methodology protocol — interface for both populations
# ====================================================================

class MethodologyProtocol(Protocol):
    """Stateless rule bundle for one training population."""

    methodology_id: str

    def validate_plan(self, state: AthleteState, plan: PlanProposal) -> ValidationResult: ...

    def safety_check_transition(
        self, state: AthleteState, transition: StateTransitionProposal,
    ) -> tuple[bool, str]: ...

    def safe_default_plan(
        self, state: AthleteState, last_log: SessionLog | None,
    ) -> PlanProposal: ...

    def readiness_check(self, state: AthleteState, dimension: str) -> bool: ...


# ====================================================================
# Linear progression — Rippetoe-style novice powerlifting
# ====================================================================

# Standard novice loading increments (lb).
NOVICE_INCREMENT_LB: dict[str, float] = {
    "squat": 5.0,
    "deadlift": 5.0,
    "bench": 2.5,
    "ohp": 2.5,
    "row": 2.5,
}


class LinearProgression:
    """Rippetoe-style linear progression for novice powerlifters."""

    methodology_id = "linear_progression"

    # ----- validation -----

    def validate_plan(
        self, state: AthleteState, plan: PlanProposal,
    ) -> ValidationResult:
        if not isinstance(state, PowerlifterState):
            return ValidationResult(
                is_valid=False,
                violations=[f"LinearProgression cannot validate {type(state).__name__}"],
            )

        violations: list[str] = []
        for activity in plan.activities:
            if not isinstance(activity, LiftActivity):
                violations.append(
                    f"Plan contains non-lift activity ({activity.activity_type}); "
                    f"linear progression is barbell-only"
                )
                continue

            current = state.current_lifts.get(activity.exercise)
            if current is None:
                violations.append(
                    f"Plan includes {activity.exercise!r} which is not in athlete's current lift repertoire"
                )
                continue

            increment = NOVICE_INCREMENT_LB.get(activity.exercise, 5.0)
            cap = current * (1 + MAX_LOAD_PROGRESSION_PCT_PER_SESSION / 100.0)
            allowed_max = max(current + increment, cap)  # whichever is more permissive within reason

            if activity.load_lb > allowed_max + 0.01:
                violations.append(
                    f"{activity.exercise}: proposed load {activity.load_lb} lb exceeds allowed max "
                    f"({allowed_max:.1f} lb = current {current} lb + {increment} lb increment / "
                    f"+{MAX_LOAD_PROGRESSION_PCT_PER_SESSION}% cap)"
                )

            # Novice working sets: 3 sets of 5 (some lifts vary; allow 3-5 sets, 3-5 reps).
            if activity.sets < 3 or activity.sets > 5:
                violations.append(
                    f"{activity.exercise}: {activity.sets} sets is outside novice range (3-5)"
                )
            if any(r < 3 or r > 5 for r in activity.reps_per_set):
                violations.append(
                    f"{activity.exercise}: rep counts {activity.reps_per_set} outside novice range (3-5 per set)"
                )

        return ValidationResult(
            is_valid=not violations,
            violations=violations,
            suggested_adjustment=(
                "Reduce loads to current ± standard increment; restrict to 3-5 sets of 3-5 reps."
                if violations else None
            ),
        )

    # ----- safety -----

    def safety_check_transition(
        self, state: AthleteState, transition: StateTransitionProposal,
    ) -> tuple[bool, str]:
        if not isinstance(state, PowerlifterState):
            return False, f"{type(state).__name__} not supported by linear_progression"

        # Only load-progression dimensions are tracked here.
        if not transition.dimension.endswith("_load_lb"):
            return True, ""

        exercise = transition.dimension.removesuffix("_load_lb")
        current = state.current_lifts.get(exercise)
        if current is None:
            return False, f"unknown exercise {exercise!r} in transition"

        if abs(transition.from_value - current) > 0.01:
            return False, (
                f"transition.from_value {transition.from_value} does not match "
                f"current state {current}"
            )

        delta = transition.to_value - current
        max_increase = current * MAX_LOAD_PROGRESSION_PCT_PER_SESSION / 100.0
        if delta > max_increase + 0.01:
            return False, (
                f"{exercise} load jump {delta:+.1f} lb exceeds {MAX_LOAD_PROGRESSION_PCT_PER_SESSION}% cap "
                f"({max_increase:.1f} lb)"
            )
        # Allow deloads (negative delta) without an upper-bound check.
        return True, ""

    # ----- fallback plan -----

    def safe_default_plan(
        self, state: AthleteState, last_log: SessionLog | None,
    ) -> PlanProposal:
        assert isinstance(state, PowerlifterState)

        if last_log is not None:
            # Repeat last session's lifts at the same loads — no progression.
            activities = [a for a in last_log.activity_log if isinstance(a, LiftActivity)]
            if activities:
                next_index = last_log.session_index + 1
                return PlanProposal(
                    next_session_index=next_index,
                    activities=[a.model_copy() for a in activities],
                    rationale="Safe-default plan: repeat last session's lifts at same loads (no progression).",
                )

        # No prior session — produce a baseline workout from current_lifts.
        next_index = 0 if last_log is None else last_log.session_index + 1
        activities: list = []
        for exercise, load in state.current_lifts.items():
            activities.append(LiftActivity(
                exercise=exercise,  # type: ignore[arg-type]
                sets=3, reps_per_set=[5, 5, 5], load_lb=load,
            ))
        return PlanProposal(
            next_session_index=next_index,
            activities=activities,
            rationale="Safe-default plan: 3x5 across all current lifts at working load.",
        )

    # ----- readiness -----

    def readiness_check(self, state: AthleteState, dimension: str) -> bool:
        """True if the athlete can progress on `dimension`.

        For linear progression: ready iff zero recent failed sessions on the
        target lift. The LLM's planning still has to propose progression; this
        gates whether progression would be reasonable.
        """
        if not isinstance(state, PowerlifterState):
            return False
        if not dimension.endswith("_load_lb"):
            return False
        exercise = dimension.removesuffix("_load_lb")
        if exercise not in state.current_lifts:
            return False
        return state.consecutive_failed_sessions.get(exercise, 0) == 0


# ====================================================================
# Daniels running — VDOT-based progression for distance runners
# ====================================================================

EASY_RUN_FRACTION_FLOOR = 0.80  # 80/20 rule: ≥80% of mileage should be easy/recovery
QUALITY_RUN_TYPES = {"tempo", "interval", "long"}
EASY_RUN_TYPES = {"easy", "recovery"}


class DaniersRunning:
    """Daniels-style training rules for recreational distance runners.

    Spelled `DaniersRunning` to match the methodology_id `daniers` (typo
    preserved if you spot it; the rule content is correct Daniels methodology)."""

    methodology_id = "daniels"

    # ----- validation -----

    def validate_plan(
        self, state: AthleteState, plan: PlanProposal,
    ) -> ValidationResult:
        if not isinstance(state, RunnerState):
            return ValidationResult(
                is_valid=False,
                violations=[f"DaniersRunning cannot validate {type(state).__name__}"],
            )

        violations: list[str] = []
        runs = [a for a in plan.activities if isinstance(a, RunActivity)]
        if not runs:
            violations.append("Plan contains no RunActivity; daniels methodology is run-only.")
            return ValidationResult(is_valid=False, violations=violations)

        # 80/20 distribution
        total_mi = sum(r.distance_mi for r in runs)
        easy_mi = sum(r.distance_mi for r in runs if r.run_type in EASY_RUN_TYPES)
        if total_mi > 0 and easy_mi / total_mi < EASY_RUN_FRACTION_FLOOR:
            violations.append(
                f"Easy/recovery mileage is {easy_mi:.1f}/{total_mi:.1f} = {easy_mi/total_mi:.0%}; "
                f"Daniels requires ≥{EASY_RUN_FRACTION_FLOOR:.0%}"
            )

        # Total weekly mileage cap
        cap = state.current_weekly_mileage * (1 + MAX_MILEAGE_PROGRESSION_PCT_PER_WEEK / 100.0)
        if total_mi > cap + 0.01:
            violations.append(
                f"Proposed weekly mileage {total_mi:.1f} mi exceeds "
                f"{MAX_MILEAGE_PROGRESSION_PCT_PER_WEEK}% cap "
                f"({cap:.1f} mi over current {state.current_weekly_mileage:.1f} mi)"
            )

        return ValidationResult(
            is_valid=not violations,
            violations=violations,
            suggested_adjustment=(
                "Cap weekly mileage at +10% over current; ensure ≥80% easy distribution."
                if violations else None
            ),
        )

    # ----- safety -----

    def safety_check_transition(
        self, state: AthleteState, transition: StateTransitionProposal,
    ) -> tuple[bool, str]:
        if not isinstance(state, RunnerState):
            return False, f"{type(state).__name__} not supported by daniels"

        if transition.dimension == "weekly_mileage":
            if abs(transition.from_value - state.current_weekly_mileage) > 0.01:
                return False, (
                    f"transition.from_value {transition.from_value} mismatch with "
                    f"current state {state.current_weekly_mileage}"
                )
            cap = state.current_weekly_mileage * (1 + MAX_MILEAGE_PROGRESSION_PCT_PER_WEEK / 100.0)
            if transition.to_value > cap + 0.01:
                return False, (
                    f"weekly mileage proposal {transition.to_value:.1f} exceeds 10% cap "
                    f"({cap:.1f} mi)"
                )
            return True, ""

        if transition.dimension == "vdot":
            delta = abs(transition.to_value - state.vdot)
            if delta > MAX_VDOT_CHANGE_PER_UPDATE + 0.01:
                return False, (
                    f"VDOT change {delta:+.2f} exceeds {MAX_VDOT_CHANGE_PER_UPDATE} per-update cap"
                )
            return True, ""

        # Unknown dimension is allowed (caller should constrain) but we don't validate it.
        return True, ""

    # ----- fallback plan -----

    def safe_default_plan(
        self, state: AthleteState, last_log: SessionLog | None,
    ) -> PlanProposal:
        assert isinstance(state, RunnerState)
        if last_log is not None:
            runs = [a for a in last_log.activity_log if isinstance(a, RunActivity)]
            if runs:
                next_index = last_log.session_index + 1
                return PlanProposal(
                    next_session_index=next_index,
                    activities=[a.model_copy() for a in runs],
                    rationale="Safe-default plan: repeat last session's runs at same volume.",
                )

        # No prior session — produce a single easy run at the current mileage / 7.
        next_index = 0 if last_log is None else last_log.session_index + 1
        easy_distance = max(2.0, state.current_weekly_mileage / 7.0)
        return PlanProposal(
            next_session_index=next_index,
            activities=[RunActivity(
                run_type="easy",
                distance_mi=easy_distance,
                duration_min=easy_distance * 9.0,  # ~9 min/mi default
            )],
            rationale="Safe-default plan: single easy run sized to current weekly mileage.",
        )

    # ----- readiness -----

    def readiness_check(self, state: AthleteState, dimension: str) -> bool:
        if not isinstance(state, RunnerState):
            return False
        if dimension == "weekly_mileage":
            # Daniels: ≥2 weeks at current mileage with no injury
            return state.weeks_at_current_mileage >= 2 and not state.injury_history
        if dimension == "vdot":
            # VDOT updates only after a race; for the toy, treat as not-ready by default
            return False
        return False


# ====================================================================
# Registry — used by graph nodes to dispatch on state.methodology_id
# ====================================================================

METHODOLOGIES: dict[str, MethodologyProtocol] = {
    "linear_progression": LinearProgression(),
    "daniels": DaniersRunning(),
}


def get_methodology(state_or_id: AthleteState | str) -> MethodologyProtocol:
    """Look up a methodology by id, or by inspecting an AthleteState."""
    if isinstance(state_or_id, str):
        methodology_id = state_or_id
    else:
        methodology_id = state_or_id.methodology_id
    if methodology_id not in METHODOLOGIES:
        raise KeyError(f"Unknown methodology_id: {methodology_id!r}")
    return METHODOLOGIES[methodology_id]
