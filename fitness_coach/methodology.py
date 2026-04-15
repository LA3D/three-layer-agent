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

    def validate_plan(
        self,
        state: AthleteState,
        plan: PlanProposal,
        halted_movements: frozenset[str] | set[str] = frozenset(),
    ) -> ValidationResult: ...

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

# Rippetoe deload trigger: after this many consecutive failed sessions on a lift,
# methodology requires a deload before any progression or hold is permitted.
DELOAD_TRIGGER_CONSECUTIVE_FAILURES = 3
DELOAD_PCT = 10.0


def _build_linear_suggestion(violations: list[str]) -> str:
    """Tailor a `suggested_adjustment` to the actual violations.

    Replaces the previous canned message — gives the LLM specific actionable
    guidance for the retry attempt rather than a generic reminder.
    """
    parts: list[str] = []
    if any("consecutive failed sessions require a deload" in v for v in violations):
        parts.append(
            f"At least one lift requires a deload (≥{DELOAD_TRIGGER_CONSECUTIVE_FAILURES} "
            f"consecutive failures): reduce affected loads by {DELOAD_PCT:.0f}%"
        )
    if any("exceeds allowed max" in v for v in violations):
        parts.append(
            "Cap loads at current + standard increment (squat/deadlift +5 lb, bench/ohp +2.5 lb)"
        )
    if any("non-lift activity" in v for v in violations):
        parts.append("Linear progression is barbell-only; drop non-lift activities")
    if any("not in athlete's current lift repertoire" in v for v in violations):
        parts.append("Restrict exercises to those in the athlete's current_lifts")
    if any("sets is outside" in v or "rep counts" in v for v in violations):
        parts.append("Use 3-5 sets of 3-5 reps per lift")
    if any("halted for this session" in v for v in violations):
        parts.append("Drop or substitute halted movements; do not include them in the plan")
    return "; ".join(parts) or "Address the violations listed above"


class LinearProgression:
    """Rippetoe-style linear progression for novice powerlifters."""

    methodology_id = "linear_progression"

    # ----- validation -----

    def validate_plan(
        self,
        state: AthleteState,
        plan: PlanProposal,
        halted_movements: frozenset[str] | set[str] = frozenset(),
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

            # Halted movements: any signal of moderate+ severity on this lift
            # removes the lift from this session's eligible plan.
            if activity.exercise in halted_movements:
                violations.append(
                    f"{activity.exercise}: halted for this session due to safety signal "
                    f"(moderate+ pain reported); drop or substitute"
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

            # Novice working sets: per Rippetoe SS, deadlift is 1x5; squat / bench /
            # ohp / row are 3x5. Allow 1 set for deadlift only; 3-5 for others.
            min_sets = 1 if activity.exercise == "deadlift" else 3
            if activity.sets < min_sets or activity.sets > 5:
                violations.append(
                    f"{activity.exercise}: {activity.sets} sets is outside novice range "
                    f"({min_sets}-5)"
                )
            if any(r < 3 or r > 5 for r in activity.reps_per_set):
                violations.append(
                    f"{activity.exercise}: rep counts {activity.reps_per_set} outside novice range (3-5 per set)"
                )

            # Rippetoe deload rule: ≥3 consecutive failures on a lift requires a deload
            # before any further progression or hold is permitted.
            fail_count = state.consecutive_failed_sessions.get(activity.exercise, 0)
            if fail_count >= DELOAD_TRIGGER_CONSECUTIVE_FAILURES:
                deload_max = current * (1 - DELOAD_PCT / 100.0)
                if activity.load_lb > deload_max + 0.01:
                    violations.append(
                        f"{activity.exercise}: {fail_count} consecutive failed sessions require a "
                        f"deload to ≤{deload_max:.1f} lb (Rippetoe rule: deload {DELOAD_PCT:.0f}% "
                        f"after {DELOAD_TRIGGER_CONSECUTIVE_FAILURES} failures, current {current} lb)"
                    )

        return ValidationResult(
            is_valid=not violations,
            violations=violations,
            suggested_adjustment=_build_linear_suggestion(violations) if violations else None,
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
    #
    # `safe_default_plan` is invoked by FallbackPlanNode after the LLM has
    # produced two consecutive validation-failing plans. The intent is "no
    # progression, no change" — repeat what the athlete just did, at the
    # same loads. This is the conservative choice for most stuck states:
    # if the LLM cannot produce a valid plan, do not introduce novelty.
    #
    # Note: a previous review flagged this as "could amplify problems"
    # (repeating a fatigued/painful session). That concern is real but
    # belongs upstream — the right fix is correct state evolution and the
    # halted_movements mechanism in validate_plan, not silently deloading
    # in the fallback. Halt-worthy signals are handled at IngestNode
    # (session-halt) and ValidateNode (movement-halt). What's left when
    # fallback fires is a plan-shape problem (e.g., LLM proposed too-
    # aggressive progression), and "do what they just did" is correct.

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
# Per Daniels: long runs are typically run at easy/easy-marathon pace, not at
# threshold/interval intensity. They count toward the easy mileage budget,
# not the quality budget. Quality work = tempo + interval only.
QUALITY_RUN_TYPES = {"tempo", "interval"}
EASY_RUN_TYPES = {"easy", "recovery", "long"}

# A runner "session" represents one training week, not a single workout.
# Daniels prescribes 4-6 runs per week in build phases; 3 is the minimum
# floor below which the schedule is no longer a coherent training week.
MIN_RUNS_PER_RUNNER_WEEK = 3


def _build_daniels_suggestion(violations: list[str]) -> str:
    """Tailor a `suggested_adjustment` for the Daniels methodology."""
    parts: list[str] = []
    if any("contains no RunActivity" in v for v in violations):
        parts.append("Daniels methodology is run-only; include RunActivity entries")
    if any("Easy/recovery mileage" in v for v in violations):
        parts.append(f"Maintain ≥{EASY_RUN_FRACTION_FLOOR:.0%} easy/recovery distribution (80/20 rule)")
    if any("exceeds" in v and "cap" in v for v in violations):
        parts.append(
            f"Cap weekly mileage at +{MAX_MILEAGE_PROGRESSION_PCT_PER_WEEK:.0f}% over current"
        )
    if any("at least" in v and "RunActivity" in v for v in violations):
        parts.append(
            f"A runner session is a full training week — include at least "
            f"{MIN_RUNS_PER_RUNNER_WEEK} distinct runs distributed across the week"
        )
    if any("halted for this session" in v for v in violations):
        parts.append("Drop or substitute halted activities")
    return "; ".join(parts) or "Address the violations listed above"


class DanielsRunning:
    """Daniels-style training rules for recreational distance runners."""

    methodology_id = "daniels"

    # ----- validation -----

    def validate_plan(
        self,
        state: AthleteState,
        plan: PlanProposal,
        halted_movements: frozenset[str] | set[str] = frozenset(),
    ) -> ValidationResult:
        if not isinstance(state, RunnerState):
            return ValidationResult(
                is_valid=False,
                violations=[f"DanielsRunning cannot validate {type(state).__name__}"],
            )

        violations: list[str] = []
        runs = [a for a in plan.activities if isinstance(a, RunActivity)]
        if not runs:
            violations.append("Plan contains no RunActivity; daniels methodology is run-only.")
            return ValidationResult(is_valid=False, violations=violations)

        # A runner session is a week, not a workout — must contain a coherent
        # training distribution, not a single mega-run.
        if len(runs) < MIN_RUNS_PER_RUNNER_WEEK:
            violations.append(
                f"Runner sessions represent a training week and must contain at least "
                f"{MIN_RUNS_PER_RUNNER_WEEK} distinct RunActivity entries; plan has {len(runs)}"
            )

        # Halted movements for runners: a "movement" can be the activity's run_type
        # (e.g., "long" halted due to ITB) OR the generic token "running" (full halt).
        for r in runs:
            if r.run_type in halted_movements or "running" in halted_movements:
                violations.append(
                    f"{r.run_type}: run type halted for this session due to safety signal; "
                    f"substitute with easy/recovery or omit"
                )

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
            suggested_adjustment=_build_daniels_suggestion(violations) if violations else None,
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
        """Produce a conservative all-easy training week sized to the athlete's
        baseline mileage.

        Important: the prior session log may be SPARSE (skipped week, illness)
        and therefore is NOT a reliable basis for repeating volume. We always
        derive the safe-default plan from `state.current_weekly_mileage` (the
        preserved baseline, see run.evolve_state). Distribute that mileage
        across 4 easy runs to satisfy the MIN_RUNS_PER_RUNNER_WEEK constraint
        and the 80/20 distribution.
        """
        assert isinstance(state, RunnerState)
        next_index = 0 if last_log is None else last_log.session_index + 1
        baseline = max(state.current_weekly_mileage, 4.0)  # never go below ~4 mi/wk

        # 4 easy runs, roughly equal distribution (with a slightly longer one).
        per_run = baseline / 4.0
        long_extra = baseline * 0.10  # ~10% bumped to the longer day
        runs: list = [
            RunActivity(run_type="easy", distance_mi=round(per_run - long_extra / 3, 1),
                        duration_min=round((per_run - long_extra / 3) * 9.0, 1)),
            RunActivity(run_type="easy", distance_mi=round(per_run - long_extra / 3, 1),
                        duration_min=round((per_run - long_extra / 3) * 9.0, 1)),
            RunActivity(run_type="recovery", distance_mi=round(per_run - long_extra / 3, 1),
                        duration_min=round((per_run - long_extra / 3) * 10.0, 1)),
            RunActivity(run_type="long", distance_mi=round(per_run + long_extra, 1),
                        duration_min=round((per_run + long_extra) * 9.0, 1)),
        ]
        return PlanProposal(
            next_session_index=next_index,
            activities=runs,
            rationale=(
                f"Safe-default plan: conservative all-easy week sized to baseline "
                f"({baseline:.1f} mi), distributed across 4 runs. Holds mileage "
                f"and avoids quality work."
            ),
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
    "daniels": DanielsRunning(),
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
