"""Unit tests for fitness_coach.methodology.

Methodology rules are deterministic Python — they have to be reliable
because the LLM cannot bypass them. Test exhaustively for the rules that
guard safety and the constraint validation that gates plan acceptance.
"""

from datetime import date

import pytest

from fitness_coach.methodology import (
    DaniersRunning,
    LinearProgression,
    METHODOLOGIES,
    get_methodology,
    movements_to_halt,
    session_should_halt,
)
from fitness_coach.schemas import (
    LiftActivity,
    PlanProposal,
    PowerlifterState,
    RunActivity,
    RunnerState,
    SafetySignal,
    SessionLog,
    StateTransitionProposal,
)


# ====================================================================
# Fixtures
# ====================================================================

def make_powerlifter(squat=225.0, bench=155.0, deadlift=275.0, ohp=95.0,
                     consecutive_fails=None) -> PowerlifterState:
    return PowerlifterState(
        athlete_id="pl_test", name="Test Lifter",
        training_age_months=6, bodyweight_lb=180.0,
        current_lifts={"squat": squat, "bench": bench, "deadlift": deadlift, "ohp": ohp},
        consecutive_failed_sessions=consecutive_fails or {},
    )


def make_runner(weekly=25.0, weeks_at=2, vdot=45.0, injuries=None) -> RunnerState:
    return RunnerState(
        athlete_id="rn_test", name="Test Runner",
        training_age_months=12, vdot=vdot,
        current_weekly_mileage=weekly,
        target_event="half_marathon",
        weeks_at_current_mileage=weeks_at,
        injury_history=injuries or [],
    )


# ====================================================================
# Universal safety pre-checks
# ====================================================================

class TestSafetyPreCheck:
    def test_high_severity_pain_halts_session(self):
        signals = [SafetySignal(severity="high", signal_type="pain",
                                location_or_movement="left knee", description="sharp")]
        halt, reason = session_should_halt(signals)
        assert halt
        assert "left knee" in reason

    def test_critical_severity_pain_halts_session(self):
        signals = [SafetySignal(severity="critical", signal_type="injury_recurrence",
                                location_or_movement="lower back", description="acute")]
        halt, _ = session_should_halt(signals)
        assert halt

    def test_moderate_severity_does_not_halt_session(self):
        signals = [SafetySignal(severity="moderate", signal_type="pain",
                                location_or_movement="shoulder", description="dull")]
        halt, _ = session_should_halt(signals)
        assert not halt

    def test_no_signals_no_halt(self):
        halt, _ = session_should_halt([])
        assert not halt

    def test_movements_to_halt_at_moderate_threshold(self):
        signals = [
            SafetySignal(severity="moderate", signal_type="pain",
                         location_or_movement="left knee", description="x"),
            SafetySignal(severity="low", signal_type="pain",
                         location_or_movement="right shoulder", description="y"),
        ]
        movs = movements_to_halt(signals)
        assert "left knee" in movs
        assert "right shoulder" not in movs


# ====================================================================
# Linear progression — validate_plan
# ====================================================================

class TestLinearProgressionValidate:
    def setup_method(self):
        self.method = LinearProgression()
        self.state = make_powerlifter()

    def test_valid_plan_within_increment(self):
        plan = PlanProposal(
            next_session_index=1,
            activities=[
                LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=230.0),
                LiftActivity(exercise="bench", sets=3, reps_per_set=[5, 5, 5], load_lb=157.5),
            ],
            rationale="standard novice progression",
        )
        result = self.method.validate_plan(self.state, plan)
        assert result.is_valid, result.violations

    def test_load_jump_too_large_rejected(self):
        plan = PlanProposal(
            next_session_index=1,
            activities=[LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=300.0)],
            rationale="aggressive jump",
        )
        result = self.method.validate_plan(self.state, plan)
        assert not result.is_valid
        assert any("squat" in v for v in result.violations)

    def test_unknown_exercise_rejected(self):
        plan = PlanProposal(
            next_session_index=1,
            activities=[LiftActivity(exercise="row", sets=3, reps_per_set=[5, 5, 5], load_lb=100.0)],
            rationale="add row",
        )
        result = self.method.validate_plan(self.state, plan)
        assert not result.is_valid
        assert any("row" in v for v in result.violations)

    def test_running_activity_in_lift_plan_rejected(self):
        plan = PlanProposal(
            next_session_index=1,
            activities=[RunActivity(run_type="easy", distance_mi=3.0, duration_min=27.0)],
            rationale="extra cardio",
        )
        result = self.method.validate_plan(self.state, plan)
        assert not result.is_valid
        assert any("non-lift" in v for v in result.violations)

    def test_too_many_sets_rejected(self):
        plan = PlanProposal(
            next_session_index=1,
            activities=[LiftActivity(exercise="squat", sets=8, reps_per_set=[5]*8, load_lb=225.0)],
            rationale="more volume",
        )
        result = self.method.validate_plan(self.state, plan)
        assert not result.is_valid

    def test_high_reps_rejected(self):
        plan = PlanProposal(
            next_session_index=1,
            activities=[LiftActivity(exercise="squat", sets=3, reps_per_set=[10, 10, 10], load_lb=180.0)],
            rationale="hypertrophy block",
        )
        result = self.method.validate_plan(self.state, plan)
        assert not result.is_valid


# ====================================================================
# Linear progression — safety_check_transition
# ====================================================================

class TestLinearSafetyTransition:
    def setup_method(self):
        self.method = LinearProgression()
        self.state = make_powerlifter(squat=225.0)

    def test_normal_increment_passes(self):
        t = StateTransitionProposal(
            dimension="squat_load_lb", from_value=225.0, to_value=230.0,
            rationale="hit RPE", confidence="medium",
        )
        ok, _ = self.method.safety_check_transition(self.state, t)
        assert ok

    def test_excessive_jump_rejected(self):
        t = StateTransitionProposal(
            dimension="squat_load_lb", from_value=225.0, to_value=275.0,
            rationale="big jump", confidence="high",
        )
        ok, msg = self.method.safety_check_transition(self.state, t)
        assert not ok
        assert "exceeds" in msg

    def test_deload_allowed(self):
        t = StateTransitionProposal(
            dimension="squat_load_lb", from_value=225.0, to_value=200.0,
            rationale="deload", confidence="high",
        )
        ok, _ = self.method.safety_check_transition(self.state, t)
        assert ok

    def test_from_value_mismatch_rejected(self):
        t = StateTransitionProposal(
            dimension="squat_load_lb", from_value=300.0, to_value=305.0,
            rationale="mismatch", confidence="medium",
        )
        ok, msg = self.method.safety_check_transition(self.state, t)
        assert not ok
        assert "from_value" in msg


# ====================================================================
# Linear progression — safe_default_plan and readiness
# ====================================================================

class TestLinearFallback:
    def test_safe_default_repeats_last_session(self):
        method = LinearProgression()
        state = make_powerlifter(squat=225.0)
        last = SessionLog(
            athlete_id="pl_test", session_index=2, session_date=date(2026, 4, 15),
            activity_log=[
                LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=225.0),
            ],
            evidence=[],
        )
        plan = method.safe_default_plan(state, last)
        assert plan.next_session_index == 3
        assert len(plan.activities) == 1
        assert plan.activities[0].load_lb == 225.0  # type: ignore[union-attr]

    def test_safe_default_with_no_prior_session(self):
        method = LinearProgression()
        state = make_powerlifter()
        plan = method.safe_default_plan(state, None)
        assert plan.next_session_index == 0
        assert len(plan.activities) == 4  # squat/bench/deadlift/ohp

    def test_readiness_blocked_by_recent_failure(self):
        method = LinearProgression()
        state = make_powerlifter(consecutive_fails={"squat": 1})
        assert method.readiness_check(state, "squat_load_lb") is False
        assert method.readiness_check(state, "bench_load_lb") is True

    def test_readiness_unknown_dimension(self):
        method = LinearProgression()
        state = make_powerlifter()
        assert method.readiness_check(state, "vdot") is False


# ====================================================================
# Daniels running — validate_plan
# ====================================================================

class TestDaniersValidate:
    def setup_method(self):
        self.method = DaniersRunning()
        self.state = make_runner(weekly=30.0)

    def test_valid_plan_under_cap(self):
        plan = PlanProposal(
            next_session_index=1,
            activities=[
                RunActivity(run_type="easy", distance_mi=6.0, duration_min=54.0),
                RunActivity(run_type="easy", distance_mi=6.0, duration_min=54.0),
                RunActivity(run_type="easy", distance_mi=4.0, duration_min=36.0),
                RunActivity(run_type="recovery", distance_mi=3.0, duration_min=30.0),
                RunActivity(run_type="long", distance_mi=8.0, duration_min=72.0),
                RunActivity(run_type="tempo", distance_mi=4.0, duration_min=28.0),
            ],
            rationale="balanced 80/20 week ~31 mi",
        )
        result = self.method.validate_plan(self.state, plan)
        # 31 mi total, easy 19/31 = 61% — fails 80/20
        assert not result.is_valid
        assert any("Easy" in v for v in result.violations)

    def test_pure_easy_under_cap_valid(self):
        plan = PlanProposal(
            next_session_index=1,
            activities=[
                RunActivity(run_type="easy", distance_mi=8.0, duration_min=72.0),
                RunActivity(run_type="easy", distance_mi=8.0, duration_min=72.0),
                RunActivity(run_type="easy", distance_mi=8.0, duration_min=72.0),
                RunActivity(run_type="recovery", distance_mi=8.0, duration_min=72.0),
            ],
            rationale="recovery week — all easy/recovery",
        )
        # 32 mi total, 100% easy/recovery → satisfies 80/20; 32 < 30 * 1.10 = 33
        result = self.method.validate_plan(self.state, plan)
        assert result.is_valid, result.violations

    def test_mileage_over_cap_rejected(self):
        plan = PlanProposal(
            next_session_index=1,
            activities=[
                RunActivity(run_type="easy", distance_mi=10.0, duration_min=90.0),
                RunActivity(run_type="easy", distance_mi=10.0, duration_min=90.0),
                RunActivity(run_type="easy", distance_mi=10.0, duration_min=90.0),
                RunActivity(run_type="long", distance_mi=10.0, duration_min=90.0),
            ],
            rationale="big jump",
        )
        # 40 mi >> 30 * 1.10 = 33
        result = self.method.validate_plan(self.state, plan)
        assert not result.is_valid
        assert any("cap" in v for v in result.violations)

    def test_no_runs_rejected(self):
        plan = PlanProposal(
            next_session_index=1,
            activities=[LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=200.0)],
            rationale="cross-training",
        )
        result = self.method.validate_plan(self.state, plan)
        assert not result.is_valid


# ====================================================================
# Daniels running — safety_check_transition + readiness
# ====================================================================

class TestDaniersSafety:
    def test_mileage_increase_within_cap(self):
        method = DaniersRunning()
        state = make_runner(weekly=30.0)
        t = StateTransitionProposal(
            dimension="weekly_mileage", from_value=30.0, to_value=33.0,
            rationale="2 weeks at 30 healthy", confidence="medium",
        )
        ok, _ = method.safety_check_transition(state, t)
        assert ok

    def test_mileage_jump_rejected(self):
        method = DaniersRunning()
        state = make_runner(weekly=30.0)
        t = StateTransitionProposal(
            dimension="weekly_mileage", from_value=30.0, to_value=40.0,
            rationale="aggressive ramp", confidence="high",
        )
        ok, msg = method.safety_check_transition(state, t)
        assert not ok
        assert "cap" in msg

    def test_vdot_change_within_cap(self):
        method = DaniersRunning()
        state = make_runner(vdot=45.0)
        t = StateTransitionProposal(
            dimension="vdot", from_value=45.0, to_value=45.3,
            rationale="recent race", confidence="medium",
        )
        ok, _ = method.safety_check_transition(state, t)
        assert ok

    def test_vdot_jump_rejected(self):
        method = DaniersRunning()
        state = make_runner(vdot=45.0)
        t = StateTransitionProposal(
            dimension="vdot", from_value=45.0, to_value=48.0,
            rationale="sudden gains", confidence="high",
        )
        ok, _ = method.safety_check_transition(state, t)
        assert not ok

    def test_readiness_2_weeks_no_injury(self):
        method = DaniersRunning()
        ready = make_runner(weekly=30.0, weeks_at=2, injuries=[])
        not_yet = make_runner(weekly=30.0, weeks_at=1, injuries=[])
        injured = make_runner(weekly=30.0, weeks_at=4, injuries=["ITB syndrome"])
        assert method.readiness_check(ready, "weekly_mileage") is True
        assert method.readiness_check(not_yet, "weekly_mileage") is False
        assert method.readiness_check(injured, "weekly_mileage") is False


# ====================================================================
# Registry
# ====================================================================

class TestRegistry:
    def test_lookup_by_id(self):
        m = get_methodology("linear_progression")
        assert m.methodology_id == "linear_progression"

    def test_lookup_by_state(self):
        state = make_powerlifter()
        m = get_methodology(state)
        assert isinstance(m, LinearProgression)

    def test_lookup_unknown_raises(self):
        with pytest.raises(KeyError):
            get_methodology("crossfit")

    def test_all_methodologies_registered(self):
        assert "linear_progression" in METHODOLOGIES
        assert "daniels" in METHODOLOGIES
