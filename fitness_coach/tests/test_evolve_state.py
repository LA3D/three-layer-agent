"""Tests for fitness_coach.run::evolve_state.

The state-evolution function is the bridge between session logs and the next
session's longitudinal context. Bugs here cascade silently into agent
behavior (e.g., absurd low-mileage caps after a skipped week, double-counted
failures from non-idempotent calls). Cover the explicit semantics here.
"""

from datetime import date

from fitness_coach.run import SPARSE_RUN_LOG_THRESHOLD, evolve_state
from fitness_coach.schemas import (
    EvidenceItem,
    LiftActivity,
    PowerlifterState,
    RunActivity,
    RunnerState,
    SessionLog,
)


# ====================================================================
# Fixtures
# ====================================================================

def make_powerlifter(
    squat=225.0, bench=155.0, deadlift=275.0, ohp=95.0,
    consecutive_fails=None, last_evolved=None,
) -> PowerlifterState:
    return PowerlifterState(
        athlete_id="pl_test", name="Test", training_age_months=6, bodyweight_lb=180.0,
        current_lifts={"squat": squat, "bench": bench, "deadlift": deadlift, "ohp": ohp},
        consecutive_failed_sessions=consecutive_fails or {},
        last_evolved_session_index=last_evolved,
    )


def make_runner(weekly=20.0, weeks_at=2, vdot=45.0, injuries=None, last_evolved=None) -> RunnerState:
    return RunnerState(
        athlete_id="rn_test", name="Test", training_age_months=12, vdot=vdot,
        current_weekly_mileage=weekly, target_event="half_marathon",
        weeks_at_current_mileage=weeks_at, injury_history=injuries or [],
        last_evolved_session_index=last_evolved,
    )


def make_lift_log(session_index, activities) -> SessionLog:
    return SessionLog(
        athlete_id="pl_test", session_index=session_index, session_date=date(2026, 4, 1),
        activity_log=activities, evidence=[],
    )


def make_run_log(session_index, runs) -> SessionLog:
    return SessionLog(
        athlete_id="rn_test", session_index=session_index, session_date=date(2026, 4, 1),
        activity_log=runs, evidence=[],
    )


# ====================================================================
# Idempotency
# ====================================================================

class TestIdempotency:
    def test_repeated_call_does_not_double_count_failures(self):
        state = make_powerlifter(bench=200.0, consecutive_fails={"bench": 1})
        log = make_lift_log(0, [
            LiftActivity(exercise="bench", sets=3, reps_per_set=[5, 5, 3], load_lb=200.0),
        ])
        s1 = evolve_state(state, log)
        s2 = evolve_state(s1, log)  # second call on same log
        assert s1.consecutive_failed_sessions["bench"] == 2
        assert s2.consecutive_failed_sessions["bench"] == 2  # not 3

    def test_short_circuit_returns_same_state_object(self):
        state = make_powerlifter()
        log = make_lift_log(2, [
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=225.0),
        ])
        s1 = evolve_state(state, log)
        s2 = evolve_state(s1, log)
        assert s1 is s2

    def test_idempotency_on_runner_too(self):
        state = make_runner(weekly=20.0, weeks_at=2)
        log = make_run_log(0, [
            RunActivity(run_type="easy", distance_mi=5.0, duration_min=50.0),
            RunActivity(run_type="easy", distance_mi=5.0, duration_min=50.0),
            RunActivity(run_type="easy", distance_mi=5.0, duration_min=50.0),
            RunActivity(run_type="long", distance_mi=8.0, duration_min=80.0),
        ])
        s1 = evolve_state(state, log)
        s2 = evolve_state(s1, log)
        assert s1.weeks_at_current_mileage == 3  # incremented once
        assert s2.weeks_at_current_mileage == 3  # not 4


# ====================================================================
# Powerlifter semantics
# ====================================================================

class TestPowerlifterEvolution:
    def test_failed_session_increments_failure_count(self):
        state = make_powerlifter(consecutive_fails={"bench": 1})
        log = make_lift_log(0, [
            LiftActivity(exercise="bench", sets=3, reps_per_set=[5, 5, 3], load_lb=200.0),
        ])
        new = evolve_state(state, log)
        assert new.consecutive_failed_sessions["bench"] == 2

    def test_successful_session_resets_failure_count(self):
        state = make_powerlifter(consecutive_fails={"bench": 2})
        log = make_lift_log(0, [
            LiftActivity(exercise="bench", sets=3, reps_per_set=[5, 5, 5], load_lb=200.0),
        ])
        new = evolve_state(state, log)
        assert new.consecutive_failed_sessions["bench"] == 0

    def test_current_lifts_updated_to_session_load(self):
        state = make_powerlifter(squat=225.0)
        log = make_lift_log(0, [
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=235.0),
        ])
        new = evolve_state(state, log)
        assert new.current_lifts["squat"] == 235.0

    def test_last_evolved_session_index_set(self):
        state = make_powerlifter()
        log = make_lift_log(3, [
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=225.0),
        ])
        new = evolve_state(state, log)
        assert new.last_evolved_session_index == 3


# ====================================================================
# Runner semantics
# ====================================================================

class TestRunnerEvolution:
    def test_full_week_increments_weeks_counter(self):
        state = make_runner(weekly=20.0, weeks_at=1)
        log = make_run_log(0, [
            RunActivity(run_type="easy", distance_mi=5.0, duration_min=50.0),
            RunActivity(run_type="easy", distance_mi=5.0, duration_min=50.0),
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=40.0),
            RunActivity(run_type="long", distance_mi=6.0, duration_min=60.0),
        ])
        new = evolve_state(state, log)
        assert new.current_weekly_mileage == 20.0
        assert new.weeks_at_current_mileage == 2

    def test_skipped_week_preserves_baseline_mileage(self):
        # Athlete was at 21 mi/wk baseline, but missed the week — only 1 run logged
        state = make_runner(weekly=21.0, weeks_at=4)
        log = make_run_log(0, [
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=42.0),
        ])
        new = evolve_state(state, log)
        # Baseline unchanged
        assert new.current_weekly_mileage == 21.0
        # Weeks counter NOT incremented (athlete didn't actually complete the week)
        assert new.weeks_at_current_mileage == 4

    def test_injury_history_resets_weeks_counter(self):
        state = make_runner(weekly=20.0, weeks_at=4, injuries=["ITB syndrome"])
        log = make_run_log(0, [
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=40.0),
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=40.0),
            RunActivity(run_type="easy", distance_mi=4.0, duration_min=40.0),
        ])
        new = evolve_state(state, log)
        # Mileage updates (it's a full week of 3 runs)
        assert new.current_weekly_mileage == 12.0
        # Counter resets — must re-earn 2 healthy weeks before progression readiness
        assert new.weeks_at_current_mileage == 0

    def test_sparse_threshold_constant(self):
        # Documents the threshold so any future change is intentional
        assert SPARSE_RUN_LOG_THRESHOLD == 3
