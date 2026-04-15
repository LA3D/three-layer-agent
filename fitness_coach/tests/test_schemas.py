"""Round-trip and discrimination tests for fitness_coach.schemas.

Goals:
  - Verify every model serializes to JSON and deserializes back identically
  - Verify discriminated unions (ActivityRecord, AthleteState) dispatch correctly
  - Verify field validators (RPE bounds, severity enums, etc.) reject bad data
  - Catch schema regressions before they hit the LLM runtime
"""

from datetime import date, datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from fitness_coach.schemas import (
    ActivityRecord,
    AthleteState,
    CoachingOutput,
    EvidenceItem,
    HaltedSession,
    HandoffDoc,
    LiftActivity,
    ObservationOutput,
    PlanProposal,
    PowerlifterState,
    ProgressionEvent,
    RunActivity,
    RunnerState,
    SafetySignal,
    SessionLog,
    SessionResult,
    StateTransitionProposal,
    ValidationResult,
)


# ====================================================================
# Discriminated union dispatch
# ====================================================================

class TestActivityRecordUnion:
    def test_lift_activity_dispatch(self):
        ta = TypeAdapter(ActivityRecord)
        raw = {
            "activity_type": "lift",
            "exercise": "squat",
            "sets": 3,
            "reps_per_set": [5, 5, 5],
            "load_lb": 245.0,
            "rpe": 7.5,
        }
        parsed = ta.validate_python(raw)
        assert isinstance(parsed, LiftActivity)
        assert parsed.exercise == "squat"
        assert parsed.rpe == 7.5

    def test_run_activity_dispatch(self):
        ta = TypeAdapter(ActivityRecord)
        raw = {
            "activity_type": "run",
            "run_type": "easy",
            "distance_mi": 5.0,
            "duration_min": 47.5,
        }
        parsed = ta.validate_python(raw)
        assert isinstance(parsed, RunActivity)
        assert parsed.run_type == "easy"

    def test_unknown_activity_type_rejected(self):
        ta = TypeAdapter(ActivityRecord)
        with pytest.raises(ValidationError):
            ta.validate_python({"activity_type": "swim", "distance_mi": 1.0})


class TestAthleteStateUnion:
    def test_powerlifter_dispatch(self):
        ta = TypeAdapter(AthleteState)
        raw = {
            "population": "powerlifter",
            "athlete_id": "pl_001",
            "name": "Test Lifter",
            "training_age_months": 6,
            "bodyweight_lb": 180.0,
            "current_lifts": {"squat": 225.0, "bench": 155.0, "deadlift": 275.0, "ohp": 95.0},
        }
        parsed = ta.validate_python(raw)
        assert isinstance(parsed, PowerlifterState)
        assert parsed.methodology_id == "linear_progression"

    def test_runner_dispatch(self):
        ta = TypeAdapter(AthleteState)
        raw = {
            "population": "runner",
            "athlete_id": "rn_001",
            "name": "Test Runner",
            "training_age_months": 12,
            "vdot": 45.0,
            "current_weekly_mileage": 25.0,
            "target_event": "half_marathon",
        }
        parsed = ta.validate_python(raw)
        assert isinstance(parsed, RunnerState)
        assert parsed.methodology_id == "daniels"

    def test_unknown_population_rejected(self):
        ta = TypeAdapter(AthleteState)
        with pytest.raises(ValidationError):
            ta.validate_python({"population": "swimmer", "athlete_id": "x", "name": "y"})


# ====================================================================
# Round-trip serialization
# ====================================================================

class TestRoundTrip:
    def test_lift_activity_round_trip(self):
        original = LiftActivity(
            exercise="bench", sets=5, reps_per_set=[5, 5, 5, 5, 5],
            load_lb=185.0, rpe=8.0, notes="bar speed slowed last set",
        )
        json_data = original.model_dump_json()
        restored = LiftActivity.model_validate_json(json_data)
        assert restored == original

    def test_run_activity_round_trip(self):
        original = RunActivity(
            run_type="tempo", distance_mi=4.0, duration_min=28.0,
            average_pace_min_per_mi=7.0, perceived_effort=8.0,
        )
        restored = RunActivity.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_session_log_round_trip(self):
        original = SessionLog(
            athlete_id="pl_001",
            session_index=2,
            session_date=date(2026, 4, 15),
            activity_log=[
                LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5], load_lb=235.0, rpe=7.0),
            ],
            evidence=[
                EvidenceItem(source="athlete_self_report", trust_weight="medium",
                             content="felt strong, no pain"),
            ],
            safety_signals=[],
        )
        restored = SessionLog.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_powerlifter_state_round_trip(self):
        original = PowerlifterState(
            athlete_id="pl_001", name="A", training_age_months=8,
            bodyweight_lb=180.0,
            current_lifts={"squat": 225.0, "bench": 155.0, "deadlift": 275.0, "ohp": 95.0},
            progression_history=[
                ProgressionEvent(
                    timestamp=datetime(2026, 4, 1, 12, tzinfo=timezone.utc),
                    dimension="squat_load_lb", from_value=220.0, to_value=225.0,
                    rationale="hit RPE target across all sets", source_session_index=1,
                ),
            ],
        )
        restored = PowerlifterState.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_runner_state_round_trip(self):
        original = RunnerState(
            athlete_id="rn_001", name="B", training_age_months=18,
            vdot=48.0, current_weekly_mileage=30.0, target_event="half_marathon",
            weeks_at_current_mileage=2,
        )
        restored = RunnerState.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_handoff_doc_round_trip(self):
        original = HandoffDoc(
            athlete_id="pl_001", session_index=3,
            areas_to_work_on=["bracing on heavy squats", "elbow position on bench"],
            what_worked=["lighter warmup helped"],
            what_to_watch_for=["recurring left-side fatigue"],
            next_session_focus="bench RPE check before adding load",
            generated_at=datetime(2026, 4, 15, 14, tzinfo=timezone.utc),
        )
        restored = HandoffDoc.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_session_result_round_trip(self):
        original = SessionResult(
            athlete_id="pl_001", session_index=2,
            coaching=CoachingOutput(
                cues=["chest up on squat"], watch_for=[], rationale="form check from last session",
            ),
            proposed_state_transition=StateTransitionProposal(
                dimension="squat_load_lb", from_value=235.0, to_value=240.0,
                rationale="hit target RPE", confidence="medium",
            ),
            handoff=HandoffDoc(
                athlete_id="pl_001", session_index=2,
                areas_to_work_on=["squat depth"],
                next_session_focus="recheck depth on warmup",
                generated_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            ),
        )
        restored = SessionResult.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_halted_session_round_trip(self):
        original = HaltedSession(
            athlete_id="pl_001", session_index=4,
            halted_at_node="IngestNode",
            reason="Athlete reported moderate knee pain on warmup",
            triggering_signals=[
                SafetySignal(severity="moderate", signal_type="pain",
                             location_or_movement="left knee",
                             description="sharp on descent"),
            ],
        )
        restored = HaltedSession.model_validate_json(original.model_dump_json())
        assert restored == original


# ====================================================================
# Field validation
# ====================================================================

class TestValidation:
    def test_rpe_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5],
                         load_lb=225.0, rpe=11.0)
        with pytest.raises(ValidationError):
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5],
                         load_lb=225.0, rpe=0.5)

    def test_negative_load_rejected(self):
        with pytest.raises(ValidationError):
            LiftActivity(exercise="squat", sets=3, reps_per_set=[5, 5, 5],
                         load_lb=-10.0)

    def test_zero_sets_rejected(self):
        with pytest.raises(ValidationError):
            LiftActivity(exercise="squat", sets=0, reps_per_set=[],
                         load_lb=225.0)

    def test_empty_evidence_content_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceItem(source="athlete_self_report", trust_weight="high", content="")

    def test_unknown_safety_severity_rejected(self):
        with pytest.raises(ValidationError):
            SafetySignal(severity="catastrophic",  # not in enum
                         signal_type="pain",
                         location_or_movement="knee", description="hurt")

    def test_empty_plan_proposal_activities_rejected(self):
        with pytest.raises(ValidationError):
            PlanProposal(next_session_index=3, activities=[], rationale="empty plan")

    def test_negative_training_age_rejected(self):
        with pytest.raises(ValidationError):
            PowerlifterState(athlete_id="x", name="y", training_age_months=-1,
                             bodyweight_lb=180.0,
                             current_lifts={"squat": 225.0})

    def test_observation_output_accepts_empty_signal_lists(self):
        # progression_signals and concern_signals can both be empty
        out = ObservationOutput(
            progression_signals=[],
            concern_signals=[],
            evidence_quality_assessment="evidence is thin this session",
        )
        assert out.progression_signals == []

    def test_validation_result_default_violations(self):
        ok = ValidationResult(is_valid=True)
        assert ok.violations == []
        assert ok.suggested_adjustment is None
