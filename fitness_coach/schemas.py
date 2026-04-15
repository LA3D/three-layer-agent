"""Pydantic models for the fitness coach toy.

Design notes:
  - `ActivityRecord` and `AthleteState` are discriminated unions, NOT Protocols.
    Pydantic cannot validate a Protocol in `OutputField` / `InputField`, and we
    want schema enforcement at the PydanticAI / DSPy boundary.
  - Evidence carries explicit `source` and `trust_weight` so the LLM reasons
    about provenance — this is the concrete form of requirement #3 ("evidence
    with provenance" from the architectural demonstration list).
  - LLM output models (PlanProposal, CoachingOutput, StateTransitionProposal,
    ObservationOutput, HandoffDoc) are kept tight — single-purpose, easy to
    GEPA-optimize against later.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


# ====================================================================
# Activity records — what an athlete actually did in a session
# ====================================================================

class LiftActivity(BaseModel):
    """One lifting movement performed in a session."""
    activity_type: Literal["lift"] = "lift"
    exercise: Literal["squat", "bench", "deadlift", "ohp", "row"]
    sets: int = Field(ge=1)
    reps_per_set: list[int]
    load_lb: float = Field(ge=0)
    rpe: float | None = Field(default=None, ge=1.0, le=10.0,
                              description="Rate of Perceived Exertion 1-10")
    notes: str = ""


class RunActivity(BaseModel):
    """One running activity performed in a session."""
    activity_type: Literal["run"] = "run"
    run_type: Literal["easy", "tempo", "interval", "long", "recovery"]
    distance_mi: float = Field(ge=0)
    duration_min: float = Field(ge=0)
    average_pace_min_per_mi: float | None = None
    average_hr: int | None = None
    perceived_effort: float | None = Field(default=None, ge=1.0, le=10.0)
    notes: str = ""


ActivityRecord = Annotated[
    Union[LiftActivity, RunActivity],
    Field(discriminator="activity_type"),
]


# ====================================================================
# Evidence with provenance — multi-stream trust-weighted observations
# ====================================================================

EvidenceSource = Literal[
    "athlete_self_report",
    "form_check",
    "objective_metric",
    "subjective_observation",
]

TrustWeight = Literal["high", "medium", "low"]


class EvidenceItem(BaseModel):
    """A single evidence statement with explicit provenance."""
    source: EvidenceSource
    trust_weight: TrustWeight
    content: str = Field(min_length=1)


# ====================================================================
# Safety signals — trigger hard overrides
# ====================================================================

SafetySeverity = Literal["low", "moderate", "high", "critical"]

SafetySignalType = Literal[
    "pain", "form_breakdown", "overtraining",
    "illness", "injury_recurrence",
]


class SafetySignal(BaseModel):
    """A safety concern raised during a session that may halt training."""
    severity: SafetySeverity
    signal_type: SafetySignalType
    location_or_movement: str = Field(min_length=1,
                                      description="Body part or movement, e.g. 'left knee' or 'squat'")
    description: str = Field(min_length=1)


# ====================================================================
# Session log — the input to the cognitive core for one session
# ====================================================================

class SessionLog(BaseModel):
    """A complete record of one session — what happened + evidence + safety."""
    athlete_id: str
    session_index: int = Field(ge=0)
    session_date: date
    activity_log: list[ActivityRecord]
    evidence: list[EvidenceItem]
    safety_signals: list[SafetySignal] = []
    notes_from_previous_handoff: str | None = None


# ====================================================================
# Progression history — typed transitions to the longitudinal state
# ====================================================================

class ProgressionEvent(BaseModel):
    """A discrete change to the athlete's longitudinal state."""
    timestamp: datetime
    dimension: str = Field(description="State dimension changed, e.g. 'squat_load_lb' or 'weekly_mileage'")
    from_value: float
    to_value: float
    rationale: str
    source_session_index: int = Field(ge=0)


# ====================================================================
# Athlete state — discriminated union (NOT Protocol; Pydantic can't validate Protocol)
# ====================================================================

class PowerlifterState(BaseModel):
    """Longitudinal state for a powerlifter following linear progression."""
    population: Literal["powerlifter"] = "powerlifter"
    athlete_id: str
    name: str
    training_age_months: int = Field(ge=0)
    bodyweight_lb: float = Field(gt=0)
    current_lifts: dict[str, float] = Field(
        description="Working load by lift, e.g. {'squat': 245.0, 'bench': 165.0}"
    )
    methodology_id: str = Field(default="linear_progression",
                                description="Which methodology this athlete follows")
    injury_history: list[str] = []
    consecutive_failed_sessions: dict[str, int] = Field(
        default_factory=dict,
        description="Per-lift count of recent sessions where target was missed",
    )
    progression_history: list[ProgressionEvent] = []
    last_evolved_session_index: int | None = Field(
        default=None,
        description="Index of the most recent session_log applied by evolve_state. "
                    "Used to short-circuit repeated evolution calls and keep counter math idempotent.",
    )


class RunnerState(BaseModel):
    """Longitudinal state for a runner following Daniels-style training."""
    population: Literal["runner"] = "runner"
    athlete_id: str
    name: str
    training_age_months: int = Field(ge=0)
    vdot: float = Field(gt=0, description="Daniels VDOT score")
    current_weekly_mileage: float = Field(ge=0)
    target_event: Literal["5k", "10k", "half_marathon", "marathon"]
    target_event_date: date | None = None
    methodology_id: str = Field(default="daniels")
    injury_history: list[str] = []
    weeks_at_current_mileage: int = Field(default=0, ge=0)
    progression_history: list[ProgressionEvent] = []
    last_evolved_session_index: int | None = Field(
        default=None,
        description="Index of the most recent session_log applied by evolve_state. "
                    "Used to short-circuit repeated evolution calls and keep counter math idempotent.",
    )


AthleteState = Annotated[
    Union[PowerlifterState, RunnerState],
    Field(discriminator="population"),
]


# ====================================================================
# LLM output models — what each cognitive-core node produces
# ====================================================================

class ObservationOutput(BaseModel):
    """ObserveNode output — coded signals from the session log."""
    progression_signals: list[str] = Field(
        description="Observed signs of progress (e.g. 'all sets at target RPE', 'pace improved')"
    )
    concern_signals: list[str] = Field(
        description="Observed signs to monitor (e.g. 'RPE creeping up', 'unilateral fatigue')"
    )
    evidence_quality_assessment: str = Field(
        description="Brief narrative on evidence trust — which sources are reliable for this session"
    )


class PlanProposal(BaseModel):
    """PlanNode output — the prescription for the NEXT session."""
    next_session_index: int = Field(ge=0)
    activities: list[ActivityRecord] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class ValidationResult(BaseModel):
    """ValidateNode output — purely deterministic, never LLM-generated."""
    is_valid: bool
    violations: list[str] = []
    suggested_adjustment: str | None = None


class CoachingOutput(BaseModel):
    """CoachNode output — cues and feedback for THIS session."""
    cues: list[str] = Field(min_length=1,
                            description="Actionable cues for the athlete during this session")
    watch_for: list[str] = Field(default_factory=list,
                                 description="Things for the athlete or coach to monitor")
    rationale: str = Field(min_length=1)


class StateTransitionProposal(BaseModel):
    """AdaptNode output — proposed update to the longitudinal state."""
    dimension: str = Field(description="State dimension to change, e.g. 'squat_load_lb'")
    from_value: float
    to_value: float
    rationale: str = Field(min_length=1)
    confidence: Literal["low", "medium", "high"]


class HandoffDoc(BaseModel):
    """SummarizeNode output — bridges to the next session's PlanNode."""
    athlete_id: str
    session_index: int = Field(ge=0)
    areas_to_work_on: list[str] = Field(min_length=1)
    what_worked: list[str] = []
    what_to_watch_for: list[str] = []
    next_session_focus: str = Field(min_length=1)
    generated_at: datetime


# ====================================================================
# Terminal session results
# ====================================================================

class SessionResult(BaseModel):
    """Terminal output of a normal session run."""
    athlete_id: str
    session_index: int
    coaching: CoachingOutput
    proposed_state_transition: StateTransitionProposal | None = None
    handoff: HandoffDoc
    used_fallback_plan: bool = Field(
        default=False,
        description="True if PlanNode failed validation twice and FallbackPlanNode took over",
    )


class HaltedSession(BaseModel):
    """Terminal output when SafetyHaltNode fires (from log or AdaptNode)."""
    athlete_id: str
    session_index: int
    halted_at_node: Literal["IngestNode", "AdaptNode"]
    reason: str = Field(min_length=1)
    triggering_signals: list[SafetySignal] = []
