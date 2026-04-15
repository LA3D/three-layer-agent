"""DSPy Signatures — declarative reasoning contracts for the cognitive core.

Five signatures, one per LLM-driven node in the graph. Population-agnostic:
each takes an `AthleteState` (discriminated union of PowerlifterState |
RunnerState) as input. Methodology-specific dispatch happens in the
deterministic methodology layer, not in signatures.

These are pure declaration — no runtime. PydanticAI agents are derived from
them at node construction time via `toy.agent_from_signature`.
"""

from __future__ import annotations

import dspy

from fitness_coach.schemas import (
    AthleteState,
    CoachingOutput,
    HandoffDoc,
    ObservationOutput,
    PlanProposal,
    SessionLog,
    StateTransitionProposal,
)


class ObserveSignature(dspy.Signature):
    """Code structured behavioral signals from a session log given the athlete's longitudinal state.

    Read the session activities and evidence streams. Identify signs of
    progression (e.g., target RPE met across all sets, pace held under
    target effort), signs of concern (e.g., RPE creeping up at the same
    load, form regression in evidence, recovery deficit), and assess the
    quality of the evidence (which sources were reliable for this session,
    where evidence conflicts).

    Reason explicitly about evidence provenance — high-trust sources
    (objective metrics, form checks) outweigh low-trust ones (vague
    self-reports). Do NOT fabricate observations not supported by the log.
    Empty signal lists are OK when the session genuinely had nothing
    notable to report."""

    athlete_state: AthleteState = dspy.InputField(
        desc="Current longitudinal state of the athlete — population, current loads/mileage, "
             "training history, injury history",
    )
    session_log: SessionLog = dspy.InputField(
        desc="The session that just happened — activities performed, evidence streams with "
             "trust weights, any safety signals raised",
    )
    observation: ObservationOutput = dspy.OutputField(
        desc="Structured observation: progression signals, concern signals, evidence quality",
    )


class PlanSignature(dspy.Signature):
    """Propose the next session prescription given longitudinal state, observation, and prior handoff.

    Produce a PlanProposal that respects the athlete's current methodology.

    For powerlifters: a "session" is one workout. Choose loads consistent
    with linear progression (current or current + standard increment of
    +5 lb squat/deadlift, +2.5 lb bench/ohp). Use 3-5 sets of 3-5 reps;
    deadlift may be 1x5 (Rippetoe convention).

    For runners: a "session" is one training WEEK, not a single run. The
    PlanProposal MUST contain at least 3 distinct RunActivity entries
    distributed across the week. Keep weekly mileage within +10% of
    current. Maintain ≥80% easy/recovery/long distribution (long runs
    count as easy mileage in Daniels — they're at conversational pace,
    not threshold). Quality work (tempo, interval) is capped at 20% and
    should ONLY be included when the prior week's evidence shows the
    athlete handled volume well without recovery deficits or pain markers.
    Default for athletes returning from disruption, injury, or low base
    mileage: ALL easy/long, NO quality. When in doubt, omit quality. Do
    NOT collapse a week into a single long run — that's structurally unsafe
    for recovering athletes and violates the distribution rule.

    Read the prior session's observation and handoff to identify what
    should be the focus this session. If the observation flags a concern
    (e.g., rising RPE, recovery deficit, returning injury markers), do
    not progress — hold or deload. After three consecutive failed
    sessions on a lift, methodology requires a 10% deload before any
    further progression or hold.

    Halted movements (declared in the prompt or implicit from session
    safety signals) MUST be omitted from the plan or substituted.

    The deterministic ValidateNode will check your proposal against
    methodology constraints. If your plan is invalid you'll get a chance
    to revise; after two failures, a safe default plan is used. Plan well
    the first time."""

    athlete_state: AthleteState = dspy.InputField(
        desc="Current longitudinal state",
    )
    observation: ObservationOutput = dspy.InputField(
        desc="Coded observation from the session that just happened",
    )
    prior_handoff: str = dspy.InputField(
        desc="Handoff document from the previous session (areas to work on, what worked, "
             "what to watch for). Empty string if this is session 0.",
    )
    plan: PlanProposal = dspy.OutputField(
        desc="Prescription for the NEXT session — activities + rationale",
    )


class CoachSignature(dspy.Signature):
    """Produce specific, actionable coaching cues for THIS session.

    Given the prescribed plan and the athlete's longitudinal context, write
    cues that are: (1) tied to the specific activities prescribed, (2)
    informed by the athlete's documented weaknesses or focus areas, (3)
    actionable in-the-moment rather than abstract.

    Avoid generic platitudes ("focus", "stay relaxed"). Prefer specific
    technical or attentional cues tied to what the athlete is actually
    doing this session. If the plan includes a deload or a held load,
    explain WHY in the rationale so the athlete understands."""

    athlete_state: AthleteState = dspy.InputField(
        desc="Current longitudinal state — informs which technical cues are relevant",
    )
    plan: PlanProposal = dspy.InputField(
        desc="The plan for this session",
    )
    coaching: CoachingOutput = dspy.OutputField(
        desc="Specific cues + watch-fors + rationale",
    )


class AdaptSignature(dspy.Signature):
    """Propose a typed state transition for the athlete's longitudinal state.

    Based on the observation from the session that just happened, propose
    AT MOST ONE state transition (e.g., increase squat working load,
    increase weekly mileage, mark consecutive_failed_sessions on a lift).
    The transition will be safety-checked by the deterministic methodology
    layer before being applied — do not propose changes that exceed
    methodology thresholds.

    If no transition is warranted (e.g., session was a hold, or the
    observation flagged concerns), set the rationale to explain that and
    return a 'no-op' transition (from_value == to_value on a placeholder
    dimension). Confidence reflects how strongly the evidence supports
    the proposed change.

    DO NOT propose unsafe changes to silently 'try them anyway' — the
    safety check exists to enforce methodology, not to be worked around."""

    athlete_state: AthleteState = dspy.InputField(
        desc="Current longitudinal state BEFORE the proposed transition",
    )
    observation: ObservationOutput = dspy.InputField(
        desc="Coded observation from the just-completed session",
    )
    transition: StateTransitionProposal = dspy.OutputField(
        desc="At most one proposed state change with rationale and confidence",
    )


class SummarizeSignature(dspy.Signature):
    """Generate the handoff document that bridges to the next session.

    The handoff is the working-memory bridge between sessions. The next
    session's PlanNode will read this — not the raw session log — as
    primary context. So compress the session into actionable forward-
    looking notes:
      - areas_to_work_on: what should next session's plan focus on
      - what_worked: strategies/cues that paid off this session
      - what_to_watch_for: signals that should trigger attention next session
      - next_session_focus: the single most important thing for next session

    Be specific. 'Focus on form' is useless; 'check left elbow position
    on bench at 200 lb' is useful. The handoff is human-readable: a real
    coach picking up this athlete next session should be able to act on
    it without reading the prior session log in detail."""

    athlete_state: AthleteState = dspy.InputField(
        desc="Current longitudinal state (after any AdaptNode transition was applied)",
    )
    session_log: SessionLog = dspy.InputField(
        desc="The session that just happened",
    )
    observation: ObservationOutput = dspy.InputField(
        desc="Coded observation",
    )
    transition: StateTransitionProposal = dspy.InputField(
        desc="The state transition proposed by AdaptNode (may be no-op)",
    )
    handoff: HandoffDoc = dspy.OutputField(
        desc="Handoff document for the next session",
    )
