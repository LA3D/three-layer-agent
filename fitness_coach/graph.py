"""pydantic-graph FSM for the cognitive core.

Nine nodes wired up per the architectural plan:

  IngestNode (parse log + universal safety pre-check)
      ├──(safety pre-check fails)──→ SafetyHaltNode → End[HaltedSession]
      └──→ ObserveNode (LLM)
              └──→ PlanNode (LLM, may run up to 2x)
                      └──→ ValidateNode (deterministic methodology check)
                              ├──(invalid AND attempts < 2)──→ PlanNode (loop back)
                              ├──(invalid AND attempts == 2)──→ FallbackPlanNode (deterministic safe default)
                              │                                      └──→ CoachNode
                              └──(valid)──→ CoachNode (LLM)
                                              └──→ AdaptNode (LLM)
                                                      ├──(transition unsafe)──→ SafetyHaltNode → End[HaltedSession]
                                                      └──(transition safe)──→ SummarizeNode (LLM) → End[SessionResult]

Single state object (`SessionState`) flows through every node; persistence
is the caller's choice (FullStatePersistence in-memory, FileStatePersistence
on disk, both are demonstrated by run.py).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Union

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

# Reuse the canonical Signature → Agent bridge from the root toy module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toy import agent_from_signature  # noqa: E402

from fitness_coach.methodology import (  # noqa: E402
    get_methodology,
    movements_to_halt,
    session_should_halt,
)
from fitness_coach.schemas import (  # noqa: E402
    AthleteState,
    CoachingOutput,
    HaltedSession,
    HandoffDoc,
    ObservationOutput,
    PlanProposal,
    SafetySignal,
    SessionLog,
    SessionResult,
    StateTransitionProposal,
    ValidationResult,
)
from fitness_coach.signatures import (  # noqa: E402
    AdaptSignature,
    CoachSignature,
    ObserveSignature,
    PlanSignature,
    SummarizeSignature,
)


SessionEndResult = Union[SessionResult, HaltedSession]


# ====================================================================
# Mutable session state — flows through every node in the graph
# ====================================================================

@dataclass
class SessionState:
    """One session's worth of mutable state. Reset per session."""

    # Inputs (set by caller before graph.run / graph.iter)
    athlete_state: AthleteState
    session_log: SessionLog
    prior_handoff: HandoffDoc | None = None  # from previous session's SummarizeNode

    # Outputs accumulated through the graph
    observation: ObservationOutput | None = None
    plan: PlanProposal | None = None
    coaching: CoachingOutput | None = None
    proposed_transition: StateTransitionProposal | None = None
    handoff: HandoffDoc | None = None

    # Plan validation tracking
    validation_attempts: int = 0
    validation_results: list[ValidationResult] = field(default_factory=list)
    used_fallback: bool = False

    # LLM model selection (per-node override possible if needed; default for the toy)
    model: str = "openai:gpt-4o-mini"


# ====================================================================
# Helper — render Pydantic input fields as JSON for prompts
# ====================================================================

def _render_prompt(signature_cls, **inputs) -> str:
    """Format a prompt using the signature's InputField labels + descriptions,
    serializing Pydantic models to JSON for readability inside the LLM context.

    None values render as a sentinel (e.g. "(none)") rather than crashing —
    relevant for nullable inputs like `prior_handoff: HandoffDoc | None`."""
    lines = []
    for name, field_info in signature_cls.input_fields.items():
        prefix = field_info.json_schema_extra.get("prefix", f"{name}:")
        desc = field_info.json_schema_extra.get("desc", "")
        value = inputs[name]

        if value is None:
            value = "(none)"
        elif hasattr(value, "model_dump_json"):
            value = value.model_dump_json(indent=2)
        elif isinstance(value, list) and value and hasattr(value[0], "model_dump"):
            import json
            value = json.dumps([v.model_dump(mode="json") for v in value], indent=2, default=str)

        header = f"{prefix} {desc}" if desc else prefix
        lines.append(f"{header}\n{value}")
    return "\n\n".join(lines)


# ====================================================================
# Forward declarations for nodes that loop back / branch
# ====================================================================

# pydantic-graph's return-type annotations are resolved at decoration time;
# we declare class names ahead of use via string forward references in
# annotations (Python `from __future__ import annotations` handles this).


# ====================================================================
# IngestNode — parse the session log and check universal safety
# ====================================================================

@dataclass
class IngestNode(BaseNode[SessionState, None, SessionEndResult]):
    """Entry point. Pre-check safety signals before the LLM ever runs."""

    async def run(
        self, ctx: GraphRunContext[SessionState],
    ) -> "ObserveNode | SafetyHaltNode":
        halt, reason = session_should_halt(ctx.state.session_log.safety_signals)
        if halt:
            return SafetyHaltNode(
                reason=f"IngestNode safety pre-check halted session: {reason}",
                source="IngestNode",
                triggering_signals=list(ctx.state.session_log.safety_signals),
            )
        return ObserveNode()


# ====================================================================
# SafetyHaltNode — terminal, fired from IngestNode or AdaptNode
# ====================================================================

@dataclass
class SafetyHaltNode(BaseNode[SessionState, None, SessionEndResult]):
    """Terminal halt node. Reason and source are populated by whoever fires it."""

    reason: str
    source: Literal["IngestNode", "AdaptNode"]
    triggering_signals: list[SafetySignal] = field(default_factory=list)

    async def run(self, ctx: GraphRunContext[SessionState]) -> End[SessionEndResult]:
        halted = HaltedSession(
            athlete_id=ctx.state.athlete_state.athlete_id,
            session_index=ctx.state.session_log.session_index,
            halted_at_node=self.source,
            reason=self.reason,
            triggering_signals=self.triggering_signals,
        )
        return End(halted)


# ====================================================================
# ObserveNode — LLM codes structured signals from the session
# ====================================================================

@dataclass
class ObserveNode(BaseNode[SessionState, None, SessionEndResult]):
    async def run(self, ctx: GraphRunContext[SessionState]) -> "PlanNode":
        agent = agent_from_signature(ObserveSignature, model=ctx.state.model, output_retries=3)
        prompt = _render_prompt(
            ObserveSignature,
            athlete_state=ctx.state.athlete_state,
            session_log=ctx.state.session_log,
        )
        result = await agent.run(prompt)
        ctx.state.observation = result.output
        return PlanNode()


# ====================================================================
# PlanNode — LLM proposes next-session prescription (may loop)
# ====================================================================

@dataclass
class PlanNode(BaseNode[SessionState, None, SessionEndResult]):
    async def run(self, ctx: GraphRunContext[SessionState]) -> "ValidateNode":
        ctx.state.validation_attempts += 1
        agent = agent_from_signature(PlanSignature, model=ctx.state.model, output_retries=3)

        # Halted movements (moderate+ severity signals) — surfaced explicitly to
        # the LLM on every attempt, not just retries. This is the LLM-side half
        # of the safety override; ValidateNode is the deterministic backstop.
        halted = movements_to_halt(ctx.state.session_log.safety_signals)
        halted_hint = ""
        if halted:
            halted_hint = (
                f"\n\nHALTED MOVEMENTS THIS SESSION (do NOT include in plan): "
                f"{sorted(halted)}\n"
                f"These movements have moderate+ pain signals from the prior session and "
                f"must be omitted or substituted."
            )

        # If we're on a retry, include the previous validation feedback in the prompt
        retry_hint = ""
        if ctx.state.validation_attempts > 1 and ctx.state.validation_results:
            last = ctx.state.validation_results[-1]
            retry_hint = (
                f"\n\nPREVIOUS PLAN FAILED VALIDATION:\n"
                f"Violations: {last.violations}\n"
                f"Suggested adjustment: {last.suggested_adjustment}\n"
                f"Revise your plan to address these."
            )

        assert ctx.state.observation is not None, "ObserveNode must run before PlanNode"
        prompt = _render_prompt(
            PlanSignature,
            athlete_state=ctx.state.athlete_state,
            observation=ctx.state.observation,
            prior_handoff=ctx.state.prior_handoff,
        ) + halted_hint + retry_hint

        result = await agent.run(prompt)
        ctx.state.plan = result.output
        return ValidateNode()


# ====================================================================
# ValidateNode — deterministic methodology check; loops back or escalates
# ====================================================================

@dataclass
class ValidateNode(BaseNode[SessionState, None, SessionEndResult]):
    async def run(
        self, ctx: GraphRunContext[SessionState],
    ) -> "PlanNode | FallbackPlanNode | CoachNode":
        assert ctx.state.plan is not None, "PlanNode must run before ValidateNode"
        methodology = get_methodology(ctx.state.athlete_state)
        # Compute the set of movements halted by this session's safety signals
        # (moderate+ severity) and pass to the methodology validator. Any plan
        # activity targeting a halted movement is rejected.
        halted = movements_to_halt(ctx.state.session_log.safety_signals)
        result = methodology.validate_plan(
            ctx.state.athlete_state, ctx.state.plan, halted_movements=halted,
        )
        ctx.state.validation_results.append(result)

        if result.is_valid:
            return CoachNode()
        if ctx.state.validation_attempts >= 2:
            return FallbackPlanNode()
        return PlanNode()  # loop back


# ====================================================================
# FallbackPlanNode — deterministic safe default after 2 LLM failures
# ====================================================================

@dataclass
class FallbackPlanNode(BaseNode[SessionState, None, SessionEndResult]):
    async def run(self, ctx: GraphRunContext[SessionState]) -> "CoachNode":
        methodology = get_methodology(ctx.state.athlete_state)
        ctx.state.plan = methodology.safe_default_plan(
            ctx.state.athlete_state, ctx.state.session_log,
        )
        ctx.state.used_fallback = True
        return CoachNode()


# ====================================================================
# CoachNode — LLM produces session-specific cues
# ====================================================================

@dataclass
class CoachNode(BaseNode[SessionState, None, SessionEndResult]):
    async def run(self, ctx: GraphRunContext[SessionState]) -> "AdaptNode":
        agent = agent_from_signature(CoachSignature, model=ctx.state.model, output_retries=3)
        assert ctx.state.plan is not None
        prompt = _render_prompt(
            CoachSignature,
            athlete_state=ctx.state.athlete_state,
            plan=ctx.state.plan,
        )
        result = await agent.run(prompt)
        ctx.state.coaching = result.output
        return AdaptNode()


# ====================================================================
# AdaptNode — LLM proposes state transition; safety-checked here
# ====================================================================

@dataclass
class AdaptNode(BaseNode[SessionState, None, SessionEndResult]):
    async def run(
        self, ctx: GraphRunContext[SessionState],
    ) -> "SummarizeNode | SafetyHaltNode":
        agent = agent_from_signature(AdaptSignature, model=ctx.state.model, output_retries=3)
        assert ctx.state.observation is not None
        prompt = _render_prompt(
            AdaptSignature,
            athlete_state=ctx.state.athlete_state,
            observation=ctx.state.observation,
        )
        result = await agent.run(prompt)
        ctx.state.proposed_transition = result.output

        # Deterministic safety check — methodology-grounded
        methodology = get_methodology(ctx.state.athlete_state)
        ok, msg = methodology.safety_check_transition(
            ctx.state.athlete_state, ctx.state.proposed_transition,
        )
        if not ok:
            return SafetyHaltNode(
                reason=f"AdaptNode proposed unsafe transition: {msg}",
                source="AdaptNode",
            )
        return SummarizeNode()


# ====================================================================
# SummarizeNode — LLM produces handoff doc; terminal
# ====================================================================

@dataclass
class SummarizeNode(BaseNode[SessionState, None, SessionEndResult]):
    async def run(self, ctx: GraphRunContext[SessionState]) -> End[SessionEndResult]:
        agent = agent_from_signature(SummarizeSignature, model=ctx.state.model, output_retries=3)
        assert ctx.state.observation is not None
        assert ctx.state.proposed_transition is not None
        prompt = _render_prompt(
            SummarizeSignature,
            athlete_state=ctx.state.athlete_state,
            session_log=ctx.state.session_log,
            observation=ctx.state.observation,
            transition=ctx.state.proposed_transition,
        )
        result = await agent.run(prompt)
        ctx.state.handoff = result.output

        assert ctx.state.coaching is not None
        return End(SessionResult(
            athlete_id=ctx.state.athlete_state.athlete_id,
            session_index=ctx.state.session_log.session_index,
            coaching=ctx.state.coaching,
            proposed_state_transition=ctx.state.proposed_transition,
            handoff=ctx.state.handoff,
            used_fallback_plan=ctx.state.used_fallback,
        ))


# ====================================================================
# Build the graph
# ====================================================================

def build_graph() -> Graph:
    """Construct the cognitive core FSM."""
    return Graph(
        nodes=[
            IngestNode,
            SafetyHaltNode,
            ObserveNode,
            PlanNode,
            ValidateNode,
            FallbackPlanNode,
            CoachNode,
            AdaptNode,
            SummarizeNode,
        ],
    )
