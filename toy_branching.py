"""Branching version: the state graph routes to different agents based on triage.

Same three-layer composition (DSPy Signature + PydanticAI Agent + pydantic-graph)
but demonstrates pydantic-graph as a real FSM with runtime routing decisions.

Flow:
                      ┌─→ FactualAnswerNode ──┐
  TriageNode ─────────┤                       ├─→ End[Answer]
                      ├─→ ReasoningAnswerNode ┤
                      └─→ CreativeAnswerNode ─┘

The transition from TriageNode is declared via Union return type:
    -> FactualAnswerNode | ReasoningAnswerNode | CreativeAnswerNode

At runtime, TriageNode.run() picks which one based on ctx.state.triage.category.
Each downstream agent has its own GEPA-optimizable signature with category-
specific instructions — this is how per-category prompt specialization works.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

import dspy
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from toy import (
    Triage,
    Answer,
    TriageSignature,
    agent_from_signature,
    format_prompt_from_signature,
)


# ====================================================================
# Three category-specific answer signatures
# ====================================================================

class FactualAnswerSignature(dspy.Signature):
    """Answer a factual question with a direct, cited-when-possible response.

    Keep the answer tight. If the question has a single well-known answer,
    state it plainly. If you are not certain, say low confidence rather
    than fabricating."""

    question: str = dspy.InputField(desc="The user's factual question.")
    answer: Answer = dspy.OutputField(desc="Direct factual answer + confidence.")


class ReasoningAnswerSignature(dspy.Signature):
    """Answer a reasoning question by showing your work step-by-step.

    Walk through the logic in ordered steps. State the final answer
    explicitly at the end. Use arithmetic or symbolic manipulation as
    appropriate. Confidence should reflect whether the reasoning chain
    is complete and checks out."""

    question: str = dspy.InputField(desc="The user's reasoning question.")
    answer: Answer = dspy.OutputField(
        desc="Step-by-step reasoning then final answer + confidence."
    )


class CreativeAnswerSignature(dspy.Signature):
    """Produce a creative response matching the requested form exactly.

    If the question asks for a specific form (haiku, limerick, six-word story),
    respect the form's constraints precisely. Be evocative but not verbose."""

    question: str = dspy.InputField(desc="The user's creative request.")
    answer: Answer = dspy.OutputField(
        desc="Creative output respecting any form constraints + confidence."
    )


# ====================================================================
# State
# ====================================================================

@dataclass
class PipelineState:
    question: str
    triage: Triage | None = None
    answer: Answer | None = None
    route_taken: str | None = None  # for demonstrating routing decisions


# ====================================================================
# Nodes — forward declarations needed for the Union return type
# ====================================================================

# Define all nodes first so TriageNode can reference them in its return type.

@dataclass
class FactualAnswerNode(BaseNode[PipelineState, None, Answer]):
    async def run(self, ctx: GraphRunContext[PipelineState]) -> End[Answer]:
        assert ctx.state.triage is not None
        ctx.state.route_taken = "FactualAnswerNode"
        agent = agent_from_signature(
            FactualAnswerSignature, model="openai:gpt-4o-mini"
        )
        prompt = format_prompt_from_signature(
            FactualAnswerSignature, question=ctx.state.question
        )
        result = await agent.run(prompt)
        ctx.state.answer = result.output
        return End(ctx.state.answer)


@dataclass
class ReasoningAnswerNode(BaseNode[PipelineState, None, Answer]):
    async def run(self, ctx: GraphRunContext[PipelineState]) -> End[Answer]:
        assert ctx.state.triage is not None
        ctx.state.route_taken = "ReasoningAnswerNode"
        agent = agent_from_signature(
            ReasoningAnswerSignature, model="anthropic:claude-haiku-4-5"
        )
        prompt = format_prompt_from_signature(
            ReasoningAnswerSignature, question=ctx.state.question
        )
        result = await agent.run(prompt)
        ctx.state.answer = result.output
        return End(ctx.state.answer)


@dataclass
class CreativeAnswerNode(BaseNode[PipelineState, None, Answer]):
    async def run(self, ctx: GraphRunContext[PipelineState]) -> End[Answer]:
        assert ctx.state.triage is not None
        ctx.state.route_taken = "CreativeAnswerNode"
        agent = agent_from_signature(
            CreativeAnswerSignature, model="anthropic:claude-haiku-4-5"
        )
        prompt = format_prompt_from_signature(
            CreativeAnswerSignature, question=ctx.state.question
        )
        result = await agent.run(prompt)
        ctx.state.answer = result.output
        return End(ctx.state.answer)


@dataclass
class TriageNode(BaseNode[PipelineState]):
    """Routes to one of three downstream nodes based on category."""

    async def run(
        self, ctx: GraphRunContext[PipelineState]
    ) -> FactualAnswerNode | ReasoningAnswerNode | CreativeAnswerNode:
        agent = agent_from_signature(TriageSignature, model="openai:gpt-4o-mini")
        prompt = format_prompt_from_signature(
            TriageSignature, question=ctx.state.question
        )
        result = await agent.run(prompt)
        ctx.state.triage = result.output

        # The state graph routes based on state content. The return type union
        # above tells pydantic-graph which transitions are legal; the runtime
        # decision is here.
        match ctx.state.triage.category:
            case "factual":
                return FactualAnswerNode()
            case "reasoning":
                return ReasoningAnswerNode()
            case "creative":
                return CreativeAnswerNode()


# ====================================================================
# Demo
# ====================================================================

async def run_and_report(question: str):
    state = PipelineState(question=question)
    graph = Graph(
        nodes=[
            TriageNode,
            FactualAnswerNode,
            ReasoningAnswerNode,
            CreativeAnswerNode,
        ]
    )
    result = await graph.run(TriageNode(), state=state)
    print(f"\nQ: {question}")
    print(f"   triage.category: {state.triage.category}")
    print(f"   route_taken:     {state.route_taken}")
    print(f"   confidence:      {state.answer.confidence}")
    print(f"   answer: {state.answer.answer[:120]}{'...' if len(state.answer.answer) > 120 else ''}")


async def main():
    # Show the state graph diagram first — proves the FSM structure exists
    # and reflects the branching declared in return-type annotations.
    graph = Graph(
        nodes=[
            TriageNode,
            FactualAnswerNode,
            ReasoningAnswerNode,
            CreativeAnswerNode,
        ]
    )
    print("=" * 72)
    print("STATE GRAPH (Mermaid)")
    print("=" * 72)
    print(graph.mermaid_code())
    print()
    print("Node registration:")
    for name in graph.node_defs:
        print(f"  - {name}")
    print()

    # Run three questions — each should route to a different downstream node.
    # This is the runtime routing decision visible.
    print("=" * 72)
    print("RUNTIME ROUTING — each question routes to a different answer node")
    print("=" * 72)
    questions = [
        "What is the boiling point of water at sea level?",  # expect factual
        "If a train leaves Chicago at 3pm going 60mph and another at 4pm going 80mph, when does the second catch up?",  # expect reasoning
        "Write a haiku about a cat watching rain.",  # expect creative
    ]
    for q in questions:
        await run_and_report(q)


if __name__ == "__main__":
    asyncio.run(main())
