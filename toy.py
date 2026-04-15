"""Three-layer composition demo: DSPy Signature + PydanticAI Agent + pydantic-graph.

Validates the pattern claimed in:
  ACE-AI Cognitive Core - Three-Layer Stack and Dual-Scale State (§1.4)

Pattern:
  1. DSPy Signature — declarative contract (typed I/O fields + docstring instruction)
  2. PydanticAI Agent — runtime, built from the signature's output type + docstring
  3. pydantic-graph Node — FSM integration, wires agents together with typed state
  4. GEPA hook (demonstrated via Agent.override) — swap instructions at runtime

Toy problem:
  Question-answering pipeline — two nodes:
    TriageNode: classify the question, plan an approach
    AnswerNode: answer using the triage approach, report confidence

  Uses two different providers (gpt-4o-mini + claude-haiku-4-5) in the same
  graph, demonstrating heterogeneous-model composition.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

import dspy
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_graph import BaseNode, End, Graph, GraphRunContext


# ====================================================================
# Layer 0: Pydantic output models — the structured types that flow
# ====================================================================

class Triage(BaseModel):
    """Triage step output."""
    category: Literal["factual", "reasoning", "creative"] = Field(
        description="What kind of question is this?"
    )
    approach: str = Field(
        description="One-sentence strategy for answering this question."
    )


class Answer(BaseModel):
    """Final answer output."""
    answer: str = Field(description="The answer to the user's question.")
    confidence: Literal["low", "medium", "high"] = Field(
        description="How confident we are in the answer."
    )


# ====================================================================
# Layer 1: DSPy Signatures — declarative reasoning contracts
# ====================================================================

class TriageSignature(dspy.Signature):
    """Classify a user question and produce a one-sentence approach plan.

    Decide whether the question is primarily a factual lookup, a reasoning
    problem, or a creative task. Produce a brief strategy describing how
    the answer should be constructed."""

    question: str = dspy.InputField(desc="The user's question.")
    triage: Triage = dspy.OutputField(
        desc="Classification and approach plan."
    )


class AnswerSignature(dspy.Signature):
    """Answer the user's question using the triage approach.

    Given the original question and the approach plan produced by triage,
    produce a concise, direct answer along with a confidence level. If the
    question is outside your knowledge, report low confidence rather than
    fabricating details."""

    question: str = dspy.InputField(desc="The user's question.")
    approach: str = dspy.InputField(desc="The one-sentence approach from triage.")
    answer: Answer = dspy.OutputField(
        desc="The answer with confidence rating."
    )


# ====================================================================
# Layer 2: Signature → Agent bridge
# ====================================================================

def agent_from_signature(
    sig: type[dspy.Signature],
    model: str,
    instructions: str | None = None,
    tools: list | None = None,
) -> Agent:
    """Derive a PydanticAI Agent from a DSPy Signature.

    - OutputField type → PydanticAI `output_type`
    - Signature docstring → default `instructions` (overridable at runtime
      via `Agent.override(instructions=...)` — the GEPA hook)
    - InputFields flow through as part of the prompt text (see node.run below)

    DSPy's runtime (`Predict`, `ChainOfThought`, `ReAct`) is deliberately not
    used; PydanticAI is the runtime. The signature serves as declarative
    scaffolding only.
    """
    output_fields = sig.output_fields
    if len(output_fields) != 1:
        raise ValueError(
            f"agent_from_signature requires exactly one OutputField; "
            f"{sig.__name__} has {len(output_fields)}."
        )
    output_field_info = next(iter(output_fields.values()))
    output_type = output_field_info.annotation

    default_instructions = instructions or (
        sig.instructions if sig.instructions else (sig.__doc__ or "").strip()
    )

    return Agent(
        model=model,
        output_type=output_type,
        instructions=default_instructions,
        tools=tools or [],
    )


def format_prompt_from_signature(sig: type[dspy.Signature], **inputs) -> str:
    """Render a prompt from the signature's input fields + values.

    Uses the field descriptions to label each input. Keeps the signature as
    the single source of truth for how inputs are structured.
    """
    lines = []
    for name, field_info in sig.input_fields.items():
        label = field_info.json_schema_extra.get("prefix", f"{name.title()}:")
        desc = field_info.json_schema_extra.get("desc", "")
        value = inputs[name]
        header = f"{label} {desc}" if desc else label
        lines.append(f"{header}\n{value}")
    return "\n\n".join(lines)


# ====================================================================
# Layer 3: pydantic-graph — typed state machine wiring
# ====================================================================

@dataclass
class PipelineState:
    """State that flows through every node."""
    question: str
    triage: Triage | None = None
    answer: Answer | None = None


@dataclass
class TriageNode(BaseNode[PipelineState]):
    """First node: triage the question."""

    async def run(self, ctx: GraphRunContext[PipelineState]) -> "AnswerNode":
        agent = agent_from_signature(
            TriageSignature,
            model="openai:gpt-4o-mini",  # cheap + fast for classification
        )
        prompt = format_prompt_from_signature(
            TriageSignature,
            question=ctx.state.question,
        )
        result = await agent.run(prompt)
        ctx.state.triage = result.output
        return AnswerNode()


@dataclass
class AnswerNode(BaseNode[PipelineState, None, Answer]):
    """Second node: answer the question using the triage approach."""

    async def run(self, ctx: GraphRunContext[PipelineState]) -> End[Answer]:
        assert ctx.state.triage is not None, "TriageNode must run first."
        agent = agent_from_signature(
            AnswerSignature,
            model="anthropic:claude-haiku-4-5",  # different provider — demonstrates heterogeneous pipeline
        )
        prompt = format_prompt_from_signature(
            AnswerSignature,
            question=ctx.state.question,
            approach=ctx.state.triage.approach,
        )
        result = await agent.run(prompt)
        ctx.state.answer = result.output
        return End(ctx.state.answer)


# ====================================================================
# Demo runner
# ====================================================================

async def run_once(question: str, *, override_triage_instructions: str | None = None):
    """Run the graph end-to-end. Optionally demonstrate the GEPA override hook."""
    state = PipelineState(question=question)
    graph = Graph(nodes=[TriageNode, AnswerNode])

    if override_triage_instructions:
        # This is the GEPA hook: at runtime, swap in an optimized instruction
        # for a specific node's agent without changing the signature or the graph.
        # In production, the optimized instruction comes from a prompt store
        # populated by offline GEPA optimization against Pydantic Evals datasets.
        triage_agent = agent_from_signature(TriageSignature, model="openai:gpt-4o-mini")
        with triage_agent.override(instructions=override_triage_instructions):
            # We'd need to plumb the override through the node; for this toy
            # we just show that the mechanism exists and works syntactically.
            pass

    result = await graph.run(TriageNode(), state=state)

    print(f"\n{'=' * 72}")
    print(f"QUESTION: {question}")
    print(f"{'=' * 72}")
    print(f"TRIAGE ({type(state.triage).__name__}):")
    print(f"  category: {state.triage.category}")
    print(f"  approach: {state.triage.approach}")
    print(f"\nANSWER ({type(state.answer).__name__}):")
    print(f"  answer:     {state.answer.answer}")
    print(f"  confidence: {state.answer.confidence}")
    print(f"\nGraph result output type: {type(result.output).__name__}")


async def main():
    # Prove the three layers introspect cleanly at startup
    print("=" * 72)
    print("STARTUP — signature introspection (proves the declarative layer)")
    print("=" * 72)
    for sig in (TriageSignature, AnswerSignature):
        out_field = next(iter(sig.output_fields.values()))
        print(f"\n{sig.__name__}:")
        print(f"  inputs: {list(sig.input_fields)}")
        print(f"  output: {out_field.annotation.__name__}")
        print(f"  instructions (first 80 chars): {sig.instructions[:80]}...")

    # Run the graph against three different question types
    questions = [
        "What is the boiling point of water at sea level?",
        "If a train leaves Chicago at 3pm going 60mph and another leaves at 4pm going 80mph, when does the second catch up?",
        "Write a haiku about a cat watching rain.",
    ]
    for q in questions:
        await run_once(q)

    # Demonstrate that Agent.override actually changes runtime behavior.
    # This is the GEPA hook: at runtime, inject an optimized instruction that
    # affects the next .run() call, then revert cleanly.
    print(f"\n{'=' * 72}")
    print("GEPA HOOK DEMONSTRATION — Agent.override actually changes output")
    print(f"{'=' * 72}")

    # Use a smaller fixed agent so we can see instruction changes clearly.
    class Tag(BaseModel):
        sentiment: str

    override_agent = Agent(
        "openai:gpt-4o-mini",
        output_type=Tag,
        instructions="Classify sentiment as positive, negative, or neutral.",
    )
    text = "The meeting ran long but we resolved the blocker."

    r1 = await override_agent.run(f"Text: {text}")
    print(f"DEFAULT instructions → sentiment: {r1.output.sentiment!r}")

    # A deliberately weird instruction that forces a different label scheme,
    # so we can see the override is actually in effect.
    with override_agent.override(
        instructions="Classify sentiment using only the labels ENERGIZED or DRAINED."
    ):
        r2 = await override_agent.run(f"Text: {text}")
    print(f"OVERRIDE instructions → sentiment: {r2.output.sentiment!r}")

    r3 = await override_agent.run(f"Text: {text}")
    print(f"AFTER override block  → sentiment: {r3.output.sentiment!r}")

    print(
        "\nOverride uses contextvars — the swap is visible during .run() calls "
        "inside the `with` block, then reverts. This is where GEPA-optimized "
        "instructions get injected at runtime without mutating the signature."
    )


if __name__ == "__main__":
    asyncio.run(main())
