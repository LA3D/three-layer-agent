"""Loop + persistence: the state graph handles loop-back transitions and captures
full trajectories for analysis / GEPA training data.

Demonstrates two FSM capabilities beyond the branching toy:

1. **Loop-back transitions** — a CritiqueNode that either accepts the draft or
   sends it back to GenerateNode with feedback. The graph cycle is declared
   via `-> GenerateNode | End[AcceptedDraft]` return annotation.

2. **State persistence** — `FullStatePersistence` captures every node transition
   in memory (useful for GEPA training data, debugging). `FileStatePersistence`
   serializes to JSON so the graph can resume after a crash.

Flow:
    GenerateNode ──→ CritiqueNode ──┐
         ▲                          │
         └──(if rejected)───────────┤
                                    │
                                    └──(if accepted or max_iter)──→ End
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import dspy
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_graph import BaseNode, End, Graph, GraphRunContext, FullStatePersistence
from pydantic_graph.persistence.file import FileStatePersistence

from toy import agent_from_signature, format_prompt_from_signature


# ====================================================================
# Output models
# ====================================================================

class Draft(BaseModel):
    """A generated draft."""
    tagline: str = Field(description="The proposed product tagline.")
    rationale: str = Field(description="Brief reasoning for this choice.")


class Critique(BaseModel):
    """A critic's evaluation of a draft."""
    accepted: bool = Field(description="Whether the draft is acceptable.")
    score: int = Field(ge=1, le=10, description="Quality score from 1 to 10.")
    feedback: str = Field(
        description="Specific actionable feedback. If accepted, explain why."
    )


class AcceptedDraft(BaseModel):
    """The final output when a draft is accepted (or max iterations hit)."""
    tagline: str
    total_iterations: int
    final_score: int
    acceptance_reason: str


# ====================================================================
# Signatures — declarative contracts
# ====================================================================

class GenerateSignature(dspy.Signature):
    """Generate a product tagline for the given task.

    Produce a punchy, memorable tagline. If feedback from prior attempts is
    provided, address the specific issues raised — don't just paraphrase the
    previous draft."""

    task: str = dspy.InputField(desc="The product tagline task.")
    prior_feedback: str = dspy.InputField(
        desc="Feedback on prior drafts. Empty string if this is the first attempt."
    )
    draft: Draft = dspy.OutputField(desc="The proposed tagline with rationale.")


class CritiqueSignature(dspy.Signature):
    """Critique a proposed tagline against a quality bar.

    A tagline is acceptable if it is: under 8 words, memorable, specific to the
    product (not generic), and does not rely on buzzwords. Score 1-10 and
    accept only drafts scoring 8 or above."""

    task: str = dspy.InputField(desc="The original tagline task.")
    draft_tagline: str = dspy.InputField(desc="The proposed tagline.")
    draft_rationale: str = dspy.InputField(desc="The drafter's rationale.")
    critique: Critique = dspy.OutputField(
        desc="Acceptance decision, score, and specific feedback."
    )


# ====================================================================
# State
# ====================================================================

@dataclass
class CriticLoopState:
    """State for the critique loop. Drafts and feedback accumulate."""
    task: str
    drafts: list[Draft] = field(default_factory=list)
    critiques: list[Critique] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 3


# ====================================================================
# Nodes
# ====================================================================

@dataclass
class GenerateNode(BaseNode[CriticLoopState]):
    """Generate a new draft using task + prior feedback."""

    async def run(self, ctx: GraphRunContext[CriticLoopState]) -> "CritiqueNode":
        ctx.state.iteration += 1
        agent = agent_from_signature(
            GenerateSignature, model="openai:gpt-4o-mini"
        )
        prior_feedback = (
            "\n".join(f"- {c.feedback}" for c in ctx.state.critiques)
            if ctx.state.critiques
            else "(none — first attempt)"
        )
        prompt = format_prompt_from_signature(
            GenerateSignature,
            task=ctx.state.task,
            prior_feedback=prior_feedback,
        )
        result = await agent.run(prompt)
        ctx.state.drafts.append(result.output)
        return CritiqueNode()


@dataclass
class CritiqueNode(BaseNode[CriticLoopState, None, AcceptedDraft]):
    """Evaluate the latest draft. Loop back or end."""

    async def run(
        self, ctx: GraphRunContext[CriticLoopState]
    ) -> GenerateNode | End[AcceptedDraft]:
        latest = ctx.state.drafts[-1]
        agent = agent_from_signature(
            CritiqueSignature, model="anthropic:claude-haiku-4-5"
        )
        prompt = format_prompt_from_signature(
            CritiqueSignature,
            task=ctx.state.task,
            draft_tagline=latest.tagline,
            draft_rationale=latest.rationale,
        )
        result = await agent.run(prompt)
        critique = result.output
        ctx.state.critiques.append(critique)

        # Accept decision: either the critic accepted, or we've hit max iterations
        if critique.accepted:
            return End(AcceptedDraft(
                tagline=latest.tagline,
                total_iterations=ctx.state.iteration,
                final_score=critique.score,
                acceptance_reason=critique.feedback,
            ))
        if ctx.state.iteration >= ctx.state.max_iterations:
            # Max iterations hit — return best-so-far (highest scoring)
            best_idx = max(
                range(len(ctx.state.critiques)),
                key=lambda i: ctx.state.critiques[i].score,
            )
            return End(AcceptedDraft(
                tagline=ctx.state.drafts[best_idx].tagline,
                total_iterations=ctx.state.iteration,
                final_score=ctx.state.critiques[best_idx].score,
                acceptance_reason=f"Max iterations ({ctx.state.max_iterations}) hit; returning best-scored draft.",
            ))

        # Loop back — the type system permits this because the return annotation
        # includes GenerateNode.
        return GenerateNode()


# ====================================================================
# Demo
# ====================================================================

async def run_with_full_history(task: str):
    """Run the graph capturing every node transition in memory.

    FullStatePersistence records a snapshot at every node boundary. After the
    run, you can inspect the trajectory — what state the graph was in when
    each node ran, what output the node produced, and so on. This is the
    raw material for GEPA optimization and for debugging.
    """
    graph = Graph(nodes=[GenerateNode, CritiqueNode])
    state = CriticLoopState(task=task, max_iterations=3)
    persistence = FullStatePersistence()

    async with graph.iter(
        GenerateNode(),
        state=state,
        persistence=persistence,
    ) as run:
        async for node in run:
            print(f"  → {type(node).__name__} (iteration {state.iteration})")
        result = run.result

    print(f"\n  Final: {result.output.tagline!r}")
    print(f"  Iterations: {result.output.total_iterations}")
    print(f"  Final score: {result.output.final_score}/10")
    print(f"  Reason: {result.output.acceptance_reason[:120]}")

    # Show the trajectory structure (the first few snapshots)
    print(f"\n  Trajectory has {len(persistence.history)} snapshots.")
    print(f"  First snapshot status: {persistence.history[0].status}")
    print(f"  First snapshot node:   {type(persistence.history[0].node).__name__}")


async def run_with_file_persistence(task: str, json_path: Path):
    """Run the graph persisting to a JSON file.

    FileStatePersistence serializes state + node transitions to disk. If the
    process crashes mid-run, the next invocation can load the file and resume
    from where it left off — critical for long-running sessions like clinical
    workflows where a dropped API call shouldn't restart the whole pipeline.
    """
    graph = Graph(nodes=[GenerateNode, CritiqueNode])
    state = CriticLoopState(task=task, max_iterations=2)
    persistence = FileStatePersistence(json_path)
    # pydantic-graph needs to know the expected state/run-end types for JSON round-trip
    persistence.set_graph_types(graph)

    async with graph.iter(
        GenerateNode(),
        state=state,
        persistence=persistence,
    ) as run:
        async for node in run:
            print(f"  → {type(node).__name__}")
        result = run.result

    print(f"  Final: {result.output.tagline!r}")
    print(f"  File size on disk: {json_path.stat().st_size} bytes")

    # Show a bite of the persisted JSON so you can see what's captured
    with json_path.open() as f:
        data = json.load(f)
    print(f"  JSON has {len(data)} snapshot entries.")


async def main():
    # Render the state graph first — proves the loop-back is declared.
    graph = Graph(nodes=[GenerateNode, CritiqueNode])
    print("=" * 72)
    print("STATE GRAPH (Mermaid) — notice the loop-back CritiqueNode → GenerateNode")
    print("=" * 72)
    print(graph.mermaid_code())
    print()

    # Demo 1: in-memory full trajectory
    print("=" * 72)
    print("DEMO 1 — In-memory FullStatePersistence")
    print("=" * 72)
    await run_with_full_history(
        task="Write a tagline for a high-end mechanical keyboard aimed at "
             "working programmers. Avoid generic tech buzzwords."
    )

    # Demo 2: file persistence
    print(f"\n{'=' * 72}")
    print("DEMO 2 — FileStatePersistence (JSON on disk, resumable)")
    print("=" * 72)
    json_path = Path("/tmp/three_layer_critic_loop.json")
    await run_with_file_persistence(
        task="Write a tagline for a noise-canceling commuter coffee mug.",
        json_path=json_path,
    )


if __name__ == "__main__":
    asyncio.run(main())
