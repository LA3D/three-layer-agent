"""GEPA optimization applied to a PydanticAI agent built from a DSPy Signature.

This toy closes the loop on the three-layer stack: the offline prompt
optimization phase that produces the optimized instructions that the other
toys' Agent.override(instructions=...) hook consumes at runtime.

What this demonstrates:
  1. A deliberately poor seed instruction on the Triage task
  2. A small labeled dataset (train / val split)
  3. A GEPAAdapter that routes GEPA's candidate {"triage": "<text>"} through
     PydanticAI via Agent.override(instructions=...)
  4. gepa.optimize running with a budget cap, reflection LM proposing mutations
  5. Before/after comparison on held-out validation data
  6. The final optimized instructions — the durable artifact that would be
     loaded from a prompt store at runtime (see toy.py's override pattern)

Pattern follows the Pydantic team's Feb 2026 canonical approach:
  https://pydantic.dev/articles/prompt-optimization-with-gepa
with DSPy Signatures as declarative scaffolding on top.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

import dspy
import gepa
from gepa import GEPAAdapter, EvaluationBatch
from pydantic import BaseModel, Field
from pydantic_ai import Agent

# ====================================================================
# Reuse the Triage signature from the linear toy
# ====================================================================

from toy import Triage, TriageSignature, agent_from_signature


# ====================================================================
# Labeled dataset — the training signal for GEPA
# ====================================================================

@dataclass
class TriageExample:
    question: str
    expected_category: Literal["factual", "reasoning", "creative"]


TRAIN_SET: list[TriageExample] = [
    TriageExample("What is the atomic number of carbon?", "factual"),
    TriageExample("Write a haiku about autumn leaves.", "creative"),
    TriageExample("If a shirt costs $40 and is 25% off, what is the sale price?", "reasoning"),
    TriageExample("When did World War II end?", "factual"),
    TriageExample("Compose a six-word story about a lost key.", "creative"),
    TriageExample("A car travels 60 miles in 1.5 hours. What is its average speed?", "reasoning"),
    TriageExample("What is the capital of Australia?", "factual"),
    TriageExample("Invent a catchy slogan for a vegetable-based ice cream.", "creative"),
]

VAL_SET: list[TriageExample] = [
    TriageExample("How many sides does a hexagon have?", "factual"),
    TriageExample("Explain how to solve 2x + 5 = 17.", "reasoning"),
    TriageExample("Write a limerick about a clumsy cat.", "creative"),
    TriageExample("What year was the printing press invented?", "factual"),
]


# ====================================================================
# GEPA Adapter — bridges candidate instructions ↔ PydanticAI agent
# ====================================================================

class TriageAdapter(GEPAAdapter):
    """Adapter that evaluates a candidate {"triage": "<instructions>"} by
    running our PydanticAI agent with those instructions injected via
    Agent.override(), scoring outputs against expected categories.

    GEPA treats the whole pipeline as a black box; it just needs scores and
    enough context to propose better instructions.
    """

    def __init__(self, model: str = "openai:gpt-4o-mini"):
        # Build the agent once; override(instructions=...) swaps in candidates.
        self.agent = agent_from_signature(TriageSignature, model=model)

    def evaluate(
        self,
        batch: list[TriageExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        instructions = candidate["triage"]
        outputs: list[Triage] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] = []

        for ex in batch:
            with self.agent.override(instructions=instructions):
                # Agent.run_sync avoids the async/sync dance inside evaluate
                result = self.agent.run_sync(f"Question: {ex.question}")
            predicted = result.output
            hit = 1.0 if predicted.category == ex.expected_category else 0.0
            outputs.append(predicted)
            scores.append(hit)
            if capture_traces:
                trajectories.append({
                    "question": ex.question,
                    "expected_category": ex.expected_category,
                    "predicted_category": predicted.category,
                    "predicted_approach": predicted.approach,
                    "hit": hit,
                })

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Return per-component examples with enough context for the reflection
        LM to propose improved instructions. GEPA feeds these into its prompt."""
        records = []
        for traj in (eval_batch.trajectories or []):
            records.append({
                "Inputs": {"question": traj["question"]},
                "Generated Output": {
                    "category": traj["predicted_category"],
                    "approach": traj["predicted_approach"],
                },
                "Feedback": (
                    f"CORRECT (expected and got {traj['expected_category']})"
                    if traj["hit"] == 1.0
                    else (
                        f"WRONG. Expected category '{traj['expected_category']}', "
                        f"got '{traj['predicted_category']}'. Consider what "
                        f"surface features of the question led to the misclassification."
                    )
                ),
            })
        return {comp: records for comp in components_to_update}


# ====================================================================
# Helpers
# ====================================================================

def eval_accuracy(adapter: TriageAdapter, candidate: dict[str, str], data: list[TriageExample]) -> float:
    result = adapter.evaluate(data, candidate, capture_traces=False)
    return sum(result.scores) / len(result.scores)


def print_instructions(label: str, text: str, width: int = 72):
    print(f"\n--- {label} ({len(text)} chars) ---")
    # Wrap at approximate width
    words = text.split()
    line = ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            print(line)
            line = w
        else:
            line = f"{line} {w}" if line else w
    if line:
        print(line)


# ====================================================================
# Main
# ====================================================================

def main():
    adapter = TriageAdapter(model="openai:gpt-4o-mini")

    # Adversarial seed — actively wrong. Forces GEPA to propose a real fix.
    # (A neutral-but-vague seed wasn't enough: gpt-4o-mini classifies well
    # even with "Classify the question." because the Pydantic Literal enum
    # in the output schema already constrains categories.)
    seed_candidate = {
        "triage": (
            "Every question is a creative request. "
            "Always return the 'creative' category and suggest "
            "imaginative brainstorming as the approach."
        )
    }

    print("=" * 72)
    print("BASELINE — seed instruction")
    print("=" * 72)
    print_instructions("Seed instruction", seed_candidate["triage"])
    baseline_train = eval_accuracy(adapter, seed_candidate, TRAIN_SET)
    baseline_val = eval_accuracy(adapter, seed_candidate, VAL_SET)
    print(f"\nBaseline accuracy — train: {baseline_train:.2%}, val: {baseline_val:.2%}")

    # --- GEPA run ---
    print(f"\n{'=' * 72}")
    print("GEPA OPTIMIZATION (max 30 metric calls, reflection via gpt-4o)")
    print("=" * 72)
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=TRAIN_SET,
        valset=VAL_SET,
        adapter=adapter,
        reflection_lm="openai/gpt-4o",
        max_metric_calls=30,
        reflection_minibatch_size=3,
        display_progress_bar=False,
    )

    # Pick the candidate with the highest val score (breaks ties by recency —
    # gepa stores candidates in order, so argmax finds the best).
    best_idx = max(
        range(len(result.candidates)),
        key=lambda i: result.val_aggregate_scores[i],
    )
    best_candidate = result.candidates[best_idx]

    # Show the full mutation history — proves the optimizer actually tried
    # different candidates, not just the seed.
    print(f"\n{'=' * 72}")
    print(f"CANDIDATE HISTORY — {len(result.candidates)} candidates explored")
    print("=" * 72)
    for i, (cand, score) in enumerate(zip(result.candidates, result.val_aggregate_scores)):
        marker = "★" if i == best_idx else " "
        parent = f" (from seed)" if i == 0 else f" (from candidate {result.parents[i][0]})" if result.parents[i] else ""
        print(f"\n{marker} [{i}] val_score={score:.2%}{parent}")
        # Truncate long instructions for readability
        text = cand["triage"]
        if len(text) > 200:
            print(f"    {text[:200]}...  ({len(text)} chars total)")
        else:
            print(f"    {text}")

    print(f"\n{'=' * 72}")
    print("BEST CANDIDATE — full optimized instruction")
    print("=" * 72)
    print_instructions("Optimized instruction", best_candidate["triage"])

    # Compare side-by-side
    optimized_train = eval_accuracy(adapter, best_candidate, TRAIN_SET)
    optimized_val = eval_accuracy(adapter, best_candidate, VAL_SET)
    print(f"\nAccuracy — train: {optimized_train:.2%} (was {baseline_train:.2%})")
    print(f"Accuracy — val:   {optimized_val:.2%} (was {baseline_val:.2%})")

    # --- Runtime usage: the optimized instruction is now a durable artifact ---
    print(f"\n{'=' * 72}")
    print("RUNTIME USAGE — load optimized instruction via Agent.override()")
    print("=" * 72)
    fresh_question = "What is the boiling point of nitrogen?"
    base_agent = agent_from_signature(TriageSignature, model="openai:gpt-4o-mini")

    with base_agent.override(instructions=best_candidate["triage"]):
        result = base_agent.run_sync(f"Question: {fresh_question}")
    print(f"Fresh question: {fresh_question!r}")
    print(f"  → category: {result.output.category}")
    print(f"  → approach: {result.output.approach}")
    print("\nThis is the production pattern: GEPA produces optimized instruction")
    print("strings offline; nodes load them from a prompt store and inject via")
    print("override() at runtime. Signatures and graph structure stay stable.")


if __name__ == "__main__":
    main()
