# three-layer-agent

A minimal toy demonstrating the three-layer composition for typed agent pipelines:

1. **DSPy Signature** — declarative reasoning contract (typed I/O fields + docstring instruction)
2. **PydanticAI Agent** — runtime, built from the signature's output type and docstring
3. **pydantic-graph** — typed state machine that wires agents together with explicit transitions

Plus the **GEPA hook** via `Agent.override(instructions=...)` for runtime prompt injection after offline optimization.

## Why this stack

- **Signatures are declarative, not runtime.** DSPy provides the declarative contract (field types, descriptions, instruction docstring); PydanticAI is the actual LM runtime. DSPy's `Predict` / `ChainOfThought` / `ReAct` modules are deliberately not used.
- **Each layer does one thing.** Signatures declare information flow. PydanticAI enforces output shape and handles multi-provider structured-output repair. `pydantic-graph` enforces execution order with typed state transitions. `gepa` (offline) optimizes instructions.
- **Replaces LangGraph `StateGraph`.** No LangChain dependency; cleaner composition with DSPy signatures; native support for `Agent.override()` as the GEPA injection hook.

## Files

- `toy.py` — linear two-node pipeline (Triage → Answer). Demonstrates the core composition + GEPA override hook.
- `toy_branching.py` — branching state graph with runtime routing (Triage → one of three category-specific Answer nodes). Demonstrates real FSM behavior with type-directed transitions.

## Run

```bash
uv run python toy.py
uv run python toy_branching.py
```

Requires `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` in environment. Uses `gpt-4o-mini` (Triage) and `claude-haiku-4-5` (Answer) — demonstrates heterogeneous-model composition; costs pennies per run.

## The state graph

`toy_branching.py` renders to this Mermaid diagram, auto-generated from `run()` return-type annotations:

```
stateDiagram-v2
  TriageNode --> FactualAnswerNode
  TriageNode --> ReasoningAnswerNode
  TriageNode --> CreativeAnswerNode
  FactualAnswerNode --> [*]
  ReasoningAnswerNode --> [*]
  CreativeAnswerNode --> [*]
```

Transitions are declared in the type system. Each node's `run()` method has a return annotation (e.g., `-> FactualAnswerNode | ReasoningAnswerNode | CreativeAnswerNode`) that tells `pydantic-graph` which transitions are legal. Runtime routing happens inside the node based on state content:

```python
match ctx.state.triage.category:
    case "factual":    return FactualAnswerNode()
    case "reasoning":  return ReasoningAnswerNode()
    case "creative":   return CreativeAnswerNode()
```

## The composition pattern

```python
# 1. DSPy Signature — declarative contract
class TriageSignature(dspy.Signature):
    """Classify a user question and produce a one-sentence approach plan."""
    question: str = dspy.InputField(desc="The user's question.")
    triage: Triage = dspy.OutputField(desc="Classification and approach plan.")


# 2. PydanticAI Agent — derived from the signature
def agent_from_signature(sig, model, ...) -> Agent:
    output_type = next(iter(sig.output_fields.values())).annotation
    return Agent(
        model=model,
        output_type=output_type,
        instructions=sig.instructions,
    )


# 3. pydantic-graph Node — wires the agent into the FSM
@dataclass
class TriageNode(BaseNode[PipelineState]):
    async def run(self, ctx) -> AnswerNode:
        agent = agent_from_signature(TriageSignature, model="openai:gpt-4o-mini")
        result = await agent.run(format_prompt_from_signature(TriageSignature, ...))
        ctx.state.triage = result.output
        return AnswerNode()


# 4. GEPA hook — inject optimized instructions at runtime, no mutation
with agent.override(instructions=gepa_optimized_text):
    result = await agent.run(prompt)
```

## Who owns what

| Concern | Owned by |
|---|---|
| Reasoning contract (fields, types, descriptions) | DSPy Signature |
| Default instruction text | DSPy Signature docstring |
| Optimized instruction text | External prompt store, injected via `Agent.override()` |
| Output schema enforcement + repair | PydanticAI (`output_type`, `ModelRetry`) |
| Tool calling | PydanticAI (`@agent.tool`) |
| Multi-provider output normalization | PydanticAI adapters |
| Execution order + typed state | `pydantic-graph` |
| State persistence, resumability | `pydantic-graph` (`FileStatePersistence`) |
| Prompt optimization (offline) | `gepa` package + `Agent.override()` |

## References

- DSPy: https://github.com/stanfordnlp/dspy
- PydanticAI: https://ai.pydantic.dev/
- pydantic-graph: https://ai.pydantic.dev/graph/
- GEPA: Agrawal et al. 2025 (ICLR 2026) — reflective prompt evolution
- Canonical GEPA + PydanticAI pattern (Pydantic team): https://pydantic.dev/articles/prompt-optimization-with-gepa

## Context

Validation toy for the three-layer stack committed to in the ACE-AI cognitive core architecture (ARPA-H project, Notre Dame CRC). The same pattern applies to RLM experiments, AI4C2 edge agents, and other structured agent work in LA3D.
