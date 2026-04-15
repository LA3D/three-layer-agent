"""Fitness coach toy — multi-session agentic demo of the three-layer stack.

Architectural parallel to the ACE-AI cognitive core, demonstrating:
  - Multi-session longitudinal state with handoff document
  - Typed state-transition reasoning with formal constraint validation
  - Evidence with provenance (multi-stream trust-weighted)
  - Per-step validation with deterministic fallback
  - Hard safety overrides
  - Cross-population generality (powerlifters + runners, same architecture)
  - Side-by-side comparison vs. rigid expert-system "straw coach"

Reuses `agent_from_signature` and `format_prompt_from_signature` from the root
`toy.py` module — those are the canonical Signature → Agent bridge.
"""
