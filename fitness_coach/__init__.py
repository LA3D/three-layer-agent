"""Fitness coach toy — multi-session agentic demo of the three-layer stack.

A worked example of architectural patterns that single-session toys don't
exercise:
  - Multi-session longitudinal state with a generated handoff document
  - Typed state-transition reasoning with formal constraint validation
  - Evidence with provenance (multi-stream trust-weighted)
  - Per-step validation with deterministic fallback
  - Tiered safety overrides
  - Cross-population generality (powerlifters + runners, same architecture)
  - Side-by-side comparison vs. rigid expert-system "straw coach"

Reuses `agent_from_signature` and `format_prompt_from_signature` from the root
`toy.py` module — those are the canonical Signature → Agent bridge.
"""
