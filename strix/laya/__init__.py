"""Local Laya decision-model integration (optional dependency).

Laya answers typed choice/score/boolean questions about a state in a single
forward pass with calibrated probabilities — no LLM, no API key, no tokens.
Strix routes its typed decisions here first and keeps the configured LLM as
the escalation path whenever Laya is disabled, not installed, or not
confident enough. The generative agent loop always stays on the LLM.
"""
