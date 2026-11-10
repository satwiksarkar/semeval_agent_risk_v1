"""
state.py
State definitions for the LangGraph Trajectory Simulator & Verifier.

SemEval 2027 Task 10 (AgentRisk) - Phase 2
"""

from typing import TypedDict


class SimulationState(TypedDict):
    """
    State representing an agent trajectory evaluated by the LangGraph verifier node.
    """
    trajectory_id: str
    tokenized_text: str
    hallucination_flag: bool
    risk_modifier: float

