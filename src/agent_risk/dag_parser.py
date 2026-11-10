"""
dag_parser.py
TrajectoryDAG: Parses multi-turn agent interaction trajectories into Directed Acyclic Graphs (DAGs),
models state transitions and causal links (API calls -> Observations), prunes dead-end error branches,
extracts topological adjacency matrices, and produces structurally tokenized text representations.

SemEval 2027 Task 10 (AgentRisk) - Subtask 1
Supports both official SemEval formatted_interaction strings and structured step dictionaries.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx

logger = logging.getLogger(__name__)

# Standard structural tokens for Phase 1 topological sequence representation
TOKEN_USER_INTENT_START = "<|user_intent|>"
TOKEN_USER_INTENT_END = "</|user_intent|>"
TOKEN_THOUGHT_START = "<|thought|>"
TOKEN_THOUGHT_END = "</|thought|>"
TOKEN_API_CALL_START = "<|api_call|>"
TOKEN_API_CALL_END = "</|api_call|>"
TOKEN_OBSERVATION_START = "<|observation|>"
TOKEN_OBSERVATION_END = "</|observation|>"
TOKEN_PROPOSED_ACTION_START = "<|proposed_action|>"
TOKEN_PROPOSED_ACTION_END = "</|proposed_action|>"


class TrajectoryDAG:
    """
    Parses a multi-turn autonomous agent interaction trajectory into a Directed Acyclic Graph (DAG),
    enforces causal dependencies (Action -> Observation -> Next Step), prunes unreferenced error
    branches that do not connect to the final evaluated action, and produces both an adjacency
    matrix and structurally tokenized text representation.
    """

    def __init__(
        self,
        raw_interaction: Union[Dict[str, Any], str],
        scale_to_1_5: bool = True,
    ):
        """
        Initialize and construct the TrajectoryDAG.

        Args:
            raw_interaction: Dict or JSON string representing the agent trajectory.
            scale_to_1_5: If True and the raw risk level is 0-4 (with 0 present), maps to the 1-5 ordinal scale.
        """
        if isinstance(raw_interaction, str):
            self.data = json.loads(raw_interaction)
        elif isinstance(raw_interaction, dict):
            self.data = raw_interaction
        else:
            raise TypeError(f"Expected dict or JSON str, got {type(raw_interaction).__name__}")

        self.scale_to_1_5 = scale_to_1_5
        self.instance_id: str = str(
            self.data.get("example_id")
            or self.data.get("id")
            or self.data.get("instance_id")
            or self.data.get("trace_id")
            or "trace-001"
        )
        self.goal: str = str(
            self.data.get("goal") or self.data.get("intent") or self.data.get("user_intent") or ""
        )
        self.trajectory_steps: List[Dict[str, Any]] = (
            self.data.get("trajectory") or self.data.get("steps") or []
        )
        self.proposed_action_data: Any = self.data.get("proposed_action") or {}
        self.risk_domains: List[str] = self.data.get("risk_domains", [])
        self.risk_drivers: List[str] = self.data.get("risk_drivers", [])

        # Parse formatted_interaction string if present (official SemEval 2027 AgentRisk format)
        formatted_text = self.data.get("formatted_interaction")
        if formatted_text and not self.trajectory_steps:
            parsed_steps, parsed_goal, parsed_proposed = self._parse_formatted_interaction(formatted_text)
            self.trajectory_steps = parsed_steps
            if not self.goal and parsed_goal:
                self.goal = parsed_goal
            if not self.proposed_action_data and parsed_proposed:
                self.proposed_action_data = parsed_proposed

        # Risk level parsing (SemEval Task 10 Subtask 1: 1-5 ordinal scale)
        raw_risk = self.data.get("risk_severity", self.data.get("risk_level", self.data.get("risk", 1)))
        try:
            risk_val = float(raw_risk)
            # If dataset is 0..4 (0 present and max <= 4), map to 1..5.
            # If already 1..5 (e.g. risk_severity in SemEval 2027), retain as is.
            if self.scale_to_1_5 and "risk_severity" not in self.data and 0.0 <= risk_val <= 4.0:
                self.risk_level = risk_val + 1.0
            else:
                self.risk_level = risk_val
        except (ValueError, TypeError):
            self.risk_level = 1.0

        # Graph initialization
        self.graph: nx.DiGraph = nx.DiGraph()
        self.root_node: str = "node_user_intent"
        self.final_node: str = "node_proposed_action"
        self.pruned_node_count: int = 0

        # Build graph and prune dead-end error branches
        self._build_dag()
        self.prune_failed_branches()

    def _parse_formatted_interaction(
        self, formatted_text: str
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        """
        Parses official SemEval 2027 formatted_interaction string:
        Step 1. [USER]: ...
        Step 2. [THOUGHT]: ...
        Step 3. [ACTION]: ...
        Step 4. [OBSERVATION]: ...
        ...
        Step N. [ACTION]: ... (Target proposed action)
        """
        pattern = r"(?m)^Step\s+(\d+)\.\s+\[(USER|THOUGHT|ACTION|OBSERVATION)\]:\s*"
        splits = re.split(pattern, formatted_text)

        goal = ""
        steps: List[Dict[str, Any]] = []
        current_thought: Optional[str] = None
        current_action: Optional[str] = None
        current_step_id = 1
        proposed_action_dict: Dict[str, Any] = {}

        elements: List[Tuple[int, str, str]] = []
        for i in range(1, len(splits), 3):
            step_num = int(splits[i])
            tag = splits[i + 1]
            content = splits[i + 2].strip()
            elements.append((step_num, tag, content))

        for idx, (num, tag, content) in enumerate(elements):
            if tag == "USER":
                if not goal:
                    goal = content
            elif tag == "THOUGHT":
                current_thought = content
            elif tag == "ACTION":
                is_final_action = (idx == len(elements) - 1) or (
                    idx == len(elements) - 2 and elements[-1][1] == "THOUGHT"
                )
                if is_final_action:
                    proposed_action_dict = {
                        "action": content,
                        "thought": current_thought,
                        "description": content,
                    }
                else:
                    current_action = content
            elif tag == "OBSERVATION":
                steps.append({
                    "step_id": current_step_id,
                    "thought": current_thought,
                    "action": current_action or "call",
                    "observation": content,
                    "is_error": False,  # Evaluated by _is_error_step
                })
                current_step_id += 1
                current_thought = None
                current_action = None

        return steps, goal, proposed_action_dict

    def _is_error_step(self, step: Dict[str, Any]) -> bool:
        """Determines whether a trajectory step resulted in an error/failure."""
        if step.get("is_error") is True:
            return True
        status = str(step.get("status", "")).strip().lower()
        if status in {"error", "failed", "failure", "exception", "rejected"}:
            return True
        obs = str(step.get("observation", step.get("output", step.get("result", "")))).strip()
        obs_lower = obs.lower()
        error_indicators = (
            "error:",
            "exception:",
            "traceback (most recent call last)",
            "command failed",
            "permission denied",
            "failed to execute",
            "access denied",
            "404 not found",
            "500 internal server error",
        )
        return any(obs_lower.startswith(ind) or ind in obs_lower[:120] for ind in error_indicators)

    def _format_action_content(self, action_dict_or_str: Union[Dict[str, Any], str]) -> str:
        """Formats an action into a clean string representation."""
        if isinstance(action_dict_or_str, str):
            return action_dict_or_str
        if not isinstance(action_dict_or_str, dict):
            return str(action_dict_or_str)

        tool = action_dict_or_str.get("tool") or action_dict_or_str.get("action") or "call"
        inputs = action_dict_or_str.get("inputs") or action_dict_or_str.get("parameters") or {}
        if isinstance(inputs, dict):
            args_str = ", ".join(f"{k}={v}" for k, v in inputs.items())
        else:
            args_str = str(inputs)
        return f"{tool}({args_str})"

    def _build_dag(self) -> None:
        """
        Constructs directed causal edges linking User Intent -> Thoughts -> API Calls -> Observations,
        and finally linking to the Proposed Action.
        """
        # 1. Root Node: User Intent
        intent_text = self.goal if self.goal else "Execute autonomous task"
        self.graph.add_node(
            self.root_node,
            node_type="user_intent",
            content=intent_text,
            is_error=False,
            step_idx=0,
        )

        prev_obs_node = self.root_node
        last_active_nodes: List[str] = [self.root_node]

        # 2. Sequential / Causal Trajectory Steps
        for idx, step in enumerate(self.trajectory_steps, start=1):
            step_id = step.get("step_id", idx)
            thought_text = step.get("thought") or step.get("reasoning")
            action_data = step.get("action") or step.get("tool_call") or step.get("tool") or {}
            obs_data = step.get("observation") or step.get("output") or step.get("result") or ""
            is_error = self._is_error_step(step)

            thought_node_id = f"thought_step_{step_id}"
            action_node_id = f"action_step_{step_id}"
            obs_node_id = f"obs_step_{step_id}"

            # Check explicit dependencies if present
            explicit_deps = step.get("dependencies") or step.get("parent_ids")
            if explicit_deps:
                parent_nodes = [f"obs_step_{d}" for d in explicit_deps if f"obs_step_{d}" in self.graph]
                if not parent_nodes:
                    parent_nodes = [self.root_node]
            else:
                parent_nodes = [prev_obs_node] if prev_obs_node in self.graph else [self.root_node]

            # Add Thought Node if present
            if thought_text:
                self.graph.add_node(
                    thought_node_id,
                    node_type="thought",
                    content=str(thought_text).strip(),
                    is_error=False,
                    step_idx=idx,
                )
                for p in parent_nodes:
                    self.graph.add_edge(p, thought_node_id)

            # Add Action / API Call Node
            formatted_action = self._format_action_content(action_data)
            self.graph.add_node(
                action_node_id,
                node_type="api_call",
                content=formatted_action,
                is_error=False,
                step_idx=idx,
                tool=str(step.get("tool", "")),
            )
            if thought_text:
                self.graph.add_edge(thought_node_id, action_node_id)
            else:
                for p in parent_nodes:
                    self.graph.add_edge(p, action_node_id)

            # Add Observation Node (Explicit causal edge: API Call -> Observation)
            self.graph.add_node(
                obs_node_id,
                node_type="observation",
                content=str(obs_data).strip(),
                is_error=is_error,
                step_idx=idx,
            )
            self.graph.add_edge(action_node_id, obs_node_id)

            prev_obs_node = obs_node_id
            last_active_nodes = [obs_node_id]

        # 3. Final Proposed Action
        prop_thought = None
        if isinstance(self.proposed_action_data, dict):
            prop_thought = self.proposed_action_data.get("thought")
            prop_content = self._format_action_content(self.proposed_action_data)
        else:
            prop_content = str(self.proposed_action_data).strip()

        if not prop_content or prop_content == "call()":
            prop_content = str(self.data.get("proposed_action_text", "Target proposed action"))

        # Add pre-action thought if present
        prop_entry_node = self.final_node
        if prop_thought:
            prop_thought_node = "thought_proposed"
            self.graph.add_node(
                prop_thought_node,
                node_type="thought",
                content=str(prop_thought).strip(),
                is_error=False,
                step_idx=len(self.trajectory_steps) + 1,
            )
            for node in last_active_nodes:
                if node in self.graph:
                    self.graph.add_edge(node, prop_thought_node)
            prop_entry_node = prop_thought_node

        self.graph.add_node(
            self.final_node,
            node_type="proposed_action",
            content=prop_content,
            is_error=False,
            step_idx=len(self.trajectory_steps) + 2,
        )

        if prop_thought:
            self.graph.add_edge(prop_entry_node, self.final_node)
        else:
            connected_to_final = False
            for node in last_active_nodes:
                if node in self.graph and node != self.final_node:
                    self.graph.add_edge(node, self.final_node)
                    connected_to_final = True
            if not connected_to_final:
                self.graph.add_edge(self.root_node, self.final_node)

        # Sanity check: Ensure acyclic property
        if not nx.is_directed_acyclic_graph(self.graph):
            logger.warning(f"Cycle detected in graph for {self.instance_id}; resolving cycles.")
            cycles = list(nx.simple_cycles(self.graph))
            for cycle in cycles:
                if len(cycle) >= 2:
                    self.graph.remove_edge(cycle[-1], cycle[0])

    def prune_failed_branches(self) -> nx.DiGraph:
        """
        Prunes branches ending in errors that do not connect to the final action.
        
        Logic:
        1. An error observation node is checked for reachability to self.final_node.
        2. If nx.has_path(graph, error_node, final_node) is True, the agent leveraged
           or pivoted from the error to reach the final action; the node is PRESERVED.
        3. If False, the error node was an abandoned dead-end. It is pruned along with
           any preceding parent nodes in that failed branch whose only descendant was
           that error (out-degree 0 after leaf removal).
        """
        if not self.graph or self.final_node not in self.graph:
            return self.graph

        while True:
            pruned_in_this_pass = False
            candidate_leaves = [
                n for n in self.graph.nodes()
                if n not in (self.final_node, self.root_node) and self.graph.out_degree(n) == 0
            ]

            for leaf in candidate_leaves:
                if not nx.has_path(self.graph, leaf, self.final_node):
                    is_err = self.graph.nodes[leaf].get("is_error", False)
                    if is_err or self.graph.nodes[leaf].get("node_type") in ("api_call", "thought"):
                        self.graph.remove_node(leaf)
                        self.pruned_node_count += 1
                        pruned_in_this_pass = True
                        break

            if not pruned_in_this_pass:
                break

        if not nx.has_path(self.graph, self.root_node, self.final_node):
            self.graph.add_edge(self.root_node, self.final_node)

        return self.graph

    def get_topological_order(self) -> List[str]:
        """Returns the list of node identifiers in topological sort order."""
        return list(nx.topological_sort(self.graph))

    def get_adjacency_matrix(self) -> List[List[int]]:
        """
        Extracts the adjacency matrix of the pruned DAG as a list of lists,
        strictly ordered according to the topological sort.
        """
        topo_order = self.get_topological_order()
        n = len(topo_order)
        node_to_idx = {node: i for i, node in enumerate(topo_order)}
        adj: List[List[int]] = [[0] * n for _ in range(n)]

        for u, v in self.graph.edges():
            if u in node_to_idx and v in node_to_idx:
                adj[node_to_idx[u]][node_to_idx[v]] = 1

        return adj

    def get_topological_sequence(self) -> str:
        """
        Returns a topologically sorted string wrapped in structural tokens:
        <|user_intent|>, <|thought|>, <|api_call|>, <|observation|>, <|proposed_action|>.
        """
        topo_order = self.get_topological_order()
        token_map = {
            "user_intent": (TOKEN_USER_INTENT_START, TOKEN_USER_INTENT_END),
            "thought": (TOKEN_THOUGHT_START, TOKEN_THOUGHT_END),
            "api_call": (TOKEN_API_CALL_START, TOKEN_API_CALL_END),
            "observation": (TOKEN_OBSERVATION_START, TOKEN_OBSERVATION_END),
            "proposed_action": (TOKEN_PROPOSED_ACTION_START, TOKEN_PROPOSED_ACTION_END),
        }

        elements: List[str] = []
        for node in topo_order:
            node_data = self.graph.nodes[node]
            node_type = node_data.get("node_type", "observation")
            content = node_data.get("content", "").strip()
            start_tag, end_tag = token_map.get(node_type, (f"<|{node_type}|>", f"</|{node_type}|>"))
            elements.append(f"{start_tag}\n{content}\n{end_tag}")

        return "\n\n".join(elements)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the processed DAG into a standard dictionary containing the
        tokenized text, adjacency matrix, metadata, and 1-5 ordinal risk score.
        """
        return {
            "id": self.instance_id,
            "goal": self.goal,
            "tokenized_text": self.get_topological_sequence(),
            "adjacency_matrix": self.get_adjacency_matrix(),
            "num_nodes": len(self.graph),
            "num_edges": self.graph.number_of_edges(),
            "risk_level": self.risk_level,
            "risk_domains": self.risk_domains,
            "risk_drivers": self.risk_drivers,
            "topological_nodes": self.get_topological_order(),
            "pruned_nodes_count": self.pruned_node_count,
        }
