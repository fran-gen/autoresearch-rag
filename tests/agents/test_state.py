from src.agents.graph import default_state


def test_default_state_has_expected_basics():
    state = default_state(max_iterations=5)

    assert state["iteration"] == 0
    assert state["max_iterations"] == 5
    assert state["should_stop"] is False
    assert state["current_phase"] == "init"
    assert state["run_id"].startswith("run_")
    assert state["best_config"] is not None
    assert state["best_config"].strategy == "dense"
    assert state["best_score"] == -1.0
    assert state["accepted_experiments"] == 0
    assert state["rejected_experiments"] == 0
