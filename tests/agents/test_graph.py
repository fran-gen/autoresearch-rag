from src.agents.graph import _route_after_evaluator, _route_after_researcher


def test_route_after_researcher_defaults_to_planner():
    state = {"research_mode": "config"}
    assert _route_after_researcher(state) == "planner"


def test_route_after_researcher_karpathy_mode():
    state = {"research_mode": "karpathy"}
    assert _route_after_researcher(state) == "code_planner"


def test_route_after_evaluator_end_when_should_stop():
    assert _route_after_evaluator({"should_stop": True}) == "end"


def test_route_after_evaluator_loop_when_not_stopping():
    assert _route_after_evaluator({"should_stop": False}) == "researcher"
