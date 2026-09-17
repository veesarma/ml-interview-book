"""The agent loop: dispatch, validation, retries, step budget, trajectory recording and outcome evaluation."""
import json

from mlbook.rl.agent_loop import (
    Action,
    Tool,
    ToolRegistry,
    Trajectory,
    calculator,
    default_tools,
    run_agent,
    scripted_policy,
    task_success,
)

TABLE = {"population_of_x": "1200", "population_of_y": "3400"}


def test_tool_registry_validates_and_reports_errors_as_observations():
    reg = default_tools(TABLE)
    assert reg.call("calculator", {"expression": "2*(3+4)"}) == ("14", True)
    assert reg.call("calculator", {})[1] is False
    assert reg.call("nope", {})[0].startswith("error: unknown tool")
    obs, ok = reg.call("lookup", {"key": "missing"})
    assert not ok and "KeyError" in obs
    obs, ok = reg.call("calculator", {"expression": "__import__('os')"})
    assert not ok and "ValueError" in obs


def test_registry_retries_flaky_tool():
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("slow")
        return "ok"

    reg = ToolRegistry([Tool("flaky", "fails once", {}, flaky)])
    assert reg.call("flaky", {}) == ("ok", True) and calls["n"] == 2


def test_run_agent_multi_step_task_and_trajectory():
    script = [
        Action(tool="lookup", args={"key": "population_of_x"}),
        Action(tool="lookup", args={"key": "population_of_y"}),
        Action(tool="calculator", args={"expression": "1200 + 3400"}),
        Action(final="4600"),
    ]
    traj = run_agent("total population of x and y?", scripted_policy(script), default_tools(TABLE), max_steps=8)
    assert traj.terminated_by == "final" and traj.n_tool_calls == 3
    assert [s.observation for s in traj.steps[:3]] == ["1200", "3400", "4600"]
    assert task_success(traj, "4600") and not task_success(traj, "4601")
    record = json.loads(traj.to_json())
    assert record["steps"][2]["tool"] == "calculator" and record["final_answer"] == "4600"


def test_run_agent_stops_at_step_budget():
    looping = [Action(tool="calculator", args={"expression": "1+1"})] * 20
    traj = run_agent("loop forever", scripted_policy(looping), default_tools(TABLE), max_steps=5)
    assert traj.terminated_by == "max_steps" and traj.n_tool_calls == 5 and traj.final_answer is None
    assert not task_success(traj, "2")


def test_scripted_policy_falls_back_to_last_observation():
    script = [Action(tool="calculator", args={"expression": "6*7"})]
    traj = run_agent("what is 6*7", scripted_policy(script), default_tools(TABLE))
    assert traj.final_answer == "42" and task_success(traj, "42")


def test_calculator_rejects_non_arithmetic():
    assert calculator("(1+2)*3") == "9"
    try:
        calculator("a+1")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_trajectory_counts_only_tool_calls():
    t = Trajectory(task="t")
    assert t.n_tool_calls == 0
