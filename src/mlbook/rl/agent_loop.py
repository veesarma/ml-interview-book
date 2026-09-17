"""A minimal, testable agent loop: Observe -> Reason -> Act -> Observe.

The "LLM" is any callable ``policy(history) -> Action``; the tests use a
deterministic scripted policy so that the *loop* (tool dispatch, retries,
step budget, trajectory recording) can be verified independently of a model.

Tools follow a JSON-schema-like contract: ``name``, ``description``, ``parameters``
and a ``run(**kwargs) -> str`` function. Every tool call is recorded in a
:class:`Trajectory`, the unit of evaluation for agentic tasks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON-schema style {"arg": "type"}
    run: Callable[..., str]


@dataclass
class Action:
    """Either a tool call (``tool`` set) or a final answer (``final`` set)."""

    tool: str | None = None
    args: dict = field(default_factory=dict)
    final: str | None = None


@dataclass
class Step:
    action: Action
    observation: str
    ok: bool


@dataclass
class Trajectory:
    task: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    terminated_by: str = "running"  # "final" | "max_steps"

    @property
    def n_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.action.tool is not None)

    def to_json(self) -> str:
        return json.dumps(
            {
                "task": self.task,
                "final_answer": self.final_answer,
                "terminated_by": self.terminated_by,
                "steps": [
                    {"tool": s.action.tool, "args": s.action.args, "observation": s.observation, "ok": s.ok}
                    for s in self.steps
                ],
            }
        )


class ToolRegistry:
    """Dispatch by name with argument validation, a timeout-free sandbox and one retry."""

    def __init__(self, tools: list[Tool]) -> None:
        self.tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in self.tools.values()]

    def call(self, name: str, args: dict, max_attempts: int = 2) -> tuple[str, bool]:
        """Return ``(observation, ok)``; errors are *observations*, never exceptions."""
        if name not in self.tools:
            return f"error: unknown tool '{name}'", False
        tool = self.tools[name]
        missing = [p for p in tool.parameters if p not in args]
        if missing:
            return f"error: missing arguments {missing} for '{name}'", False
        last_error = ""
        for _ in range(max_attempts):
            try:
                return str(tool.run(**args)), True
            except Exception as exc:  # noqa: BLE001 - tool failures are data for the policy
                last_error = f"error: {type(exc).__name__}: {exc}"
        return last_error, False


def run_agent(task: str, policy: Callable[[list[dict]], Action], registry: ToolRegistry, max_steps: int = 8) -> Trajectory:
    """The loop. ``history`` is the list of ``{"role", "content"}`` messages the policy sees."""
    traj = Trajectory(task=task)
    history: list[dict] = [{"role": "user", "content": task}]
    for _ in range(max_steps):
        action = policy(history)  # reason
        if action.final is not None:
            traj.final_answer = action.final
            traj.terminated_by = "final"
            traj.steps.append(Step(action, "", True))
            return traj
        observation, ok = registry.call(action.tool or "", action.args)  # act
        traj.steps.append(Step(action, observation, ok))
        history.append({"role": "assistant", "content": json.dumps({"tool": action.tool, "args": action.args})})
        history.append({"role": "tool", "content": observation})  # observe
    traj.terminated_by = "max_steps"
    return traj


# --- mock tools ------------------------------------------------------------------

_SAFE_CHARS = set("0123456789+-*/(). ")


def calculator(expression: str) -> str:
    """Arithmetic on a whitelisted character set (a stand-in for a sandboxed code tool)."""
    if not set(expression) <= _SAFE_CHARS:
        raise ValueError("expression contains non-arithmetic characters")
    return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 - whitelisted characters only


def make_lookup(table: dict[str, str]) -> Callable[..., str]:
    def lookup(key: str) -> str:
        if key not in table:
            raise KeyError(key)
        return table[key]

    return lookup


def default_tools(table: dict[str, str]) -> ToolRegistry:
    return ToolRegistry(
        [
            Tool("calculator", "Evaluate an arithmetic expression.", {"expression": "string"}, calculator),
            Tool("lookup", "Look up a fact by key.", {"key": "string"}, make_lookup(table)),
        ]
    )


def scripted_policy(script: list[Action]) -> Callable[[list[dict]], Action]:
    """A deterministic 'LLM': returns the scripted actions in order, then a fallback final answer."""
    state = {"i": 0}

    def policy(history: list[dict]) -> Action:
        i = state["i"]
        state["i"] = i + 1
        if i < len(script):
            return script[i]
        last_obs = history[-1]["content"] if history and history[-1]["role"] == "tool" else ""
        return Action(final=last_obs)

    return policy


def task_success(traj: Trajectory, expected: str) -> bool:
    """Outcome-based evaluation: did the final answer match, regardless of the path?"""
    return traj.terminated_by == "final" and traj.final_answer is not None and traj.final_answer.strip() == expected
