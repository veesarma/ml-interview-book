# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/rl/agent_loop.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k agent_loop -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py rl/agent_loop --force

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
    parameters: dict
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
    terminated_by: str = 'running'

    @property
    def n_tool_calls(self) -> int:
        raise NotImplementedError('TODO: implement n_tool_calls (see the reference in src/mlbook)')

    def to_json(self) -> str:
        raise NotImplementedError('TODO: implement to_json (see the reference in src/mlbook)')

class ToolRegistry:
    """Dispatch by name with argument validation, a timeout-free sandbox and one retry."""

    def __init__(self, tools: list[Tool]) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def schemas(self) -> list[dict]:
        raise NotImplementedError('TODO: implement schemas (see the reference in src/mlbook)')

    def call(self, name: str, args: dict, max_attempts: int=2) -> tuple[str, bool]:
        """Return ``(observation, ok)``; errors are *observations*, never exceptions."""
        raise NotImplementedError('TODO: implement call (see the reference in src/mlbook)')

def run_agent(task: str, policy: Callable[[list[dict]], Action], registry: ToolRegistry, max_steps: int=8) -> Trajectory:
    """The loop. ``history`` is the list of ``{"role", "content"}`` messages the policy sees."""
    raise NotImplementedError('TODO: implement run_agent (see the reference in src/mlbook)')
_SAFE_CHARS = set('0123456789+-*/(). ')

def calculator(expression: str) -> str:
    """Arithmetic on a whitelisted character set (a stand-in for a sandboxed code tool)."""
    raise NotImplementedError('TODO: implement calculator (see the reference in src/mlbook)')

def make_lookup(table: dict[str, str]) -> Callable[..., str]:
    raise NotImplementedError('TODO: implement make_lookup (see the reference in src/mlbook)')

def default_tools(table: dict[str, str]) -> ToolRegistry:
    raise NotImplementedError('TODO: implement default_tools (see the reference in src/mlbook)')

def scripted_policy(script: list[Action]) -> Callable[[list[dict]], Action]:
    """A deterministic 'LLM': returns the scripted actions in order, then a fallback final answer."""
    raise NotImplementedError('TODO: implement scripted_policy (see the reference in src/mlbook)')

def task_success(traj: Trajectory, expected: str) -> bool:
    """Outcome-based evaluation: did the final answer match, regardless of the path?"""
    raise NotImplementedError('TODO: implement task_success (see the reference in src/mlbook)')
