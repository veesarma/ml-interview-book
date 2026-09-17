# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/capstone/tool_loop.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k tool_loop -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py capstone/tool_loop --force

"""Canon #60, part 8 -- the agent loop: observe, reason, act, observe again.

An agent is a policy that gets to take actions before it answers. The loop here
has two kinds of action and a hard step limit:

* ``call_tool``: run ``count_shapes`` on the scene and splice the result back
  into the context as ``observation : <n>``.
* ``answer``: emit an answer token and stop.

The controller is deterministic and readable on purpose: the model proposes an
answer, and the loop calls the tool when the model's top-token probability is
below ``confidence_threshold`` and the question is inside the tool's domain.
Confidence-gated tool use is what most production scaffolds actually do, and it
makes the cost of the loop explicit (one extra forward pass per gated example).

The tool has a domain, which is the part people forget. ``count_shapes`` cannot
answer "what colour is the largest shape", so those examples are never gated and
the loop must not pretend otherwise.

Every episode is recorded as a :class:`Trajectory`, which is what you would log
in production and what you replay for evaluation or for RL on tool use.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
from mlbook.capstone import synthetic_task as task
from mlbook.capstone.multimodal_model import TinyVLM
from mlbook.capstone.reward_stage import verifier_reward
from mlbook.capstone.sft_stage import next_token_logits
TOOL_NAME = 'count_shapes'

def count_shapes_tool(scene: task.Scene, colour: str | None=None, kind: str | None=None) -> str:
    """The one tool the agent can call. Returns the count as a vocabulary token.

    It reads the scene graph, which is the mock stand-in for a detector, a
    database query or a Python interpreter. Its output is a string because that
    is what gets tokenised back into the context.
    """
    raise NotImplementedError('TODO: implement count_shapes_tool (see the reference in src/mlbook)')

@dataclass
class ToolStep:
    """One turn of the loop."""
    index: int
    observation: str
    thought: str
    action: str
    action_input: str
    result: str

@dataclass
class Trajectory:
    """One episode: the steps taken, the answer given, and the verifier's score."""
    question: str
    gold_answer: str
    steps: list[ToolStep] = field(default_factory=list)
    final_answer: str = ''
    reward: float = 0.0
    used_tool: bool = False

    def is_well_formed(self) -> bool:
        """A trajectory is well formed if it has steps, they are numbered from 0
        in order, every action is known, and exactly the last step answers."""
        raise NotImplementedError('TODO: implement is_well_formed (see the reference in src/mlbook)')

@torch.no_grad()
def run_episodes(model: TinyVLM, examples: list[task.Example], confidence_threshold: float=0.9) -> list[Trajectory]:
    """Run the loop over a batch of examples. Returns one :class:`Trajectory` each.

    Two batched forward passes at most: one for the initial proposal, one for
    the examples that were gated to the tool.
    """
    raise NotImplementedError('TODO: implement run_episodes (see the reference in src/mlbook)')

def run_episode(model: TinyVLM, example: task.Example, confidence_threshold: float=0.9) -> Trajectory:
    """Single-example convenience wrapper around :func:`run_episodes`."""
    raise NotImplementedError('TODO: implement run_episode (see the reference in src/mlbook)')

def trajectory_report(trajectories: list[Trajectory]) -> dict[str, float]:
    """Aggregate accuracy, tool-call rate and mean episode length."""
    raise NotImplementedError('TODO: implement trajectory_report (see the reference in src/mlbook)')

def render_trajectory(traj: Trajectory) -> str:
    """Human-readable transcript, the thing you paste into a bug report."""
    raise NotImplementedError('TODO: implement render_trajectory (see the reference in src/mlbook)')
