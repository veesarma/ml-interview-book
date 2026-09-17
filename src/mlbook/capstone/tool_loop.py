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

TOOL_NAME = "count_shapes"


def count_shapes_tool(scene: task.Scene, colour: str | None = None, kind: str | None = None) -> str:
    """The one tool the agent can call. Returns the count as a vocabulary token.

    It reads the scene graph, which is the mock stand-in for a detector, a
    database query or a Python interpreter. Its output is a string because that
    is what gets tokenised back into the context.
    """
    return str(task.count_shapes(scene, colour, kind))


@dataclass
class ToolStep:
    """One turn of the loop."""

    index: int
    observation: str    # what the agent had in context entering this step
    thought: str        # the model's own proposal and how sure it was
    action: str         # "call_tool" or "answer"
    action_input: str   # tool arguments, or the answer token
    result: str         # tool output, or "" for a terminal answer


@dataclass
class Trajectory:
    """One episode: the steps taken, the answer given, and the verifier's score."""

    question: str
    gold_answer: str
    steps: list[ToolStep] = field(default_factory=list)
    final_answer: str = ""
    reward: float = 0.0
    used_tool: bool = False

    def is_well_formed(self) -> bool:
        """A trajectory is well formed if it has steps, they are numbered from 0
        in order, every action is known, and exactly the last step answers."""
        if not self.steps:
            return False
        if [s.index for s in self.steps] != list(range(len(self.steps))):
            return False
        if any(s.action not in ("call_tool", "answer") for s in self.steps):
            return False
        return [s.action for s in self.steps].count("answer") == 1 and self.steps[-1].action == "answer"


@torch.no_grad()
def run_episodes(
    model: TinyVLM,
    examples: list[task.Example],
    confidence_threshold: float = 0.9,
) -> list[Trajectory]:
    """Run the loop over a batch of examples. Returns one :class:`Trajectory` each.

    Two batched forward passes at most: one for the initial proposal, one for
    the examples that were gated to the tool.
    """
    was_training = model.training
    model.eval()

    logits = next_token_logits(model, examples)                          # (B, V)
    probs = torch.softmax(logits, dim=-1)                                # (B, V)
    confidence, predicted = probs.max(dim=-1)                            # (B,), (B,)

    trajectories: list[Trajectory] = []
    gated: list[int] = []
    observations: list[str] = []
    for i, ex in enumerate(examples):
        proposal = task.ITOS[int(predicted[i])]
        conf = float(confidence[i])
        traj = Trajectory(question=ex.question, gold_answer=ex.answer)
        thought = f"best guess {proposal!r} with p={conf:.2f}"
        if ex.tool_query is not None and conf < confidence_threshold:
            colour, kind = ex.tool_query
            result = count_shapes_tool(ex.scene, colour, kind)
            traj.steps.append(ToolStep(
                index=0,
                observation=f"image + question: {ex.question}",
                thought=thought + ", below threshold, and the tool covers this question",
                action="call_tool",
                action_input=f"{TOOL_NAME}(colour={colour}, kind={kind})",
                result=result,
            ))
            traj.used_tool = True
            gated.append(i)
            observations.append(result)
        else:
            traj.steps.append(ToolStep(
                index=0,
                observation=f"image + question: {ex.question}",
                thought=thought + ", confident enough to answer" if conf >= confidence_threshold
                else thought + ", but the tool cannot answer this question",
                action="answer",
                action_input=proposal,
                result="",
            ))
            traj.final_answer = proposal
        trajectories.append(traj)

    if gated:
        second = next_token_logits(model, [examples[i] for i in gated], observations)  # (n_gated, V)
        answers = second.argmax(dim=-1)                                  # (n_gated,)
        for slot, i in enumerate(gated):
            answer = task.ITOS[int(answers[slot])]
            traj = trajectories[i]
            traj.steps.append(ToolStep(
                index=1,
                observation=f"observation : {observations[slot]}",
                thought="re-reading the question with the tool result in context",
                action="answer",
                action_input=answer,
                result="",
            ))
            traj.final_answer = answer

    for traj, ex in zip(trajectories, examples):
        traj.reward = verifier_reward(traj.final_answer, ex)

    model.train(was_training)
    return trajectories


def run_episode(model: TinyVLM, example: task.Example, confidence_threshold: float = 0.9) -> Trajectory:
    """Single-example convenience wrapper around :func:`run_episodes`."""
    return run_episodes(model, [example], confidence_threshold)[0]


def trajectory_report(trajectories: list[Trajectory]) -> dict[str, float]:
    """Aggregate accuracy, tool-call rate and mean episode length."""
    n = max(len(trajectories), 1)
    return {
        "accuracy": sum(t.reward for t in trajectories) / n,
        "tool_call_rate": sum(1.0 for t in trajectories if t.used_tool) / n,
        "mean_steps": sum(len(t.steps) for t in trajectories) / n,
        "well_formed_rate": sum(1.0 for t in trajectories if t.is_well_formed()) / n,
    }


def render_trajectory(traj: Trajectory) -> str:
    """Human-readable transcript, the thing you paste into a bug report."""
    lines = [f"Q: {traj.question}   (gold {traj.gold_answer})"]
    for step in traj.steps:
        lines.append(f"  [{step.index}] obs    {step.observation}")
        lines.append(f"      thought  {step.thought}")
        lines.append(f"      action   {step.action}({step.action_input})")
        if step.result:
            lines.append(f"      result   {step.result}")
    lines.append(f"  answer: {traj.final_answer}   reward {traj.reward:.0f}")
    return "\n".join(lines)
