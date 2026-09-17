"""DPO on the toy LM: implicit reward margin beta*[log(pi/pi_ref)(y_w) - log(pi/pi_ref)(y_l)]
and the chosen / rejected implicit rewards over training steps, showing both dropping below
zero (likelihood displacement) while the margin grows.
Writes docs/assets/figures/part07_dpo_margin.png.
Run from the repository root:  python figures/part07_dpo_margin.py
"""
from __future__ import annotations

import copy
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mlbook.posttrain.chat_template import ToyTokenizer, tokenize_chat  # noqa: E402
from mlbook.posttrain.dpo import dpo_train_step  # noqa: E402
from mlbook.posttrain.sft import pack_examples, train_sft  # noqa: E402
from mlbook.posttrain.toy_lm import TinyCausalLM, ToyLMConfig, sequence_log_prob  # noqa: E402
from mlbook.posttrain.verifiers import all_arithmetic_pairs, arithmetic_answer, arithmetic_messages  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part07_dpo_margin.png"


def build_pairs(tok: ToyTokenizer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chosen = correct answer, rejected = wrong digit. Returns (chosen, chosen_mask, rejected, rejected_mask)."""
    chosen_rows, rejected_rows, masks_c, masks_r = [], [], [], []
    for a, b in all_arithmetic_pairs():
        ans = arithmetic_answer(a, b)
        wrong = str((int(ans) + 3) % 10)
        ids_c, m_c = tokenize_chat(arithmetic_messages(a, b) + [{"role": "assistant", "content": ans}], tok)
        ids_r, m_r = tokenize_chat(arithmetic_messages(a, b) + [{"role": "assistant", "content": wrong}], tok)
        chosen_rows.append(ids_c), rejected_rows.append(ids_r), masks_c.append(m_c), masks_r.append(m_r)
    return (torch.tensor(chosen_rows), torch.tensor(masks_c, dtype=torch.float),
            torch.tensor(rejected_rows), torch.tensor(masks_r, dtype=torch.float))


def main() -> None:
    torch.manual_seed(0)
    tok = ToyTokenizer()
    cfg = ToyLMConfig(vocab_size=tok.vocab_size, d_model=32, n_heads=2, n_layers=2, max_len=32)
    policy = TinyCausalLM(cfg)
    # Light SFT on a mix of correct and wrong answers so both responses start plausible.
    examples = []
    for a, b in all_arithmetic_pairs():
        for ans in (arithmetic_answer(a, b), str((int(arithmetic_answer(a, b)) + 3) % 10)):
            examples.append(tokenize_chat(arithmetic_messages(a, b) + [{"role": "assistant", "content": ans}], tok))
    train_sft(policy, [pack_examples(examples, max_len=20, pad_id=tok.pad_id)], epochs=40, lr=1e-2)
    ref = copy.deepcopy(policy)
    chosen, m_c, rejected, m_r = build_pairs(tok)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    beta = 0.1
    steps, margins, r_chosen, r_rejected = [], [], [], []
    for step in range(120):
        dpo_train_step(policy, ref, opt, chosen, m_c, rejected, m_r, beta)
        with torch.no_grad():
            rc = beta * (sequence_log_prob(policy, chosen, m_c) - sequence_log_prob(ref, chosen, m_c))  # (B,)
            rr = beta * (sequence_log_prob(policy, rejected, m_r) - sequence_log_prob(ref, rejected, m_r))  # (B,)
        steps.append(step), margins.append(float((rc - rr).mean()))
        r_chosen.append(float(rc.mean())), r_rejected.append(float(rr.mean()))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor="white")
    axes[0].plot(steps, margins, color=colors[0], linewidth=2)
    axes[0].set_xlabel("DPO step")
    axes[0].set_ylabel("implicit reward margin")
    axes[0].set_title("Margin  $\\beta[\\log\\frac{\\pi}{\\pi_{ref}}(y_w) - \\log\\frac{\\pi}{\\pi_{ref}}(y_l)]$ rises", fontsize=10, loc="left")
    axes[1].plot(steps, r_chosen, color=colors[2], linewidth=2, label="chosen  $\\beta\\log\\frac{\\pi}{\\pi_{ref}}(y_w)$")
    axes[1].plot(steps, r_rejected, color=colors[3], linewidth=2, label="rejected  $\\beta\\log\\frac{\\pi}{\\pi_{ref}}(y_l)$")
    axes[1].axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("DPO step")
    axes[1].set_title("Both implicit rewards can fall: only the gap is constrained", fontsize=10, loc="left")
    axes[1].legend(fontsize=8, frameon=False)
    for a in axes:
        a.grid(color="#eeeeee", linewidth=0.5)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
