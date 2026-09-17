"""Per-GPU training memory for Llama-2-7B and Llama-3-70B under ZeRO stages 0-3."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlbook.systems.memory_calc import GB, LLAMA2_7B, LLAMA3_70B, ParallelPlan, training_memory_per_gpu

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part14_memory_breakdown.png"


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    cases = [("Llama-2-7B, 8 GPUs (dp=8, tp=1)", LLAMA2_7B, 8, 1, 4096),
             ("Llama-3-70B, 64 GPUs (dp=8, tp=8)", LLAMA3_70B, 8, 8, 8192)]
    for ax, (title, cfg, dp, tp, s) in zip(axes, cases):
        stages = list(range(4))
        parts = {"weights": [], "grads": [], "optimizer": [], "activations": []}
        for st in stages:
            m = training_memory_per_gpu(cfg, ParallelPlan(dp=dp, tp=tp, zero_stage=st), s, 1,
                                        recompute="selective", sequence_parallel=True)
            for k in parts:
                parts[k].append(getattr(m, k) / GB)
        bottom = [0.0] * 4
        for i, (k, vals) in enumerate(parts.items()):
            ax.bar(stages, vals, bottom=bottom, label=k, color=f"C{i}")
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.axhline(80, color="k", ls="--", lw=1)
        ax.text(3.45, 82, "80 GB HBM", ha="right", fontsize=9)
        ax.set_xticks(stages)
        ax.set_xticklabels(["DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3 / FSDP"])
        ax.set_ylabel("GB per GPU")
        ax.set_title(title, fontsize=10)
        for x, tot in zip(stages, bottom):
            ax.text(x, tot + 2, f"{tot:.0f}", ha="center", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("16 bytes/param of training state, sharded by ZeRO stage; activations with selective recompute (micro-batch 1)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
