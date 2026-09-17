"""Speculative decoding: expected tokens per target call vs acceptance rate, with
empirical points from the toy bigram models at several draft temperatures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbook.systems.speculative_decoding import acceptance_rate, expected_tokens_per_call, generate, make_toy_models

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part14_speculative_acceptance.png"


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    alpha = np.linspace(0, 1, 200)
    for k in (1, 2, 4, 8):
        ax.plot(alpha, [expected_tokens_per_call(a, k) for a in alpha], label=f"K = {k} (theory)")
    rng = np.random.default_rng(0)
    for temp in (1.0, 1.3, 2.0, 4.0, 10.0):
        P, Q = make_toy_models(V=10, temperature_draft=temp, rng=rng)
        a = float(np.mean([acceptance_rate(P[i], Q[i]) for i in range(10)]))
        toks, calls = generate(P, Q, 0, 3000, k=4, rng=rng)
        ax.plot(a, len(toks) / calls, "ko", ms=6)
        ax.annotate(f"draft T={temp}", (a, len(toks) / calls), textcoords="offset points", xytext=(5, -12), fontsize=8)
    ax.set_xlabel("acceptance rate alpha = 1 - TV(p, q)")
    ax.set_ylabel("expected tokens per target forward pass")
    ax.set_title("Speculative decoding: (1 - alpha^(K+1)) / (1 - alpha); dots = simulated K=4", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
