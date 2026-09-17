"""Timeline of publicly announced architecture / paradigm shifts at Tesla and Waymo.
Every entry is a public event (talk, blog post, paper, dataset release); see the
Tesla and Waymo chapters for the sources."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part18_architecture_timeline.png"

TESLA = [
    (2019.3, "Autonomy Day: HW3 FSD computer,\ncamera-first stack"),
    (2021.4, "Tesla Vision: radar dropped\n(CVPR'21 WAD keynote)"),
    (2021.6, "AI Day 2021: multi-camera BEV\n'vector space', auto-labelling,\ntriggers, Dojo announced"),
    (2022.45, "CVPR'22 WAD: occupancy\nnetworks replace per-object 3D"),
    (2022.75, "AI Day 2022: occupancy + flow,\nlane 'language' decoder, Optimus,\nDojo (Hot Chips 34)"),
    (2024.2, "FSD V12: end-to-end network\n(Q4'23 shareholder letter)"),
]
WAYMO = [
    (2019.6, "Waymo Open Dataset\n(lidar + camera)"),
    (2020.8, "Rider-only service in Phoenix;\nsafety methodologies paper"),
    (2021.5, "SimulationCity;\nOpen Motion Dataset"),
    (2022.55, "Wayformer (attention-based\nmotion forecasting)"),
    (2023.75, "MotionLM (motion as tokens);\nWaymax simulator (NeurIPS)"),
    (2024.65, "6th-gen Driver;\nEMMA (Gemini-based end-to-end)"),
    (2025.45, "Scaling laws for motion\nforecasting & planning"),
    (2026.1, "Waymo World Model\n(built on Genie 3) for sim"),
]


def draw_lane(ax, events, y, color, above: bool) -> None:
    for i, (t, label) in enumerate(events):
        sign = 1 if (above ^ (i % 2 == 1)) else -1
        ax.plot([t, t], [y, y + sign * 0.35], color=color, lw=1)
        ax.plot(t, y, "o", color=color, ms=7, markeredgecolor="white", markeredgewidth=1.2)
        ax.text(t, y + sign * 0.4, label, ha="center", va="bottom" if sign > 0 else "top",
                fontsize=6.9, color="0.15")


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(2018.8, 2026.7)
    ax.set_ylim(-0.2, 3.3)
    ax.axis("off")

    for y, name, color in [(2.5, "Tesla", "C3"), (0.6, "Waymo", "C0")]:
        ax.plot([2018.9, 2026.6], [y, y], color=color, lw=2, alpha=0.5)
        ax.text(2018.85, y, name, ha="right", va="center", fontsize=11, color=color, fontweight="bold")
    draw_lane(ax, TESLA, 2.5, "C3", above=True)
    draw_lane(ax, WAYMO, 0.6, "C0", above=True)

    for yr in range(2019, 2027):
        ax.text(yr, -0.15, str(yr), ha="center", fontsize=8, color="0.4")
        ax.plot([yr, yr], [-0.05, 3.25], color="0.9", lw=0.6, zorder=0)

    ax.set_title("Publicly announced paradigm shifts: Tesla (vision-only → occupancy → end-to-end) vs "
                 "Waymo (lidar fusion → tokenised prediction → multimodal end-to-end research)", fontsize=10.5)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
