#!/usr/bin/env python
"""Draw LightStereo-Instance + PG-GUIC architecture v4 with data-flow symbols."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

OUT = Path("architecture_figs/lightstereo_openworld_architecture_v4.jpg")


def box(ax, xy, w, h, text, fc, ec, fontsize=8.5, lw=1.4, radius=0.02):
    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1a1a1a",
        zorder=3,
        linespacing=1.25,
    )
    return (x, y, w, h)


def arrow(ax, p1, p2, color="#333333", style="-|>", lw=1.35, connectionstyle="arc3,rad=0"):
    a = FancyArrowPatch(
        p1,
        p2,
        arrowstyle=style,
        mutation_scale=11,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        zorder=4,
    )
    ax.add_patch(a)


def dashed_arrow(ax, p1, p2, color="#c45c26", lw=1.2):
    a = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=lw,
        color=color,
        linestyle="--",
        connectionstyle="arc3,rad=0",
        zorder=4,
    )
    ax.add_patch(a)


def mid_bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


def mid_top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def mid_left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def mid_right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def main():
    fig, ax = plt.subplots(figsize=(16.5, 10.2), dpi=180)
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 10.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Title
    ax.text(
        8.25,
        9.85,
        "LightStereo-Instance + PG-GUIC (core)",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#1f2a37",
    )
    ax.text(
        8.25,
        9.48,
        "Data flow with symbols  |  Blue = trainable known model  |  Orange = frozen / rule (no train on KITTI leave-out)",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#555555",
    )

    # Colors
    c_in = "#f7f7f7"
    e_in = "#555555"
    c_blue = "#e8f1fb"
    e_blue = "#2f6fad"
    c_blue_d = "#d5e6f7"
    c_orange = "#fff0e6"
    e_orange = "#c45c26"
    c_orange_d = "#ffe3d1"
    c_out = "#eef6ee"
    e_out = "#2f7d4a"

    # ========== Inputs (parallel, no arrows between) ==========
    ax.text(1.55, 8.95, "Inputs (parallel)", fontsize=9, fontweight="bold", color="#333", ha="center")
    b_il = box(
        ax,
        (0.35, 8.15),
        2.4,
        0.65,
        r"$I_L$  Left RGB" + "\n" + r"(KITTI2015)",
        c_in,
        e_in,
        8.2,
    )
    b_ir = box(
        ax,
        (0.35, 7.25),
        2.4,
        0.65,
        r"$I_R$  Right RGB" + "\n" + r"(KITTI2015)",
        c_in,
        e_in,
        8.2,
    )
    b_p = box(
        ax,
        (0.35, 6.35),
        2.4,
        0.65,
        r"$P=\{p_1,\ldots,p_m\}$" + "\n" + r"Text prompts (left-out)",
        c_in,
        e_in,
        8.0,
    )

    # ========== Known model group ==========
    known_frame = FancyBboxPatch(
        (3.05, 2.55),
        6.55,
        6.55,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor="#f8fbfe",
        edgecolor=e_blue,
        linewidth=2.0,
        zorder=1,
    )
    ax.add_patch(known_frame)
    ax.text(
        6.325,
        8.9,
        "LightStereo-Instance Known Model  (trainable, $n_c=K$)",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=e_blue,
        zorder=3,
    )

    b_bb = box(
        ax,
        (4.55, 8.05),
        3.5,
        0.6,
        r"Shared Backbone (MobileNetV2–FPN)" + "\n" + r"in: $I_L,I_R$  →  out: $F_L,F_R$",
        c_blue_d,
        e_blue,
        7.8,
    )

    # Left path (disparity)
    ax.text(4.55, 7.7, "Stereo / Disparity path", fontsize=7.5, color=e_blue, fontweight="bold")
    b_cv = box(
        ax,
        (3.25, 6.85),
        2.9,
        0.58,
        r"Cost Volume" + "\n" + r"in: $F_L^{(0)},F_R^{(0)}$ → out: $C$",
        c_blue,
        e_blue,
        7.3,
    )
    b_agg = box(
        ax,
        (3.25, 6.05),
        2.9,
        0.58,
        r"Aggregation + LCA" + "\n" + r"in: $C$ + $F_L$ (both) → out: $V$",
        c_blue,
        e_blue,
        7.3,
    )
    b_reg = box(
        ax,
        (3.25, 5.25),
        2.9,
        0.58,
        r"Disp Regression" + "\n" + r"in: $V$ → out: $d_{\mathrm{init}}$",
        c_blue,
        e_blue,
        7.3,
    )
    b_ref = box(
        ax,
        (3.25, 4.45),
        2.9,
        0.58,
        r"Refinement" + "\n" + r"in: $d_{\mathrm{init}},F_L,I_L$ → $\hat{D}$",
        c_blue,
        e_blue,
        7.3,
    )
    b_d = box(
        ax,
        (3.25, 3.45),
        2.9,
        0.7,
        r"Disparity Map  $\hat{D}$" + "\n" + r"$H\!\times\!W$ (dense)",
        "#dceaf8",
        e_blue,
        8.2,
        lw=1.8,
    )

    # Right path (instance)
    ax.text(7.85, 7.7, "Known instance path", fontsize=7.5, color=e_blue, fontweight="bold")
    b_proj = box(
        ax,
        (6.45, 6.85),
        2.9,
        0.58,
        r"Projection Adapters" + "\n" + r"in: $F_L$ → out: $P_3,P_4,P_5$",
        c_blue,
        e_blue,
        7.3,
    )
    b_yolo = box(
        ax,
        (6.45, 6.05),
        2.9,
        0.58,
        r"YOLO Seg Neck+Head ($n_c\!=\!K$)" + "\n" + r"in: $P_{3\!-\!5}$ → out: $S$",
        c_blue,
        e_blue,
        7.2,
    )
    b_ea = box(
        ax,
        (6.45, 5.05),
        2.9,
        0.72,
        r"EdgeAware SSEG" + "\n" + r"in: $S,\,\hat{D}$ (aux)" + "\n" + r"out: refined prototypes",
        c_blue,
        e_blue,
        7.2,
    )
    b_k = box(
        ax,
        (6.45, 3.45),
        2.9,
        0.7,
        r"Known Instances  $\mathcal{K}$" + "\n" + r"$\{(b,M,s,c)\}_{c\in\mathcal{C}_K}$",
        "#dceaf8",
        e_blue,
        8.0,
        lw=1.8,
    )

    # ========== Open-world ==========
    ow_frame = FancyBboxPatch(
        (10.0, 2.55),
        6.1,
        6.55,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor="#fffaf6",
        edgecolor=e_orange,
        linewidth=2.0,
        zorder=1,
    )
    ax.add_patch(ow_frame)
    ax.text(
        13.05,
        8.9,
        "Open-World core (not optional)  —  frozen / rule",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=e_orange,
        zorder=3,
    )

    b_yoloe = box(
        ax,
        (10.35, 7.55),
        5.4,
        0.85,
        r"Frozen YOLOE-26x-seg" + "\n" + r"in: $I_L,\,P$   (no $I_R$)" + "\n" + r"out: $\mathcal{U}_{\mathrm{raw}}$  unknown candidates",
        c_orange_d,
        e_orange,
        8.0,
    )
    b_g0 = box(
        ax,
        (10.35, 6.15),
        5.4,
        0.95,
        r"$G_0$ Geometry Refine  (rule, no params)" + "\n" + r"in: $\mathcal{U}_{\mathrm{raw}},\,\hat{D}$" + "\n" + r"out: $\mathcal{U}$  (edge-aligned masks + geom. score)",
        c_orange,
        e_orange,
        7.8,
    )
    b_cf = box(
        ax,
        (10.35, 4.75),
        5.4,
        0.95,
        r"Conflict Fusion  (rule, no params)" + "\n" + r"in: $\mathcal{K},\,\mathcal{U}$" + "\n" + r"out: $\mathcal{O}$  Known + $\mathrm{unknown}_j$",
        c_orange,
        e_orange,
        7.8,
    )
    b_o = box(
        ax,
        (10.35, 3.45),
        5.4,
        0.7,
        r"Fused Output  $\mathcal{O}=\{(b_i,M_i,s_i,c_i,\mathrm{id}_i)\}$" + "\n" + r"$c_i\in\mathcal{C}_K\cup\{\mathrm{unknown}_j\}$",
        "#ffe8d6",
        e_orange,
        8.0,
        lw=1.8,
    )

    # ========== Point cloud ==========
    b_pc = box(
        ax,
        (5.5, 0.55),
        7.5,
        1.15,
        r"Instance Point Cloud  $\{Q_i\}$" + "\n"
        + r"in: $\hat{D}$ (depth),  $\mathcal{O}$ (masks),  $I_L$ (RGB color)" + "\n"
        + r"out: per-instance 3D points colored by instance / class",
        c_out,
        e_out,
        8.3,
        lw=1.8,
    )

    # ========== Arrows: inputs ==========
    # I_L, I_R -> backbone (no arrows between inputs)
    arrow(ax, mid_right(b_il), (4.55, 8.35), color=e_blue)
    arrow(ax, mid_right(b_ir), (4.55, 8.2), color=e_blue)
    # I_L + P -> YOLOE
    arrow(ax, (2.75, 8.45), (10.35, 8.15), color=e_orange, connectionstyle="arc3,rad=-0.12")
    arrow(ax, mid_right(b_p), (10.35, 7.85), color=e_orange, connectionstyle="arc3,rad=0.05")
    ax.text(6.5, 8.55, r"$I_L,\,P$", fontsize=7.2, color=e_orange, ha="center")

    # Backbone -> two paths
    arrow(ax, (5.5, 8.05), mid_top(b_cv), color=e_blue)
    arrow(ax, (7.5, 8.05), mid_top(b_proj), color=e_blue)
    # F_L also feeds Aggregation+LCA (with C) and Refinement
    dashed_arrow(ax, (4.7, 8.05), (3.25, 6.35), color=e_blue)
    ax.text(3.95, 7.15, r"$F_L$", fontsize=7, color=e_blue, ha="center")
    dashed_arrow(ax, (4.55, 8.05), (3.25, 4.75), color=e_blue)
    ax.text(3.75, 5.55, r"$F_L$", fontsize=6.8, color=e_blue, ha="center")

    # Disparity chain: C is main vertical input into Aggregation
    arrow(ax, mid_bottom(b_cv), mid_top(b_agg), color=e_blue)
    ax.text(4.85, 6.55, r"$C$", fontsize=7, color=e_blue, ha="left")
    arrow(ax, mid_bottom(b_agg), mid_top(b_reg), color=e_blue)
    arrow(ax, mid_bottom(b_reg), mid_top(b_ref), color=e_blue)
    arrow(ax, mid_bottom(b_ref), mid_top(b_d), color=e_blue)

    # Instance chain
    arrow(ax, mid_bottom(b_proj), mid_top(b_yolo), color=e_blue)
    arrow(ax, mid_bottom(b_yolo), mid_top(b_ea), color=e_blue)
    arrow(ax, mid_bottom(b_ea), mid_top(b_k), color=e_blue)

    # D-hat -> EdgeAware (aux)  IMPORTANT: not D <-> K
    dashed_arrow(ax, mid_right(b_d), mid_left(b_ea), color=e_blue)
    ax.text(6.15, 4.55, r"$\hat{D}$ aux", fontsize=7, color=e_blue, ha="center")

    # Open-world chain
    arrow(ax, mid_bottom(b_yoloe), mid_top(b_g0), color=e_orange)
    arrow(ax, mid_bottom(b_g0), mid_top(b_cf), color=e_orange)
    arrow(ax, mid_bottom(b_cf), mid_top(b_o), color=e_orange)

    # D-hat -> G0
    dashed_arrow(ax, mid_right(b_d), mid_left(b_g0), color=e_orange)
    ax.text(9.55, 5.85, r"$\hat{D}$", fontsize=7.5, color=e_orange, ha="center")

    # K -> Conflict Fusion (one-way only; NO bidir with EdgeAware)
    arrow(ax, mid_right(b_k), mid_left(b_cf), color=e_orange)
    ax.text(9.55, 4.55, r"$\mathcal{K}$", fontsize=7.5, color=e_orange, ha="center")

    # To point cloud
    arrow(ax, mid_bottom(b_d), (6.5, 1.7), color=e_out, connectionstyle="arc3,rad=0.15")
    arrow(ax, mid_bottom(b_o), (10.5, 1.7), color=e_out, connectionstyle="arc3,rad=-0.12")
    ax.text(5.35, 2.35, r"$\hat{D}$", fontsize=7.5, color=e_out)
    ax.text(11.6, 2.35, r"$\mathcal{O}$", fontsize=7.5, color=e_out)

    # Legend / notes
    ax.text(
        0.35,
        2.15,
        "Notes",
        fontsize=8.5,
        fontweight="bold",
        color="#333",
    )
    notes = (
        r"• $I_L,I_R,P$ are parallel inputs (no arrows among them)." + "\n"
        r"• YOLOE uses only $I_L$ and $P$; $I_R$ is for stereo only." + "\n"
        r"• $\hat{D}\!\to\!$ EdgeAware SSEG (aux); no $\hat{D}\leftrightarrow\mathcal{K}$ peer link." + "\n"
        r"• Conflict Fusion $\leftarrow\mathcal{K},\mathcal{U}$ one-way; no link back to EdgeAware." + "\n"
        r"• $G_0$ / Fusion: rule modules (no KITTI leave-out training)."
    )
    ax.text(0.35, 0.35, notes, fontsize=7.2, color="#444", va="bottom", linespacing=1.35)

    # Legend boxes
    box(ax, (0.35, 5.45), 2.4, 0.55, "trainable (params)", c_blue, e_blue, 7.5)
    box(ax, (0.35, 4.7), 2.4, 0.55, "frozen / rule", c_orange, e_orange, 7.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=200, facecolor="white")
    # also png
    out_png = OUT.with_suffix(".png")
    fig.savefig(out_png, bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {OUT.resolve()}")
    print(f"saved {out_png.resolve()}")


if __name__ == "__main__":
    main()
