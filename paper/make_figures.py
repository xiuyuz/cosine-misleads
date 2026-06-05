"""Generate paper figures (primary-5 LVR family).

Figures produced:
    fig1_teaser.pdf                       single-panel cosine-vs-V*B scatter
    fig_faithfulness.pdf                  grouped corruption Δ bars
    fig_gap_vs_corruption_2panel.pdf      G vs σ=0.1 / G vs trunc scatter

Usage:
    python paper/make_figures.py \
        --probe-results $WORKSPACE/interpretability_results/probes_20260514/probe_results_ce.json \
        --faith-dir     $WORKSPACE/interpretability_results/faith_bs1 \
        --out-dir       paper/figures

The script reads on-disk artifacts when present; for the gap-vs-corruption
figure it uses the table-anchored values baked into this module so the figure
can be regenerated without running the entire faithfulness pipeline.
"""
import argparse
import glob
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3":        "#8BCF8B",
    "red_2":          "#E9A6A1",
    "red_strong":     "#B64342",
    "neutral":        "#CFCECE",
    "teal":           "#42949E",
}


def apply_publication_style(font_size=11, axes_linewidth=1.4):
    mpl.rcParams.update({
        "font.family": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 3,
        "legend.frameon": False,
        "axes.linewidth": axes_linewidth,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": axes_linewidth * 0.6,
        "ytick.major.width": axes_linewidth * 0.6,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
    })


# ---------------------------------------------------------------------------
# Variant metadata (primary 5 only).
# ---------------------------------------------------------------------------
VARIANT_ORDER = ["lvr_baseline", "nlvr", "dlvr_a", "plvr2", "plvr3"]
VARIANT_LABELS = {
    "lvr_baseline": "LVR",
    "nlvr":         "N-LVR",
    "dlvr_a":       "D-LVR",
    "plvr2":        "P-LVR-2",
    "plvr3":        "P-LVR-3",
}
VARIANT_COLORS = {
    "lvr_baseline": PALETTE["blue_secondary"],
    "nlvr":         PALETTE["green_3"],
    "dlvr_a":       PALETTE["teal"],
    "plvr2":        PALETTE["red_strong"],
    "plvr3":        PALETTE["red_2"],
}
VARIANT_MARKERS = {
    "lvr_baseline": "o",
    "nlvr":         "s",
    "dlvr_a":       "D",
    "plvr2":        "^",
    "plvr3":        "v",
}

# Table 1 anchor values (V*Bench accuracy + teacher-target cosine).
V_BENCH = {
    "lvr_baseline": 70.2, "nlvr": 71.7, "dlvr_a": 69.6,
    "plvr2": 57.1, "plvr3": 57.1,
}
COSINE = {
    "lvr_baseline": 0.555, "nlvr": 0.556, "dlvr_a": 0.464,
    "plvr2": 0.777, "plvr3": 0.769,
}
# Probe gap G = Acc(a) - Acc(b), in percentage points.
GAP_G = {
    "lvr_baseline": 36.6, "nlvr": 24.1, "dlvr_a": 31.9,
    "plvr2": 14.6, "plvr3": 14.1,
}
# V*Bench Δ accuracy under each corruption (pp).
DELTA_NOISE_01 = {
    "lvr_baseline": -1.6, "nlvr": -0.5, "dlvr_a": 0.0,
    "plvr2": 2.1, "plvr3": 2.6,
}
DELTA_TRUNC = {
    "lvr_baseline": -2.6, "nlvr": -2.6, "dlvr_a": -1.1,
    "plvr2": 0.0, "plvr3": 2.1,
}


# ---------------------------------------------------------------------------
# Artifact loaders
# ---------------------------------------------------------------------------
def load_faith_results(faith_dir):
    """Map (variant, benchmark) -> per-corruption result dict."""
    by_var_bench = {}
    for p in sorted(glob.glob(os.path.join(faith_dir, "faith_*.json"))):
        with open(p) as f:
            d = json.load(f)
        by_var_bench[(d["variant"], d["benchmark"])] = d["results"]
    return by_var_bench


# ---------------------------------------------------------------------------
# Figure 1: cosine vs V*Bench (single panel)
# ---------------------------------------------------------------------------
def fig1_teaser(out_path):
    apply_publication_style(font_size=11, axes_linewidth=1.4)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    xs = np.array([COSINE[v] for v in VARIANT_ORDER])
    ys = np.array([V_BENCH[v] for v in VARIANT_ORDER])
    z = np.polyfit(xs, ys, 1)
    r = np.corrcoef(xs, ys)[0, 1]
    x_line = np.linspace(xs.min() - 0.03, xs.max() + 0.03, 50)
    ax.plot(x_line, z[0] * x_line + z[1], "--",
            color=PALETTE["neutral"], linewidth=1.5, zorder=1)
    nudge = {"lvr_baseline": +0.004, "nlvr": -0.004,
             "plvr2": +0.004, "plvr3": -0.004}
    for v in VARIANT_ORDER:
        dx = nudge.get(v, 0.0)
        ax.scatter(COSINE[v] + dx, V_BENCH[v], s=120, c=VARIANT_COLORS[v],
                   marker=VARIANT_MARKERS[v], edgecolors="black",
                   linewidths=1.1, label=VARIANT_LABELS[v], zorder=10)
    ax.set_xlabel(r"Cosine alignment to teacher target $\mathbf{v}_t$",
                  fontsize=10.5, labelpad=3)
    ax.set_ylabel(r"V*Bench accuracy (%)", fontsize=10.5, labelpad=3)
    ax.set_title(f"Cosine misleads:  $r{{=}}{r:+.2f}$ across variants",
                 fontsize=11, pad=5, fontweight="bold")
    ax.set_xlim(xs.min() - 0.04, xs.max() + 0.04)
    ax.set_ylim(54, 76)
    ax.tick_params(labelsize=9.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=len(VARIANT_ORDER), fontsize=8.0,
              columnspacing=0.7, handlelength=1.0, handletextpad=0.3,
              borderpad=0.3, frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure: faithfulness corruption bars
# ---------------------------------------------------------------------------
def fig_faithfulness(faith, out_path):
    apply_publication_style(font_size=13, axes_linewidth=1.6)
    corruptions = ["truncate", "noise_0.1", "noise_0.3", "noise_1.0", "swap"]
    corr_labels = ["trunc.", r"$\sigma{=}.1$", r"$\sigma{=}.3$",
                   r"$\sigma{=}1$", "swap"]
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    n_var = len(VARIANT_ORDER)
    width = 0.15
    x = np.arange(len(corruptions))
    all_deltas = []
    for i, v in enumerate(VARIANT_ORDER):
        r = faith.get((v, "vstar"))
        if r is None or "clean" not in r:
            continue
        clean = r["clean"]["accuracy"]
        deltas = [(r[c]["accuracy"] - clean) * 100 if c in r else 0
                  for c in corruptions]
        all_deltas.extend(deltas)
        ax.bar(x + (i - n_var / 2 + 0.5) * width, deltas, width,
               label=VARIANT_LABELS[v], color=VARIANT_COLORS[v],
               edgecolor="black", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(corr_labels, fontsize=11)
    ax.set_xlabel("Corruption applied to LVR latent", fontsize=12,
                  fontweight="medium", labelpad=4)
    ax.set_ylabel(r"$\Delta$ V*Bench accuracy (pp)", fontsize=12,
                  fontweight="medium", labelpad=4)
    ax.tick_params(axis="y", labelsize=11)
    if all_deltas:
        lo, hi = min(all_deltas), max(all_deltas)
        ax.set_ylim(lo - 1.2, hi + 1.0)
    else:
        ax.set_ylim(-5, 6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=n_var, fontsize=8.5,
              columnspacing=0.8, handlelength=1.1, handletextpad=0.3,
              borderpad=0.3, frameon=False)
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure: probe gap G vs corruption sensitivity
# ---------------------------------------------------------------------------
LABEL_OFFSETS_NOISE = {
    "lvr_baseline": ( 0.9,  0.00, "left",  "center"),
    "nlvr":         ( 0.0,  0.45, "center", "bottom"),
    "dlvr_a":       ( 0.9,  0.30, "left",  "bottom"),
    "plvr2":        ( 0.9, -0.30, "left",  "top"),
    "plvr3":        ( 0.9,  0.30, "left",  "bottom"),
}
LABEL_OFFSETS_TRUNC = {
    "lvr_baseline": ( 0.9, -0.05, "left",  "center"),
    "nlvr":         ( 0.0,  0.40, "center", "bottom"),
    "dlvr_a":       ( 0.9, -0.05, "left",  "center"),
    "plvr2":        ( 0.9, -0.30, "left",  "top"),
    "plvr3":        ( 0.9,  0.05, "left",  "center"),
}


def _gap_panel(ax, delta_map, label_offsets, ylabel, mode_title):
    xs = np.array([GAP_G[v] for v in VARIANT_ORDER])
    ys = np.array([delta_map[v] for v in VARIANT_ORDER])
    z = np.polyfit(xs, ys, 1)
    r = float(np.corrcoef(xs, ys)[0, 1])
    x_line = np.linspace(xs.min() - 2.5, xs.max() + 2.5, 50)
    ax.plot(x_line, z[0] * x_line + z[1], "--",
            color=PALETTE["neutral"], linewidth=1.4, zorder=2)
    ax.axhline(0, color="black", linewidth=0.9, alpha=0.55, zorder=1)
    for v in VARIANT_ORDER:
        ax.scatter(GAP_G[v], delta_map[v], s=130, c=VARIANT_COLORS[v],
                   marker=VARIANT_MARKERS[v], edgecolors="black",
                   linewidths=1.1, zorder=10)
        dx, dy, ha, va = label_offsets[v]
        ax.annotate(VARIANT_LABELS[v],
                    xy=(GAP_G[v], delta_map[v]),
                    xytext=(GAP_G[v] + dx, delta_map[v] + dy),
                    fontsize=10, ha=ha, va=va, zorder=11)
    ax.text(0.97, 0.97, f"$r = {r:+.2f}$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="medium",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=PALETTE["neutral"], linewidth=0.8))
    ax.set_xlabel(r"Decodability gap $G$ (pp)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(mode_title, fontsize=11, pad=4)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlim(xs.min() - 3, xs.max() + 7)
    pad_y = max(0.9, (ys.max() - ys.min()) * 0.22)
    ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)


def fig_gap_vs_corruption(out_path, two_panel=True):
    apply_publication_style(font_size=11, axes_linewidth=1.4)
    if two_panel:
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
        _gap_panel(axes[0], DELTA_NOISE_01, LABEL_OFFSETS_NOISE,
                   ylabel=r"$\Delta$ V*Bench accuracy (pp)",
                   mode_title=r"$\sigma{=}0.1$ noise corruption")
        _gap_panel(axes[1], DELTA_TRUNC, LABEL_OFFSETS_TRUNC,
                   ylabel=r"$\Delta$ V*Bench accuracy (pp)",
                   mode_title="truncation corruption")
    else:
        fig, ax = plt.subplots(figsize=(3.8, 3.2))
        _gap_panel(ax, DELTA_NOISE_01, LABEL_OFFSETS_NOISE,
                   ylabel=r"$\Delta$ V*Bench accuracy (pp)",
                   mode_title=r"$\sigma{=}0.1$ noise corruption")
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faith-dir", required=True,
                    help="directory containing faith_*.json files for the five variants")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    faith = load_faith_results(args.faith_dir)
    fig1_teaser(os.path.join(args.out_dir, "fig1_teaser.pdf"))
    fig_faithfulness(faith, os.path.join(args.out_dir, "fig_faithfulness.pdf"))
    fig_gap_vs_corruption(
        os.path.join(args.out_dir, "fig_gap_vs_corruption_2panel.pdf"),
        two_panel=True,
    )


if __name__ == "__main__":
    main()
