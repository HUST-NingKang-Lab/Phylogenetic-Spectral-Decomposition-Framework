#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised host-associated macro–micro spectral phase space
========================================================

Changes relative to the previous version
----------------------------------------
1) Remove error bars.
2) Use non-overlapping trajectory arrows:
   - Palleja: blue curved arrows
   - Infant: orange straight arrow
   - IBD: red straight arrow
3) Ensure all group-center colors are distinct.
4) Increase center-point separation by applying a monotonic contrast stretch
   after within-dataset robust scaling, so the phase-space geometry is easier
   to interpret visually while preserving the ordering of samples.
5) Lighten background points and keep dataset ellipses/subgroups visible.

Outputs
-------
- host_macro_micro_phase_space.png
- host_macro_micro_phase_space.pdf
"""

from __future__ import annotations

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D

# =============================================================================
# 0. Paths and parameters
# =============================================================================
PALLEJA_ABU_PATH = "/home/zhangyuli/傅里叶非时序/data/Palleja/abundance_phylogeny_ordered.csv"
PALLEJA_META_PATH = "/home/zhangyuli/傅里叶非时序/data/Palleja/meta.csv"

INFANT_ABU_PATH = "/home/zhangyuli/傅里叶非时序/data/infant/abundance.csv"
INFANT_META_PATH = "/home/zhangyuli/傅里叶非时序/data/infant/metadata.csv"

IBD_ABU_PATH = "/home/zhangyuli/傅里叶非时序/data/IBD/abundance.csv"
IBD_META_PATH = "/home/zhangyuli/傅里叶非时序/data/IBD/meta.csv"

PHYLO_PATH = "/home/zhangyuli/傅里叶非时序/data/phylogeny.csv"

OUT_DIR = Path("/home/zhangyuli/傅里叶非时序/figures/cross_dataset/host_macro_micro_phase_space")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PSEUDOCOUNT = 1e-9
FMAX = 0.20
LOW_FRAC_OF_MODES = 0.25
HIGH_FRAC_OF_MODES = 0.35
RANDOM_STATE = 42
SAVE_DPI = 600
MAX_RAW_POINTS_PER_GROUP = 300

# Dataset colors and shapes (for raw clouds / legend)
DATASET_COLORS = {
    "Palleja": "#86A7C8",   # muted blue for background clouds
    "Infant": "#B98A3E",    # muted ochre-brown for background clouds
    "IBD": "#C98A86",       # muted rose-red for background clouds
}
DATASET_MARKERS = {
    "Palleja": "o",
    "Infant": "^",
    "IBD": "D",
}

# Distinct center colors for each biological state/group.
GROUP_COLORS = {
    # Palleja: three related blues
    "Pre-stable": "#8BB8E8",     # light blue
    "Perturbation": "#4C78A8",   # medium blue
    "Recovery": "#1F5C99",       # deep blue
    # Infant: two related oranges
    "Immature": "#E3A137",       # light orange
    "Mature": "#B97A18",         # deep orange
    # IBD: two related reds
    "Healthy": "#E79A92",        # light red
    "IBD": "#C9453B",            # deep red
}

REGION_COLORS = {
    "unstable": "#D8D1C9",   # neutral warm gray, not yellow
    "stable": "#CCD5CF",     # neutral sage-gray, not blue
}

# Arrow-specific colors as requested.
ARROW_COLORS = {
    "Palleja": "#3972B3",  # blue curved
    "Infant": "#D99128",   # orange straight
    "IBD": "#C9453B",      # red straight
}

TRAJECTORIES = {
    "Palleja": ["Pre-stable", "Perturbation", "Recovery"],
    "Infant": ["Immature", "Mature"],
    "IBD": ["Healthy", "IBD"],
}

# Fine-grained label offsets to reduce overlaps.
LABEL_OFFSETS = {
    "Pre-stable": (0.010, 0.018, "left"),
    "Perturbation": (0.012, 0.020, "left"),
    "Recovery": (0.010, -0.022, "left"),
    "Immature": (0.010, 0.014, "left"),
    "Mature": (0.010, 0.018, "left"),
    "Healthy": (0.010, -0.020, "left"),
    "IBD": (0.010, 0.020, "left"),
}


# =============================================================================
# 1. Plotting helpers
# =============================================================================
def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.6,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": SAVE_DPI,
        "figure.dpi": 160,
    })


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2F2330")
    ax.spines["bottom"].set_color("#2F2330")
    ax.tick_params(colors="#2F2330")
    ax.grid(True, color="#E8E2D8", lw=0.7, alpha=0.85)
    ax.set_axisbelow(True)


def robust_scale_01(x: pd.Series, qlo=0.02, qhi=0.98) -> pd.Series:
    x = pd.Series(x, dtype=float)
    lo, hi = x.quantile(qlo), x.quantile(qhi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = x.min(), x.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.full(len(x), 0.5), index=x.index)
    return ((x - lo) / (hi - lo)).clip(0, 1)


def contrast_spread_01(x: pd.Series | np.ndarray, k: float = 6.0) -> pd.Series:
    """Monotonic contrast stretch around 0.5 to push central values outward.
    Preserves rank order while making centers less clustered in the middle.
    """
    x = pd.Series(x, dtype=float)
    x = x.clip(0, 1)
    z = 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
    z0 = 1.0 / (1.0 + np.exp(-k * (0.0 - 0.5)))
    z1 = 1.0 / (1.0 + np.exp(-k * (1.0 - 0.5)))
    out = (z - z0) / (z1 - z0)
    return out.clip(0, 1)


def confidence_ellipse(ax, x, y, color, alpha=0.10, lw=1.0, n_std=1.20, zorder=1):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.maximum(vals, 1e-12)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(
        xy=(np.nanmedian(x), np.nanmedian(y)),
        width=width, height=height, angle=theta,
        facecolor=color, edgecolor=color, lw=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(ell)


def label_point(ax, x, y, text, color, dx=0.012, dy=0.012, ha="left"):
    ax.text(
        x + dx, y + dy, text, fontsize=7.2, color=color,
        ha=ha, va="center", zorder=35,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.78),
    )


def draw_arrow(ax, start, end, color, lw=2.0, rad=0.0, shrinkA=9, shrinkB=9, zorder=26):
    patch = FancyArrowPatch(
        posA=start, posB=end,
        arrowstyle="-|>",
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=11,
        lw=lw,
        color=color,
        alpha=0.92,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
        zorder=zorder,
    )
    ax.add_patch(patch)


# =============================================================================
# 2. Taxonomy helpers
# =============================================================================
def norm_taxon(x: str) -> str:
    x = str(x).strip()
    if x.startswith("sk__"):
        x = "k__" + x[4:]
    x = x.replace("|", ";")
    return x


def to_genus_taxon(x: str) -> str:
    x = norm_taxon(x)
    parts = [p for p in x.split(";") if p]
    out = []
    for p in parts:
        if p.startswith("s__"):
            break
        out.append(p)
        if p.startswith("g__"):
            break
    return ";".join(out)


def load_phylogeny_genus_order() -> list[str]:
    phylo = pd.read_csv(PHYLO_PATH, low_memory=False)
    phy_col = phylo.columns[0]
    order = (
        phylo[phy_col]
        .astype(str)
        .map(norm_taxon)
        .map(to_genus_taxon)
        .drop_duplicates()
        .tolist()
    )
    return order


# =============================================================================
# 3. Dataset loaders
# =============================================================================
def load_palleja() -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_col = "sample_id"
    group_col = "label"
    group_map = {
        "pre_stable": "Pre-stable",
        "perturbation_unstable": "Perturbation",
        "recovery_stable": "Recovery",
    }
    group_order = list(group_map.keys())

    abu = pd.read_csv(PALLEJA_ABU_PATH, low_memory=False)
    meta = pd.read_csv(PALLEJA_META_PATH, low_memory=False)
    abu[sample_col] = abu[sample_col].astype(str)
    meta[sample_col] = meta[sample_col].astype(str)
    abu = abu.set_index(sample_col)
    meta = meta.set_index(sample_col)
    meta = meta[meta[group_col].isin(group_order)].copy()
    shared = abu.index.intersection(meta.index)
    abu = abu.loc[shared].copy()
    meta = meta.loc[shared].copy()
    abu = abu.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    meta["dataset"] = "Palleja"
    meta["group_plot"] = meta[group_col].map(group_map)
    return abu, meta[["dataset", "group_plot"]]


def load_infant() -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_col = "sample_id"
    group_col = "group"
    group_map = {"immature": "Immature", "mature": "Mature"}
    genus_ordered_list = pd.read_csv(PHYLO_PATH).iloc[:, 0].astype(str).str.strip().tolist()

    abu = pd.read_csv(INFANT_ABU_PATH, low_memory=False)
    meta = pd.read_csv(INFANT_META_PATH, low_memory=False)
    abu[sample_col] = abu[sample_col].astype(str)
    meta[sample_col] = meta[sample_col].astype(str)
    abu = abu.set_index(sample_col)
    meta = meta.set_index(sample_col)

    mapped_cols = []
    for c in abu.columns:
        parts = str(c).split("|")
        genus_parts = []
        for p in parts:
            if p.startswith("s__") or p.startswith("t__"):
                break
            if p.startswith("k__"):
                genus_parts.append("s" + p)
            else:
                genus_parts.append(p)
        genus_path = ";".join(genus_parts)
        if genus_path in genus_ordered_list:
            mapped_cols.append(genus_path)
        else:
            mapped_cols.append("UNMATCHED")
    abu.columns = mapped_cols
    if "UNMATCHED" in abu.columns:
        abu = abu.drop(columns=["UNMATCHED"])
    abu = abu.T.groupby(level=0).sum().T
    ordered_cols = [c for c in genus_ordered_list if c in abu.columns]
    abu = abu[ordered_cols]

    meta = meta[meta[group_col].isin(group_map.keys())].copy()
    shared = abu.index.intersection(meta.index)
    abu = abu.loc[shared].copy()
    meta = meta.loc[shared].copy()
    abu = abu.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    meta["dataset"] = "Infant"
    meta["group_plot"] = meta[group_col].map(group_map)
    return abu, meta[["dataset", "group_plot"]]


def load_ibd() -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_col = "sample_id"
    group_col = "label"
    group_map = {"healthy": "Healthy", "disease": "IBD"}
    phy = load_phylogeny_genus_order()

    abu = pd.read_csv(IBD_ABU_PATH, low_memory=False)
    meta = pd.read_csv(IBD_META_PATH, low_memory=False)
    abu[sample_col] = abu[sample_col].astype(str)
    meta[sample_col] = meta[sample_col].astype(str)
    abu = abu.set_index(sample_col)
    meta = meta.set_index(sample_col)

    abu.columns = [to_genus_taxon(c) for c in abu.columns]
    abu = abu.T.groupby(level=0).sum().T
    inter = abu.columns.intersection(phy)
    if len(inter) == 0:
        raise ValueError("No overlapping IBD taxa with genus-level phylogeny")
    abu = abu.loc[:, inter]
    ordered_cols = [c for c in phy if c in abu.columns]
    abu = abu[ordered_cols]

    meta = meta[meta[group_col].isin(group_map.keys())].copy()
    shared = abu.index.intersection(meta.index)
    abu = abu.loc[shared].copy()
    meta = meta.loc[shared].copy()
    abu = abu.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    meta["dataset"] = "IBD"
    meta["group_plot"] = meta[group_col].map(group_map)
    return abu, meta[["dataset", "group_plot"]]


# =============================================================================
# 4. Spectral metrics
# =============================================================================
def compute_metrics_for_dataset(abu: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    abu = abu.loc[:, abu.sum(axis=0) > 0].copy()
    row_sums = abu.sum(axis=1)
    abu = abu.loc[row_sums > 0].div(row_sums[row_sums > 0], axis=0)
    meta = meta.loc[abu.index].copy()

    n = abu.shape[1]
    if n < 8:
        raise ValueError(f"Too few taxa for {meta['dataset'].iloc[0]} after filtering: n={n}")
    rel = abu + PSEUDOCOUNT
    log_rel = np.log(rel)
    clr = log_rel.sub(log_rel.mean(axis=1), axis=0)

    window = np.hanning(n).astype(float)
    freq = np.fft.rfftfreq(n, d=1.0)
    fmin = 2.0 / n
    mask = (freq >= fmin) & (freq <= FMAX)
    modes = np.where(mask)[0]
    if len(modes) < 6:
        raise ValueError(f"Too few frequency modes for {meta['dataset'].iloc[0]}: {len(modes)}")

    n_low = max(2, int(np.ceil(len(modes) * LOW_FRAC_OF_MODES)))
    n_high = max(2, int(np.ceil(len(modes) * HIGH_FRAC_OF_MODES)))
    low_modes = modes[:n_low]
    high_modes = modes[-n_high:]
    mid_modes = np.array([m for m in modes if (m not in set(low_modes)) and (m not in set(high_modes))])

    records = []
    for sid in clr.index:
        xw = clr.loc[sid].values.astype(float) * window
        psd = (np.abs(np.fft.rfft(xw)) ** 2) / n
        low = float(np.nansum(psd[low_modes]))
        mid = float(np.nansum(psd[mid_modes])) if len(mid_modes) else 0.0
        high = float(np.nansum(psd[high_modes]))
        records.append({
            "sample": sid,
            "dataset": meta.loc[sid, "dataset"],
            "group": meta.loc[sid, "group_plot"],
            "low_power": low,
            "mid_power": mid,
            "high_power": high,
            "low_dominance_raw": np.log10((low + 1e-30) / (mid + high + 1e-30)),
            "high_fragment_raw": np.log10((high + 1e-30) / (low + 1e-30)),
        })
    return pd.DataFrame(records).set_index("sample")


def load_all_metrics() -> pd.DataFrame:
    parts = []
    for name, loader in [("Palleja", load_palleja), ("Infant", load_infant), ("IBD", load_ibd)]:
        print(f"[load] {name}")
        abu, meta = loader()
        d = compute_metrics_for_dataset(abu, meta)
        # Within-dataset robust scaling for platform/taxon-count differences.
        d["x_low_macro"] = robust_scale_01(d["low_dominance_raw"])
        d["y_high_micro"] = robust_scale_01(d["high_fragment_raw"])
        print(f"[data] {name}: samples={d.shape[0]}, groups={d['group'].value_counts().to_dict()}")
        parts.append(d)
    df = pd.concat(parts, axis=0)

    # Additional global contrast stretch to reduce center crowding while preserving order.
    df["x_low_macro"] = contrast_spread_01(df["x_low_macro"], k=7.0)
    df["y_high_micro"] = contrast_spread_01(df["y_high_micro"], k=6.0)
    return df


# =============================================================================
# 5. Figure
# =============================================================================
def plot_host_phase_space(df: pd.DataFrame) -> None:
    set_style()
    rng = np.random.default_rng(RANDOM_STATE)

    fig, ax = plt.subplots(figsize=(6.25, 4.95))
    ax.set_facecolor("#FBF8F3")

    # Conceptual background regions.
    # Keep them as subtle corner cues rather than large blocks, so empty corners do not dominate.
    ax.fill_between([0.0, 0.38], [0.66, 0.66], [1.06, 1.06],
                    color="#F2B37F", alpha=0.115, lw=0, zorder=0)
    ax.fill_between([0.58, 1.08], [-0.04, -0.04], [0.34, 0.34],
                    color="#82B6D9", alpha=0.12, lw=0, zorder=0)
    ax.text(0.050, 0.985, "micro-fragmented\n/ unstable", color="#B25C36", fontsize=7.4,
            ha="left", va="top", alpha=0.88)
    ax.text(0.945, 0.075, "macro-structured\nstability basin", color="#3576A7", fontsize=7.4,
            ha="right", va="bottom", alpha=0.92)

    # Dataset clouds and sample points.
    for dataset in ["Palleja", "Infant", "IBD"]:
        dsub = df[df["dataset"].eq(dataset)]
        dcolor = DATASET_COLORS[dataset]
        marker = DATASET_MARKERS[dataset]
        confidence_ellipse(
            ax, dsub["x_low_macro"], dsub["y_high_micro"],
            color=dcolor, alpha=0.07, lw=0.9, n_std=1.22, zorder=1
        )
        for _, sub in dsub.groupby("group"):
            if len(sub) > MAX_RAW_POINTS_PER_GROUP:
                sub = sub.sample(MAX_RAW_POINTS_PER_GROUP, random_state=RANDOM_STATE)
            xj = (sub["x_low_macro"].values + rng.normal(0, 0.004, len(sub))).clip(0, 1)
            yj = (sub["y_high_micro"].values + rng.normal(0, 0.004, len(sub))).clip(0, 1)
            ax.scatter(xj, yj, s=10, marker=marker, c=dcolor, alpha=0.06,
                       edgecolor="none", zorder=2)

    # Group medians.
    centers = []
    for dataset, dsub in df.groupby("dataset"):
        for group, sub in dsub.groupby("group"):
            centers.append({
                "dataset": dataset,
                "group": group,
                "x": sub["x_low_macro"].median(),
                "y": sub["y_high_micro"].median(),
                "n": len(sub),
            })
    centers = pd.DataFrame(centers)

    # Draw trajectories: Palleja curved, Infant straight, IBD straight.
    cdict = {k: v.set_index("group") for k, v in centers.groupby("dataset")}

    # Palleja curved arrows to avoid overlap.
    if "Palleja" in cdict:
        csub = cdict["Palleja"]
        a = (csub.loc["Pre-stable", "x"], csub.loc["Pre-stable", "y"])
        b = (csub.loc["Perturbation", "x"], csub.loc["Perturbation", "y"])
        c = (csub.loc["Recovery", "x"], csub.loc["Recovery", "y"])
        draw_arrow(ax, a, b, color=ARROW_COLORS["Palleja"], lw=2.25, rad=0.24, zorder=27)
        draw_arrow(ax, b, c, color=ARROW_COLORS["Palleja"], lw=2.25, rad=-0.22, zorder=27)

    # Infant straight arrow.
    if "Infant" in cdict:
        csub = cdict["Infant"]
        a = (csub.loc["Immature", "x"], csub.loc["Immature", "y"])
        b = (csub.loc["Mature", "x"], csub.loc["Mature", "y"])
        draw_arrow(ax, a, b, color=ARROW_COLORS["Infant"], lw=2.25, rad=0.0, zorder=28)

    # IBD straight arrow.
    if "IBD" in cdict:
        csub = cdict["IBD"]
        a = (csub.loc["Healthy", "x"], csub.loc["Healthy", "y"])
        b = (csub.loc["IBD", "x"], csub.loc["IBD", "y"])
        draw_arrow(ax, a, b, color=ARROW_COLORS["IBD"], lw=2.25, rad=0.0, zorder=29)

    # Median markers and labels (no error bars).
    for _, r in centers.iterrows():
        marker = DATASET_MARKERS[r["dataset"]]
        gcolor = GROUP_COLORS.get(r["group"], DATASET_COLORS[r["dataset"]])
        ax.scatter(
            r["x"], r["y"], s=118, marker=marker, c=gcolor,
            edgecolor="white", linewidth=1.1, zorder=32,
        )
        dx, dy, ha = LABEL_OFFSETS.get(r["group"], (0.012, 0.012, "left"))
        label_point(ax, r["x"], r["y"], r["group"], color=gcolor, dx=dx, dy=dy, ha=ha)

    # Slightly expanded limits avoid the bottom-right basin looking visually clipped.
    ax.set_xlim(-0.035, 1.095)
    ax.set_ylim(-0.035, 1.085)
    ax.set_xlabel("Low-frequency macro-organization")
    ax.set_ylabel("High-frequency micro-fragmentation")
    ax.set_title("Macro–micro spectral phase space of host microbiomes", pad=10)
    style_axis(ax)

    legend_handles = [
        Line2D([0], [0], marker=DATASET_MARKERS[d], color="none",
               markerfacecolor=DATASET_COLORS[d], markeredgecolor="white",
               markeredgewidth=0.8, markersize=7.3, label=d)
        for d in ["Palleja", "Infant", "IBD"]
    ]
    ax.legend(
        handles=legend_handles, title="Dataset", frameon=True, fancybox=False,
        edgecolor="#E3D9CC", facecolor="white", loc="lower left",
        bbox_to_anchor=(0.02, 0.02), borderpad=0.45,
        handletextpad=0.45, labelspacing=0.35,
    )

    ax.text(
        0.0, -0.12,
        "Each point is a sample; large markers show group medians. "
        "Axes are derived from within-dataset robust scaling of low-dominance and high/low spectral ratios, "
        "followed by a monotonic contrast stretch for visual separation.",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.35, color="#7A7370",
    )

    fig.savefig(OUT_DIR / "host_macro_micro_phase_space.png", bbox_inches="tight", dpi=SAVE_DPI)
    fig.savefig(OUT_DIR / "host_macro_micro_phase_space.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[saved]", OUT_DIR / "host_macro_micro_phase_space.png")
    print("[saved]", OUT_DIR / "host_macro_micro_phase_space.pdf")


def main() -> None:
    df = load_all_metrics()
    plot_host_phase_space(df)


if __name__ == "__main__":
    main()
