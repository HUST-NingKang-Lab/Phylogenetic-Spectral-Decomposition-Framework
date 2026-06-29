#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MGnify emergent spectral compressibility — memory-safe v6
---------------------------------------------------------
Changes relative to v2
1. Replaces the previous yellow-heavy palette with a cleaner, less repetitive palette.
2. Adds two standalone supplementary figures:
   - low-order energy concentration (E10 / E20)
   - cumulative-energy difference curves among richness tiers
3. Keeps boxplots without overlaid points.
4. Keeps top-8 biome filtering identical to the original MGnify spectral_slope script.
"""
from __future__ import annotations

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, kruskal

# =============================================================================
# 0. Paths and parameters
# =============================================================================
abu_path = "/home/zhangyuli/傅里叶非时序/data/mgnify/abu.h5"
meta_path = "/home/zhangyuli/傅里叶非时序/data/mgnify/metadata.csv"
phylo_path = "/home/zhangyuli/傅里叶非时序/data/phylogeny.csv"

out_dir = Path("/home/zhangyuli/傅里叶非时序/figures/mgnify/emergent_spectral_compressibility")
out_dir.mkdir(parents=True, exist_ok=True)

min_samples_per_biome = 100
top_n_plot = 8
biome_col = "level_3"
max_samples_per_biome_for_fft = 1200
random_state = 42

pseudocount = 1e-9
fmax = 0.20
cumulative_threshold = 0.80
richness_threshold = 0.0
save_dpi = 600

# =============================================================================
# 1. Cleaner palette (reduced yellow usage)
# =============================================================================
COL = {
    "dark": "#452A3D",      # color 3
    "grid": "#E9E3DA",
    "text_grey": "#7E746B",
    "trend": "#452A3D",
    "box_edge": "#452A3D",
}

# Global richness-tier colors
TIER_ORDER = ["Low richness", "Mid richness", "High richness"]
TIER_COLORS = {
    "Low richness": "#B7B5A0",   # muted warm grey
    "Mid richness": "#E5855D",   # warm orange
    "High richness": "#44757A",  # teal
}

# Distinct biome colors with minimal repetition and no dominant yellow
BIOME_COLORS = {
    "Human": "#C56A48",
    "Aquatic": "#3E7897",
    "Mammals": "#A94955",
    "Terrestrial": "#9A7934",
    "Plants": "#5A8C5A",
    "Birds": "#7A3446",
    "Animal": "#B95B7A",
    "Wastewater": "#6F6087",
}

# Difference-curve colors
DIFF_COLORS = {
    "High - Low": "#D44C3C",
    "High - Mid": "#B66065",
    "Mid - Low": "#44757A",
}

# =============================================================================
# 2. Plot style helpers
# =============================================================================
def set_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.4,
        "axes.titlesize": 8.7,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 6.9,
        "legend.fontsize": 6.4,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": save_dpi,
        "figure.dpi": 160,
    })


def style_axis(ax, grid_axis="both"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COL["dark"])
    ax.spines["bottom"].set_color(COL["dark"])
    ax.tick_params(colors=COL["dark"])
    if grid_axis == "y":
        ax.grid(axis="y", color=COL["grid"], lw=0.65, alpha=0.85)
    elif grid_axis == "x":
        ax.grid(axis="x", color=COL["grid"], lw=0.65, alpha=0.85)
    else:
        ax.grid(color=COL["grid"], lw=0.65, alpha=0.85)
    ax.set_axisbelow(True)


def add_panel_label(ax, label):
    ax.text(-0.14, 1.06, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="left", va="bottom",
            color=COL["dark"])


def format_p(p):
    if not np.isfinite(p):
        return "P=NA"
    if p < 1e-4:
        return "P<1e-4"
    if p < 0.001:
        return f"P={p:.1e}"
    return f"P={p:.3g}"


def norm_taxon(x):
    x = str(x).strip()
    if x.startswith("sk__"):
        x = "k__" + x[4:]
    return x

# =============================================================================
# 3. Load and filter
# =============================================================================
def load_and_filter():
    print("[load] abundance, metadata, phylogeny")
    abu = pd.read_hdf(abu_path, "genus")
    meta = pd.read_csv(meta_path)
    phylo = pd.read_csv(phylo_path)

    meta = meta.rename(columns={"SampleID": "sample", "Env": "biome"})
    meta["sample"] = meta["sample"].astype(str)
    meta = meta.set_index("sample")

    env_split = meta["biome"].astype(str).str.split(":", expand=True)
    for i in range(env_split.shape[1]):
        meta[f"level_{i+1}"] = env_split[i]

    abu.index = abu.index.astype(str)
    abu.columns = [norm_taxon(c) for c in abu.columns]

    phy = phylo.iloc[:, 0].astype(str).str.split(";", expand=True)
    phy.index = phy[5]
    phy = phy[~phy.index.duplicated(keep="first")]
    fullnames = phy[0] + ";" + phy[1] + ";" + phy[2] + ";" + phy[3] + ";" + phy[4] + ";" + phy[5]

    inter = abu.columns.intersection(fullnames.index)
    abu = abu.loc[:, inter]
    abu.columns = fullnames.loc[abu.columns].values

    shared_samples = abu.index.intersection(meta.index)
    abu = abu.loc[shared_samples]
    meta = meta.loc[shared_samples]

    meta = meta[meta[biome_col].notna()]
    abu = abu.loc[meta.index]

    counts = meta[biome_col].value_counts()
    valid_biomes = counts[counts >= min_samples_per_biome].index.tolist()
    meta = meta[meta[biome_col].isin(valid_biomes)]
    abu = abu.loc[meta.index]

    ordered_cols = [c for c in fullnames.values if c in abu.columns]
    abu = abu[ordered_cols]
    abu = abu.loc[:, abu.sum(axis=0) > 0]

    biome_order = meta[biome_col].value_counts().index[:top_n_plot].tolist()
    meta = meta[meta[biome_col].isin(biome_order)].copy()
    abu = abu.loc[meta.index].copy()
    meta["biome_plot"] = pd.Categorical(meta[biome_col], categories=biome_order, ordered=True)

    print("[data] full top-8 samples:", abu.shape[0])
    print("[data] taxa after phylogeny filter:", abu.shape[1])
    print("[data] top 8 biomes:", biome_order)
    print(meta["biome_plot"].value_counts())

    if max_samples_per_biome_for_fft is not None:
        keep = []
        rng = np.random.default_rng(random_state)
        for biome in biome_order:
            idx = meta.index[meta["biome_plot"].astype(str).eq(biome)].to_numpy()
            if len(idx) > max_samples_per_biome_for_fft:
                idx = rng.choice(idx, size=max_samples_per_biome_for_fft, replace=False)
            keep.extend(idx.tolist())
        keep = pd.Index(keep).intersection(abu.index)
        abu = abu.loc[keep].copy()
        meta = meta.loc[keep].copy()
        meta["biome_plot"] = pd.Categorical(meta[biome_col], categories=biome_order, ordered=True)
        print("[sample] using balanced subset for FFT")
        print("[sample] max_samples_per_biome_for_fft:", max_samples_per_biome_for_fft)
        print("[sample] samples used:", abu.shape[0])
        print(meta["biome_plot"].value_counts())

    return abu, meta, biome_order

# =============================================================================
# 4. Spectral metrics
# =============================================================================
def fit_beta_from_psd(psd, freq_array, fmin, fmax):
    mask = (freq_array >= fmin) & (freq_array <= fmax) & np.isfinite(psd) & (psd > 0)
    xf = np.log10(freq_array[mask])
    yp = np.log10(psd[mask])
    if len(xf) < 3:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(xf, yp, 1)
    yhat = slope * xf + intercept
    ss_res = np.sum((yp - yhat) ** 2)
    ss_tot = np.sum((yp - yp.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return -slope, intercept, r2


def first_fraction_reaching_threshold(cum_energy, threshold):
    idx = np.where(cum_energy >= threshold)[0]
    if len(idx) == 0:
        return 1.0
    return (idx[0] + 1) / len(cum_energy)


def compute_metrics(abu, meta):
    print("[compute] normalize + CLR + FFT")
    abu = abu.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    row_sums = abu.sum(axis=1)
    abu = abu.loc[row_sums > 0].div(row_sums[row_sums > 0], axis=0)
    meta = meta.loc[abu.index].copy()

    richness = (abu > richness_threshold).sum(axis=1).astype(int)

    n = abu.shape[1]
    window = np.hanning(n).astype(float)
    freq = np.fft.rfftfreq(n, d=1.0)
    fmin = 2.0 / n
    mask = (freq >= fmin) & (freq <= fmax)
    active_freq = freq[mask]
    m = int(mask.sum())
    mode_fraction = np.arange(1, m + 1) / m

    records = []
    cumulative_curves = []
    sample_ids = []

    for i, sid in enumerate(abu.index):
        if (i + 1) % 1000 == 0:
            print(f"  processed {i+1}/{abu.shape[0]} samples")
        x = abu.loc[sid].values.astype(float)
        logx = np.log(x + pseudocount)
        clr = logx - logx.mean()
        xw = clr * window

        psd = (np.abs(np.fft.rfft(xw)) ** 2) / n
        active_power = psd[mask].astype(float)
        total_power = float(np.nansum(active_power))
        if total_power <= 0 or not np.isfinite(total_power):
            continue

        p = active_power / total_power
        cum = np.cumsum(p)
        c50 = first_fraction_reaching_threshold(cum, 0.50)
        c80 = first_fraction_reaching_threshold(cum, 0.80)
        c90 = first_fraction_reaching_threshold(cum, 0.90)
        entropy = -float(np.nansum(p * np.log(p + 1e-30)))
        deff_abs = float(np.exp(entropy))
        deff_norm = deff_abs / m
        entropy_norm = entropy / np.log(m)
        n10 = max(1, int(np.ceil(0.10 * m)))
        n20 = max(1, int(np.ceil(0.20 * m)))
        e10 = float(cum[n10 - 1])
        e20 = float(cum[n20 - 1])
        centroid = float(np.nansum(active_freq * p) / (active_freq.max() + 1e-30))
        beta, intercept, r2 = fit_beta_from_psd(psd, freq, fmin=fmin, fmax=fmax)

        records.append({
            "sample": sid,
            "biome": str(meta.loc[sid, "biome_plot"]),
            "richness": int(richness.loc[sid]),
            "log10_richness": float(np.log10(richness.loc[sid] + 1)),
            "C50": c50,
            "C80": c80,
            "C90": c90,
            "E10_low_order_energy": e10,
            "E20_low_order_energy": e20,
            "spectral_entropy_norm": entropy_norm,
            "effective_spectral_dimension": deff_norm,
            "effective_mode_number": deff_abs,
            "spectral_centroid_norm": centroid,
            "beta": beta,
            "beta_r2": r2,
            "n_active_modes": m,
            "n_taxa": n,
            "fmin": fmin,
            "fmax": fmax,
        })
        cumulative_curves.append(cum)
        sample_ids.append(sid)

    metrics = pd.DataFrame(records).set_index("sample")
    curves = np.vstack(cumulative_curves)

    ranks = metrics["richness"].rank(method="first")
    metrics["richness_tier"] = pd.qcut(ranks, 3, labels=TIER_ORDER)
    metrics["log10_richness_resid"] = metrics["log10_richness"] - metrics.groupby("biome")["log10_richness"].transform("median")
    metrics["C80_resid"] = metrics["C80"] - metrics.groupby("biome")["C80"].transform("median")
    metrics["Deff_resid"] = metrics["effective_spectral_dimension"] - metrics.groupby("biome")["effective_spectral_dimension"].transform("median")

    tier_by_sample = metrics.loc[sample_ids, "richness_tier"].astype(str).values
    print("[compute] done")
    return metrics, curves, mode_fraction, tier_by_sample

# =============================================================================
# 5. Stats helpers
# =============================================================================
def spearman_safe(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    mask = x.notna() & y.notna()
    if mask.sum() < 5:
        return np.nan, np.nan
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p)


def line_fit(x, y, xgrid=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None, None
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    if xgrid is None:
        xgrid = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
    return xgrid, slope * xgrid + intercept

# =============================================================================
# 6. Main figure
# =============================================================================
def make_main_figure(metrics, curves, mode_fraction, tier_by_sample, biome_order):
    set_style()
    fig = plt.figure(figsize=(10.4, 6.8))
    gs = fig.add_gridspec(2, 3, wspace=0.34, hspace=0.47)

    # a. cumulative curves by richness tier
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "a")
    for tier in TIER_ORDER:
        idx = np.where(tier_by_sample == tier)[0]
        if len(idx) == 0:
            continue
        sub = curves[idx, :]
        med = np.nanmedian(sub, axis=0)
        q1 = np.nanquantile(sub, 0.25, axis=0)
        q3 = np.nanquantile(sub, 0.75, axis=0)
        ax.plot(mode_fraction, med, color=TIER_COLORS[tier], lw=2.0, label=tier)
        ax.fill_between(mode_fraction, q1, q3, color=TIER_COLORS[tier], alpha=0.14, lw=0)
    ax.axhline(cumulative_threshold, color=COL["dark"], ls="--", lw=0.8, alpha=0.55)
    ax.text(0.96, cumulative_threshold + 0.025, "80% energy", ha="right", va="bottom", fontsize=6.6, color=COL["dark"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Fraction of frequency modes\naccumulated from low to high")
    ax.set_ylabel("Cumulative spectral energy")
    ax.set_title("Emergent spectral compressibility")
    ax.legend(frameon=False, loc="lower right", handlelength=1.8)
    style_axis(ax)

    # b. richness vs C80
    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, "b")
    for biome in biome_order:
        sub = metrics[metrics["biome"].eq(biome)]
        ax.scatter(sub["richness"], sub["C80"], s=14, c=BIOME_COLORS.get(biome, "#00A087"), alpha=0.28, edgecolor="none")
    xgrid, yhat = line_fit(metrics["richness"], metrics["C80"])
    if xgrid is not None:
        ax.plot(xgrid, yhat, color=COL["trend"], lw=1.5)
    rho_all, p_all = spearman_safe(metrics["richness"], metrics["C80"])
    rho_res, p_res = spearman_safe(metrics["log10_richness_resid"], metrics["C80_resid"])
    ax.text(0.03, 0.96,
            f"Across samples: ρ={rho_all:.2f}, {format_p(p_all)}\nWithin-biome residuals: ρ={rho_res:.2f}, {format_p(p_res)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.7, color=COL["dark"])
    ax.set_xlabel("Taxonomic richness\n(observed genera)")
    ax.set_ylabel("C80 compressibility index")
    ax.set_title("Does richness imply complexity?")
    style_axis(ax)

    # c. within-biome residual relationship
    ax = fig.add_subplot(gs[0, 2])
    add_panel_label(ax, "c")
    for biome in biome_order:
        sub = metrics[metrics["biome"].eq(biome)]
        ax.scatter(sub["log10_richness_resid"], sub["C80_resid"], s=14,
                   c=BIOME_COLORS.get(biome, "#00A087"), alpha=0.28, edgecolor="none")
    xgrid, yhat = line_fit(metrics["log10_richness_resid"], metrics["C80_resid"])
    if xgrid is not None:
        ax.plot(xgrid, yhat, color=COL["trend"], lw=1.5)
    ax.axhline(0, color=COL["dark"], lw=0.75, ls=":", alpha=0.55)
    ax.axvline(0, color=COL["dark"], lw=0.75, ls=":", alpha=0.55)
    ax.text(0.03, 0.96, f"Biome-centered:\nρ={rho_res:.2f}, {format_p(p_res)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.7, color=COL["dark"])
    ax.set_xlabel("Richness residual\nwithin biome")
    ax.set_ylabel("C80 residual\nwithin biome")
    ax.set_title("Not explained by biome identity")
    style_axis(ax)

    # d. C80 by richness tier
    ax = fig.add_subplot(gs[1, 0])
    add_panel_label(ax, "d")
    pos = np.arange(1, 4)
    data = [metrics.loc[metrics["richness_tier"].astype(str).eq(t), "C80"].dropna().values for t in TIER_ORDER]
    bp = ax.boxplot(data, positions=pos, widths=0.56, patch_artist=True, showfliers=False,
                    medianprops=dict(color=COL["box_edge"], lw=1.2),
                    boxprops=dict(color=COL["box_edge"], lw=0.9),
                    whiskerprops=dict(color=COL["box_edge"], lw=0.9),
                    capprops=dict(color=COL["box_edge"], lw=0.9))
    for patch, tier in zip(bp["boxes"], TIER_ORDER):
        patch.set_facecolor(TIER_COLORS[tier])
        patch.set_alpha(0.42)
    _, p_kw = kruskal(*data)
    ax.text(0.03, 0.96, f"Kruskal {format_p(p_kw)}", transform=ax.transAxes, ha="left", va="top", fontsize=6.7)
    ax.text(0.97, 0.96, "lower = more compressible", transform=ax.transAxes, ha="right", va="top", fontsize=6.7, color=COL["text_grey"])
    ax.set_xticks(pos)
    ax.set_xticklabels(["Low", "Mid", "High"])
    ax.set_xlabel("Richness tier")
    ax.set_ylabel("C80: low-order modes needed\nfor 80% spectral energy")
    ax.set_title("High richness needs fewer modes")
    ax.set_ylim(0.5, 1.0)
    style_axis(ax, grid_axis="y")

    # e. effective spectral dimension by richness tier
    ax = fig.add_subplot(gs[1, 1])
    add_panel_label(ax, "e")
    data_deff = [metrics.loc[metrics["richness_tier"].astype(str).eq(t), "effective_spectral_dimension"].dropna().values for t in TIER_ORDER]
    bp = ax.boxplot(data_deff, positions=pos, widths=0.56, patch_artist=True, showfliers=False,
                    medianprops=dict(color=COL["box_edge"], lw=1.2),
                    boxprops=dict(color=COL["box_edge"], lw=0.9),
                    whiskerprops=dict(color=COL["box_edge"], lw=0.9),
                    capprops=dict(color=COL["box_edge"], lw=0.9))
    for patch, tier in zip(bp["boxes"], TIER_ORDER):
        patch.set_facecolor(TIER_COLORS[tier])
        patch.set_alpha(0.42)
    _, p_kw_deff = kruskal(*data_deff)
    rho_deff, p_deff = spearman_safe(metrics["richness"], metrics["effective_spectral_dimension"])
    ax.text(0.03, 0.96, f"Kruskal {format_p(p_kw_deff)}\nρ richness={rho_deff:.2f}, {format_p(p_deff)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.7)
    ax.set_xticks(pos)
    ax.set_xticklabels(["Low", "Mid", "High"])
    ax.set_xlabel("Richness tier")
    ax.set_ylabel("Effective spectral dimension\n(normalized mode number)")
    ax.set_title("Frequency-domain dimensionality")
    ax.set_ylim(0.5, 1.0)
    style_axis(ax, grid_axis="y")

    # f. biome-specific correlations
    ax = fig.add_subplot(gs[1, 2])
    add_panel_label(ax, "f")
    biome_stats = []
    for biome in biome_order:
        sub = metrics[metrics["biome"].eq(biome)]
        rho, p = spearman_safe(sub["richness"], sub["C80"])
        biome_stats.append({"biome": biome, "rho": rho, "p": p})
    biome_stats = pd.DataFrame(biome_stats)
    y_pos = np.arange(len(biome_order))[::-1]
    for y, biome in zip(y_pos, biome_order):
        row = biome_stats[biome_stats["biome"].eq(biome)].iloc[0]
        ax.scatter(row["rho"], y, s=50, c=BIOME_COLORS.get(biome, "#00A087"), edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(row["rho"] + 0.03, y, format_p(row["p"]), va="center", ha="left", fontsize=6.1, color=COL["text_grey"])
    ax.axvline(0, color=COL["dark"], lw=0.8, ls=":", alpha=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(biome_order)
    ax.set_xlim(-0.8, 0.45)
    ax.set_xlabel("Spearman ρ\nrichness vs C80")
    ax.set_title("Negative trends recur across biomes")
    style_axis(ax, grid_axis="x")

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=BIOME_COLORS.get(b, "#00A087"),
                      markeredgecolor="white", markeredgewidth=0.5, markersize=5.6, label=b)
               for b in biome_order]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.015),
               ncol=4, frameon=False, columnspacing=0.95, handletextpad=0.35)
    fig.suptitle("Richness reveals emergent spectral compressibility in microbial ecosystems",
                 x=0.5, y=1.01, fontsize=11.2)

    out_png = out_dir / "mgnify_emergent_spectral_compressibility_memorysafe_v6.png"
    out_pdf = out_dir / "mgnify_emergent_spectral_compressibility_memorysafe_v6.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=save_dpi)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("[saved]", out_png)
    print("[saved]", out_pdf)

    return {
        "rho_all_C80": rho_all,
        "p_all_C80": p_all,
        "rho_res_C80": rho_res,
        "p_res_C80": p_res,
        "rho_deff": rho_deff,
        "p_deff": p_deff,
        "p_kw_C80": p_kw,
        "p_kw_Deff": p_kw_deff,
        "biome_stats": biome_stats,
    }

# =============================================================================
# 7. Supplementary figures
# =============================================================================
def make_supplementary_loworder_energy(metrics):
    set_style()
    fig = plt.figure(figsize=(7.0, 3.2))
    gs = fig.add_gridspec(1, 2, wspace=0.35)
    variables = [
        ("E10_low_order_energy", "Energy captured by\nlowest 10% modes", "E10"),
        ("E20_low_order_energy", "Energy captured by\nlowest 20% modes", "E20"),
    ]
    pvals = {}
    for j, (var, ylabel, title_tag) in enumerate(variables):
        ax = fig.add_subplot(gs[0, j])
        add_panel_label(ax, chr(ord('a') + j))
        pos = np.arange(1, 4)
        data = [metrics.loc[metrics["richness_tier"].astype(str).eq(t), var].dropna().values for t in TIER_ORDER]
        bp = ax.boxplot(data, positions=pos, widths=0.56, patch_artist=True, showfliers=False,
                        medianprops=dict(color=COL["box_edge"], lw=1.2),
                        boxprops=dict(color=COL["box_edge"], lw=0.9),
                        whiskerprops=dict(color=COL["box_edge"], lw=0.9),
                        capprops=dict(color=COL["box_edge"], lw=0.9))
        for patch, tier in zip(bp["boxes"], TIER_ORDER):
            patch.set_facecolor(TIER_COLORS[tier])
            patch.set_alpha(0.42)
        _, p_kw = kruskal(*data)
        pvals[var] = p_kw
        ax.text(0.03, 0.96, f"Kruskal {format_p(p_kw)}", transform=ax.transAxes, ha="left", va="top", fontsize=6.7)
        ax.set_xticks(pos)
        ax.set_xticklabels(["Low", "Mid", "High"])
        ax.set_xlabel("Richness tier")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Low-order energy concentration ({title_tag})")
        style_axis(ax, grid_axis="y")

    fig.suptitle("Supplementary Figure: low-order energy concentration", y=1.02, fontsize=10.5)
    out_png = out_dir / "supp_low_order_energy_concentration.png"
    out_pdf = out_dir / "supp_low_order_energy_concentration.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=save_dpi)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("[saved]", out_png)
    print("[saved]", out_pdf)
    return pvals


def make_supplementary_difference_curves(curves, mode_fraction, tier_by_sample):
    set_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    # median curves per tier
    tier_curves = {}
    for tier in TIER_ORDER:
        idx = np.where(tier_by_sample == tier)[0]
        tier_curves[tier] = np.nanmedian(curves[idx, :], axis=0)

    diff_map = {
        "High - Low": tier_curves["High richness"] - tier_curves["Low richness"],
        "High - Mid": tier_curves["High richness"] - tier_curves["Mid richness"],
        "Mid - Low": tier_curves["Mid richness"] - tier_curves["Low richness"],
    }

    for name, diff in diff_map.items():
        ax.plot(mode_fraction, diff, lw=2.0, color=DIFF_COLORS[name], label=name)

    ax.axhline(0, color=COL["dark"], lw=0.85, ls=":", alpha=0.75)
    ax.axvline(0.10, color=COL["text_grey"], lw=0.8, ls="--", alpha=0.6)
    ax.axvline(0.20, color=COL["text_grey"], lw=0.8, ls="--", alpha=0.6)
    ax.text(0.10, 0.98, "10% modes", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.3, color=COL["text_grey"])
    ax.text(0.20, 0.98, "20% modes", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.3, color=COL["text_grey"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of frequency modes")
    ax.set_ylabel("Difference in cumulative energy")
    ax.set_title("Supplementary Figure: difference curves among richness tiers")
    ax.legend(frameon=False, loc="best", handlelength=2.0)
    style_axis(ax)

    out_png = out_dir / "supp_difference_curves_richness_tiers.png"
    out_pdf = out_dir / "supp_difference_curves_richness_tiers.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=save_dpi)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("[saved]", out_png)
    print("[saved]", out_pdf)

# =============================================================================
# 8. Report
# =============================================================================
def write_report(metrics, stats, biome_order):
    report = []
    report.append("MGnify emergent spectral compressibility analysis (memory-safe v6)")
    report.append("=" * 68)
    report.append("")
    report.append("Filtering rule")
    report.append("--------------")
    report.append(f"min_samples_per_biome = {min_samples_per_biome}")
    report.append(f"top_n_plot = {top_n_plot}")
    report.append(f"biome_col = {biome_col}")
    report.append(f"top-8 biome order = {biome_order}")
    report.append(f"max_samples_per_biome_for_fft = {max_samples_per_biome_for_fft}")
    report.append("")
    report.append("Richness-tier definition")
    report.append("------------------------")
    report.append("Low / Mid / High richness are global tertiles on the sampled data.")
    report.append("Richness values are first ranked (method='first') and then split into three equal-sized bins with qcut.")
    report.append("")
    report.append("Main result")
    report.append("-----------")
    report.append(f"Across samples richness vs C80: rho={stats['rho_all_C80']:.4f}, p={stats['p_all_C80']:.4e}")
    report.append(f"Within-biome residual richness vs residual C80: rho={stats['rho_res_C80']:.4f}, p={stats['p_res_C80']:.4e}")
    report.append(f"Richness vs effective spectral dimension: rho={stats['rho_deff']:.4f}, p={stats['p_deff']:.4e}")
    out_txt = out_dir / "mgnify_emergent_spectral_story_memorysafe_v6.txt"
    out_txt.write_text("\n".join(report), encoding="utf-8")
    print("[saved]", out_txt)

# =============================================================================
# 9. Main
# =============================================================================
def main():
    abu, meta, biome_order = load_and_filter()
    metrics, curves, mode_fraction, tier_by_sample = compute_metrics(abu, meta)
    metrics.to_csv(out_dir / "mgnify_emergent_spectral_metrics_memorysafe_v6.csv", index=True)
    print("[saved]", out_dir / "mgnify_emergent_spectral_metrics_memorysafe_v6.csv")
    stats = make_main_figure(metrics, curves, mode_fraction, tier_by_sample, biome_order)
    write_report(metrics, stats, biome_order)
    print("[done]")

if __name__ == "__main__":
    main()
