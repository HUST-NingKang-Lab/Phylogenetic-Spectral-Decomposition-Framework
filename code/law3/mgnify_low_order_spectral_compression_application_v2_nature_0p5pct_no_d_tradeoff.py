#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MGnify low-order spectral compression application (revised)
==========================================================

This script extends the richness-compressibility story by asking:

Can a small LOW-ORDER spectral bandwidth preserve enough ecological information
to classify biome identity, while reducing feature dimensionality and computation?

Key updates relative to the previous version
--------------------------------------------
1. Keep the first k LOWEST-frequency Fourier modes only
   (no supervised top-mode selection).
2. Re-draw the main application figure to include:
   - macro-AUC vs low-order mode fraction, with alpha-combined and full-abundance ROC-AUC references,
   - macro-average ROC,
   - per-biome ROC for 0.5% low-order modes,
   - no accuracy bar panel,
   - dimensionality / runtime tradeoff.
3. Use FULL TAXON ABUNDANCE (raw abundance table) as the high-dimensional
   reference instead of CLR-transformed taxon features.
4. Use the color palette from the earlier acc_roc script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullLocator

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

warnings.filterwarnings("ignore")


# =============================================================================
# 0. Paths and parameters
# =============================================================================

ABU_PATH = "/home/zhangyuli/傅里叶非时序/data/mgnify/abu.h5"
META_PATH = "/home/zhangyuli/傅里叶非时序/data/mgnify/metadata.csv"
PHYLO_PATH = "/home/zhangyuli/傅里叶非时序/data/phylogeny.csv"

ABU_KEY = "genus"   # set to "species" if you want the species table
OUT_DIR = Path("/home/zhangyuli/傅里叶非时序/figures/mgnify/low_order_spectral_compression_application")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIOME_COL = "level_3"
TOP_N_BIOMES = 8
MIN_SAMPLES_PER_BIOME = 1000
N_SAMPLES_PER_BIOME = 1000

PSEUDOCOUNT = 1e-9
USE_HANNING_WINDOW = True

RANDOM_STATE = 123
N_SPLITS = 3
SAVE_DPI = 600

# Fractions of non-zero Fourier modes retained from low to high frequency
LOW_ORDER_FRACTIONS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
FOCAL_LOW_ORDER_FRACTION = 0.005  # explicit 0.5% low-order mode representation used in focal panels


# =============================================================================
# 1. Style and colors (matched to the earlier acc_roc script)
# =============================================================================

COL = {
    "dark": "#3A2634",
    "grey": "#AFA99B",
    "grid": "#E8E3DB",
}

FIG_METHOD_COLORS = {
    "Shannon": "#5D8790",
    "Richness": "#D75A49",
    "Chao1": "#6FAE8D",
    "Evenness": "#D8A24C",
    "Alpha diversity combined": "#6B5570",
    "Minimal low-order bandwidth": "#57928B",
    "Full taxon abundance": "#A8BCCC",
}

BIOME_COLORS = [
    "#5D8790", "#D75A49", "#6FAE8D", "#D8A24C",
    "#6B5570", "#B98B55", "#57928B", "#A8BCCC",
]


def set_nature_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.3,
        "axes.titlesize": 8.4,
        "axes.labelsize": 7.6,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.1,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": SAVE_DPI,
        "figure.dpi": 160,
    })


def style_axis(ax, grid: bool = True, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COL["dark"])
    ax.spines["bottom"].set_color(COL["dark"])
    ax.tick_params(colors=COL["dark"], length=3, width=0.75)
    if grid:
        if grid_axis == "both":
            ax.grid(color=COL["grid"], lw=0.65, alpha=0.85)
        else:
            ax.grid(axis=grid_axis, color=COL["grid"], lw=0.65, alpha=0.85)
        ax.set_axisbelow(True)


def panel_label(ax, label: str) -> None:
    ax.text(-0.115, 1.075, label, transform=ax.transAxes,
            fontsize=10.0, fontweight="bold", va="top", ha="left", color=COL["dark"])


def format_fraction_label(frac: float) -> str:
    pct = frac * 100
    if pct < 1:
        return f"{pct:.2g}%"
    if pct < 10:
        return f"{pct:.1f}%"
    return f"{pct:.0f}%"


def format_time(t: float) -> str:
    if not np.isfinite(t):
        return "NA"
    if t < 1:
        return f"{t*1000:.0f} ms"
    if t < 60:
        return f"{t:.1f} s"
    return f"{t/60:.1f} min"


set_nature_style()


# =============================================================================
# 2. Data loading and preprocessing
# =============================================================================

def norm_taxon(x: str) -> str:
    x = str(x).strip()
    if x.startswith("sk__"):
        x = "k__" + x[4:]
    return x


def short_taxon_name(fullname: str) -> str:
    s = str(fullname)
    last = s.split(";")[-1]
    if "__" in last:
        last = last.split("__")[-1]
    return last if last else s


def load_mgnify_top8_balanced() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    print("[load] abundance, metadata, phylogeny")
    abu = pd.read_hdf(ABU_PATH, ABU_KEY)
    meta = pd.read_csv(META_PATH)
    phylo = pd.read_csv(PHYLO_PATH)

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

    meta = meta[meta[BIOME_COL].notna()]
    abu = abu.loc[meta.index]

    counts = meta[BIOME_COL].value_counts()
    biome_order = counts[counts >= MIN_SAMPLES_PER_BIOME].index[:TOP_N_BIOMES].tolist()
    if len(biome_order) < TOP_N_BIOMES:
        raise ValueError(f"Only {len(biome_order)} biomes have >= {MIN_SAMPLES_PER_BIOME} samples.")

    meta = meta[meta[BIOME_COL].isin(biome_order)].copy()
    abu = abu.loc[meta.index].copy()
    meta[BIOME_COL] = pd.Categorical(meta[BIOME_COL], categories=biome_order, ordered=True)

    rng = np.random.default_rng(RANDOM_STATE)
    sampled_idx: List[str] = []
    for biome in biome_order:
        ids = meta.index[meta[BIOME_COL].astype(str) == str(biome)].to_numpy()
        ids = rng.choice(ids, size=N_SAMPLES_PER_BIOME, replace=False)
        sampled_idx.extend(ids.tolist())

    sampled_idx = np.array(sampled_idx, dtype=object)
    rng.shuffle(sampled_idx)

    meta = meta.loc[sampled_idx].copy()
    abu = abu.loc[meta.index].copy()

    ordered_cols = [c for c in fullnames.values if c in abu.columns]
    abu = abu[ordered_cols]
    abu = abu.loc[:, abu.sum(axis=0) > 0]

    taxa_order_df = pd.DataFrame({
        "taxon_order": np.arange(1, len(abu.columns) + 1),
        "taxon_fullname": abu.columns,
        "taxon_short": [short_taxon_name(c) for c in abu.columns],
    })

    print("[data] samples:", abu.shape[0])
    print("[data] taxa:", abu.shape[1])
    print("[data] biomes:", biome_order)
    return abu, meta, taxa_order_df, biome_order


def relative_and_clr(abu: pd.DataFrame) -> pd.DataFrame:
    row_sums = abu.sum(axis=1)
    abu = abu.loc[row_sums > 0]
    rel = abu.div(abu.sum(axis=1), axis=0).astype(np.float32)
    log_rel = np.log(rel + np.float32(PSEUDOCOUNT)).astype(np.float32)
    clr = log_rel.sub(log_rel.mean(axis=1), axis=0).astype(np.float32)
    return clr


def alpha_diversity_from_counts(abu: pd.DataFrame) -> pd.DataFrame:
    arr = abu.values.astype(float)
    row_sum = arr.sum(axis=1)
    rel = np.divide(arr, row_sum[:, None], out=np.zeros_like(arr), where=row_sum[:, None] > 0)
    log_rel = np.where(rel > 0, np.log(rel), 0.0)

    shannon = -np.sum(rel * log_rel, axis=1)
    richness = np.sum(arr > 0, axis=1).astype(float)
    evenness = np.divide(shannon, np.log(richness), out=np.zeros_like(shannon), where=richness > 1)

    rounded = np.rint(arr).astype(int)
    f1 = np.sum(rounded == 1, axis=1).astype(float)
    f2 = np.sum(rounded == 2, axis=1).astype(float)
    chao1 = richness + np.where(f2 > 0, (f1 * f1) / (2.0 * f2), (f1 * (f1 - 1.0)) / 2.0)

    return pd.DataFrame({
        "sample": abu.index.astype(str),
        "Shannon": shannon,
        "Richness": richness,
        "Chao1": chao1,
        "Evenness": evenness,
    }).set_index("sample")


# =============================================================================
# 3. Spectral features and classifiers
# =============================================================================

def compute_fft_modes(clr: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = clr.values.astype(np.float32, copy=False)
    n = X.shape[1]
    if USE_HANNING_WINDOW:
        X = X * np.hanning(n).astype(np.float32)[None, :]
    fft = np.fft.rfft(X, axis=1).astype(np.complex64)
    freq = np.fft.rfftfreq(n, d=1.0).astype(np.float32)
    mode_idx = np.arange(1, len(freq))
    coeff = fft[:, mode_idx]
    return coeff, freq[mode_idx], mode_idx


def coeff_to_pair_features(coeff: np.ndarray, mode_indices: np.ndarray | None = None) -> np.ndarray:
    sub = coeff if mode_indices is None else coeff[:, mode_indices]
    return np.concatenate([sub.real, sub.imag], axis=1).astype(np.float32)


def get_cv(y: np.ndarray) -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def make_clf():
    return make_pipeline(StandardScaler(), RidgeClassifier())


def cv_predictions_timed(X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    classes = np.unique(y)
    pred = np.empty_like(y)
    score = np.empty((len(y), len(classes)), dtype=float)

    t0 = time.perf_counter()
    for tr, te in get_cv(y).split(X, y):
        clf = make_clf()
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
        s = clf.decision_function(X[te])
        if s.ndim == 1:
            s = np.vstack([-s, s]).T
        score[te] = s
    elapsed = time.perf_counter() - t0

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "score": score,
        "cv_time_sec": float(elapsed),
    }


def roc_tables(y: np.ndarray, score: np.ndarray, class_names: List[str], representation: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    classes = np.arange(len(class_names))
    Y = label_binarize(y, classes=classes)

    curve_records = []
    auc_records = []
    mean_fpr = np.linspace(0.0, 1.0, 201)
    tprs = []

    for ci, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(Y[:, ci], score[:, ci])
        class_auc = auc(fpr, tpr)
        auc_records.append({"representation": representation, "class": cname, "auc": float(class_auc)})
        curve_records.extend([
            {"representation": representation, "class": cname, "curve": "class",
             "fpr": float(x), "tpr": float(yv), "auc": float(class_auc)}
            for x, yv in zip(fpr, tpr)
        ])

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    macro_tpr = np.mean(np.vstack(tprs), axis=0)
    macro_tpr[-1] = 1.0
    macro_auc = auc(mean_fpr, macro_tpr)
    auc_records.append({"representation": representation, "class": "macro", "auc": float(macro_auc)})
    curve_records.extend([
        {"representation": representation, "class": "macro", "curve": "macro",
         "fpr": float(x), "tpr": float(yv), "auc": float(macro_auc)}
        for x, yv in zip(mean_fpr, macro_tpr)
    ])

    return pd.DataFrame(curve_records), pd.DataFrame(auc_records)


# =============================================================================
# 4. Analysis
# =============================================================================

def run_analysis() -> Dict[str, object]:
    raw_abu, meta, taxa_order_df, biome_order = load_mgnify_top8_balanced()

    # FFT is still computed on phylogeny-ordered CLR
    clr = relative_and_clr(raw_abu)
    raw_abu = raw_abu.loc[clr.index]
    meta = meta.loc[clr.index]
    alpha = alpha_diversity_from_counts(raw_abu).loc[clr.index]

    y_raw = meta[BIOME_COL].astype(str).values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    coeff, freq, mode_idx = compute_fft_modes(clr)
    n_modes_total = coeff.shape[1]
    n_taxa = raw_abu.shape[1]

    pred_store: Dict[str, Dict[str, object]] = {}
    benchmark_records = []

    # alpha diversity
    alpha_metric_map = {
        "Shannon": ["Shannon"],
        "Richness": ["Richness"],
        "Chao1": ["Chao1"],
        "Evenness": ["Evenness"],
        "Alpha diversity combined": ["Shannon", "Richness", "Chao1", "Evenness"],
    }

    print("[cv] alpha diversity metrics")
    for name, cols in alpha_metric_map.items():
        X = alpha[cols].values.astype(np.float32)
        out = cv_predictions_timed(X, y)
        pred_store[name] = out
        curves, aucs = roc_tables(y, out["score"], class_names, name)
        macro_auc = aucs[(aucs["class"].eq("macro"))]["auc"].iloc[0]
        benchmark_records.append({
            "method": name,
            "representation_type": "alpha",
            "mode_fraction": np.nan,
            "n_modes": 0,
            "n_features": X.shape[1],
            "accuracy": out["accuracy"],
            "macro_auc": macro_auc,
            "cv_time_sec": out["cv_time_sec"],
            "feature_reduction_vs_taxa": n_taxa / X.shape[1],
        })

    # full abundance reference: ORIGINAL abundance table, not CLR
    print("[cv] full taxon abundance (raw)")
    X_full = raw_abu.values.astype(np.float32, copy=False)
    full_name = "Full taxon abundance"
    out = cv_predictions_timed(X_full, y)
    pred_store[full_name] = out
    curves, aucs = roc_tables(y, out["score"], class_names, full_name)
    macro_auc = aucs[(aucs["class"].eq("macro"))]["auc"].iloc[0]
    benchmark_records.append({
        "method": full_name,
        "representation_type": "full_taxon",
        "mode_fraction": np.nan,
        "n_modes": np.nan,
        "n_features": X_full.shape[1],
        "accuracy": out["accuracy"],
        "macro_auc": macro_auc,
        "cv_time_sec": out["cv_time_sec"],
        "feature_reduction_vs_taxa": 1.0,
    })

    # low-order modes only
    print("[cv] low-order spectral mode fractions")
    for frac in LOW_ORDER_FRACTIONS:
        k = int(np.ceil(frac * n_modes_total))
        k = max(1, min(k, n_modes_total))
        selected = np.arange(k)
        X = coeff_to_pair_features(coeff, selected)
        name = f"Low-order {format_fraction_label(frac)} modes"
        print(f"  {name}: k={k}, features={X.shape[1]}")
        out = cv_predictions_timed(X, y)
        pred_store[name] = out
        curves, aucs = roc_tables(y, out["score"], class_names, name)
        macro_auc = aucs[(aucs["class"].eq("macro"))]["auc"].iloc[0]
        benchmark_records.append({
            "method": name,
            "representation_type": "low_order_spectral",
            "mode_fraction": frac,
            "n_modes": k,
            "n_features": X.shape[1],
            "accuracy": out["accuracy"],
            "macro_auc": macro_auc,
            "cv_time_sec": out["cv_time_sec"],
            "feature_reduction_vs_taxa": n_taxa / X.shape[1],
        })

    benchmark = pd.DataFrame(benchmark_records)

    alpha_max_acc = benchmark.loc[benchmark["representation_type"].eq("alpha"), "accuracy"].max()
    alpha_max_auc = benchmark.loc[benchmark["representation_type"].eq("alpha"), "macro_auc"].max()
    low = benchmark[benchmark["representation_type"].eq("low_order_spectral")].sort_values("mode_fraction").copy()

    eligible = low[(low["accuracy"] > alpha_max_acc) & (low["macro_auc"] > alpha_max_auc)]
    if len(eligible) > 0:
        selected_row = eligible.iloc[0].copy()
        selected_reason = "minimal low-order fraction exceeding both best alpha accuracy and best alpha macro-AUC"
    else:
        selected_row = low.sort_values("macro_auc", ascending=False).iloc[0].copy()
        selected_reason = "fallback: best macro-AUC low-order representation"

    selected_method = str(selected_row["method"])

    # always keep an explicit 0.5% low-order representation for focal panels
    focal_name = f"Low-order {format_fraction_label(FOCAL_LOW_ORDER_FRACTION)} modes"
    if focal_name not in pred_store:
        raise ValueError(f"{focal_name} not found in pred_store; check LOW_ORDER_FRACTIONS.")

    # compact ROC set for macro comparison
    roc_methods = ["Shannon", "Richness", "Chao1", "Evenness", "Alpha diversity combined", focal_name, full_name]
    roc_curve_tables = []
    roc_auc_tables = []
    for rep in roc_methods:
        curves, aucs = roc_tables(y, pred_store[rep]["score"], class_names, rep)
        roc_curve_tables.append(curves)
        roc_auc_tables.append(aucs)

    roc_curves = pd.concat(roc_curve_tables, ignore_index=True)
    roc_aucs = pd.concat(roc_auc_tables, ignore_index=True)

    # ROC table specifically for 0.5% modes per biome
    per_biome_curves, per_biome_aucs = roc_tables(y, pred_store[focal_name]["score"], class_names, focal_name)

    return {
        "class_names": class_names,
        "biome_order": biome_order,
        "taxa_order_df": taxa_order_df,
        "n_taxa": n_taxa,
        "n_modes_total": n_modes_total,
        "benchmark": benchmark,
        "roc_curves": roc_curves,
        "roc_aucs": roc_aucs,
        "per_biome_curves": per_biome_curves,
        "per_biome_aucs": per_biome_aucs,
        "selected_method": selected_method,
        "selected_row": selected_row,
        "selected_reason": selected_reason,
        "alpha_max_acc": alpha_max_acc,
        "alpha_max_auc": alpha_max_auc,
        "full_method": full_name,
        "focal_method": focal_name,
        "per_biome_method": focal_name,
    }


# =============================================================================
# 5. Plotting
# =============================================================================

def plot_performance_vs_fraction(ax, res: Dict[str, object]) -> None:
    """Panel a: macro-ROC AUC only; accuracy is intentionally not shown."""
    df = res["benchmark"]
    low = df[df["representation_type"].eq("low_order_spectral")].sort_values("mode_fraction").copy()

    x = low["mode_fraction"].values * 100
    y = low["macro_auc"].values
    ax.plot(x, y, marker="o", ms=4.0, lw=1.8, color=FIG_METHOD_COLORS["Minimal low-order bandwidth"],
            label="Low-order spectral")

    alpha = df[df["method"].eq("Alpha diversity combined")].iloc[0]
    full = df[df["method"].eq(res["full_method"])].iloc[0]
    ax.axhline(alpha["macro_auc"], color=FIG_METHOD_COLORS["Alpha diversity combined"], lw=1.15,
               ls="--", alpha=0.92, label=f"Alpha combined ROC-AUC={alpha['macro_auc']:.3f}")
    ax.axhline(full["macro_auc"], color=FIG_METHOD_COLORS["Full taxon abundance"], lw=1.15,
               ls=":", alpha=0.98, label=f"Full abundance ROC-AUC={full['macro_auc']:.3f}")

    focal = df[df["method"].eq(res["focal_method"])].iloc[0]
    ax.axvline(focal["mode_fraction"] * 100, color="#D75A49", lw=0.9, alpha=0.48)
    ax.scatter(focal["mode_fraction"] * 100, focal["macro_auc"], s=54, marker="o",
               color="#D75A49", edgecolor="white", lw=0.8, zorder=5)
    ax.text(focal["mode_fraction"] * 100 * 1.07, focal["macro_auc"] + 0.008,
            f"{format_fraction_label(focal['mode_fraction'])}", fontsize=6.6, color="#D75A49", ha="left", va="bottom")

    ymin = max(0.0, min(low["macro_auc"].min(), alpha["macro_auc"], full["macro_auc"]) - 0.035)
    ymax = min(1.01, max(low["macro_auc"].max(), alpha["macro_auc"], full["macro_auc"]) + 0.035)
    ax.set_ylim(ymin, ymax)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.5, 1, 5, 10, 30])
    ax.set_xticklabels(["0.1", "0.5", "1", "5", "10", "30"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Lowest-frequency Fourier modes retained (%)")
    ax.set_ylabel("Macro-average ROC AUC")
    ax.set_title("Low-order bandwidth recovers ROC performance")
    ax.legend(frameon=False, loc="lower right", handlelength=1.8)
    style_axis(ax, grid=True, grid_axis="both")


def plot_accuracy_bar(ax, res: Dict[str, object]) -> None:
    df = res["benchmark"]
    focal = df[df["method"].eq(res["focal_method"])].iloc[0]
    selected_label = f"Low-order {format_fraction_label(focal['mode_fraction'])}"

    method_rows = [
        ("Shannon", "Shannon"),
        ("Richness", "Richness"),
        ("Chao1", "Chao1"),
        ("Evenness", "Evenness"),
        ("Alpha diversity combined", "Alpha diversity combined"),
        (focal["method"], selected_label),
        (res["full_method"], "Full taxon abundance"),
    ]

    plot_df = []
    for original, display in method_rows:
        row = df[df["method"].eq(original)].iloc[0].copy()
        row["display"] = display
        plot_df.append(row)
    plot_df = pd.DataFrame(plot_df)

    colors = [
        FIG_METHOD_COLORS["Shannon"],
        FIG_METHOD_COLORS["Richness"],
        FIG_METHOD_COLORS["Chao1"],
        FIG_METHOD_COLORS["Evenness"],
        FIG_METHOD_COLORS["Alpha diversity combined"],
        FIG_METHOD_COLORS["Minimal low-order bandwidth"],
        FIG_METHOD_COLORS["Full taxon abundance"],
    ]
    ax.bar(np.arange(len(plot_df)), plot_df["accuracy"], color=colors, width=0.78)
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(plot_df["display"], rotation=35, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Biome classification accuracy")
    style_axis(ax, grid=True, grid_axis="y")


def plot_macro_roc(ax, res: Dict[str, object]) -> None:
    curves = res["roc_curves"]
    aucs = res["roc_aucs"]
    focal = res["benchmark"][res["benchmark"]["method"].eq(res["focal_method"])].iloc[0]
    focal_label = f"Low-order {format_fraction_label(focal['mode_fraction'])} ({int(focal['n_features'])} features)"

    method_display = [
        ("Shannon", "Shannon"),
        ("Richness", "Richness"),
        ("Chao1", "Chao1"),
        ("Evenness", "Evenness"),
        ("Alpha diversity combined", "Alpha diversity combined"),
        (res["focal_method"], focal_label),
        (res["full_method"], "Full taxon abundance"),
    ]
    color_map = {
        "Shannon": FIG_METHOD_COLORS["Shannon"],
        "Richness": FIG_METHOD_COLORS["Richness"],
        "Chao1": FIG_METHOD_COLORS["Chao1"],
        "Evenness": FIG_METHOD_COLORS["Evenness"],
        "Alpha diversity combined": FIG_METHOD_COLORS["Alpha diversity combined"],
        focal_label: FIG_METHOD_COLORS["Minimal low-order bandwidth"],
        "Full taxon abundance": FIG_METHOD_COLORS["Full taxon abundance"],
    }

    for rep, display in method_display:
        sub = curves[(curves["representation"].eq(rep)) & (curves["class"].eq("macro"))]
        macro_auc = aucs[(aucs["representation"].eq(rep)) & (aucs["class"].eq("macro"))]["auc"].iloc[0]
        label = f"{display} (AUC={macro_auc:.3f})"
        lw = 1.15 if display in {"Shannon", "Richness", "Chao1", "Evenness"} else 1.55
        alpha = 0.82 if display in {"Shannon", "Richness", "Chao1", "Evenness"} else 0.98
        ax.plot(sub["fpr"], sub["tpr"], lw=lw, alpha=alpha, color=color_map[display], label=label)

    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=COL["grey"], alpha=0.9)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Macro-average one-vs-rest ROC")
    ax.legend(frameon=False, loc="lower right", handlelength=1.65, borderaxespad=0.25)
    style_axis(ax, grid=False)


def plot_per_biome_roc(ax, res: Dict[str, object]) -> None:
    curves = res["per_biome_curves"]
    aucs = res["per_biome_aucs"]
    biome_names = res["class_names"]

    for i, biome in enumerate(biome_names):
        sub = curves[(curves["representation"].eq(res["per_biome_method"])) & (curves["class"].eq(biome))]
        class_auc = aucs[(aucs["representation"].eq(res["per_biome_method"])) & (aucs["class"].eq(biome))]["auc"].iloc[0]
        ax.plot(sub["fpr"], sub["tpr"], lw=1.25, color=BIOME_COLORS[i % len(BIOME_COLORS)],
                label=f"{biome} ({class_auc:.2f})")

    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=COL["grey"])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Per-biome ROC for {format_fraction_label(FOCAL_LOW_ORDER_FRACTION)} low-order modes")
    ax.legend(frameon=False, loc="lower right", ncol=1, handlelength=1.3, borderaxespad=0.2)
    style_axis(ax, grid=False)


def plot_tradeoff(ax, res: Dict[str, object]) -> None:
    """Panel e: compact feature/runtime tradeoff, returning to the earlier annotated point design."""
    df = res["benchmark"].copy()
    focal = df[df["method"].eq(res["focal_method"])].iloc[0]

    method_rows = [
        ("Alpha diversity combined", "Alpha diversity combined"),
        (res["focal_method"], f"Low-order {format_fraction_label(focal['mode_fraction'])}"),
        (res["full_method"], "Full taxon abundance"),
    ]

    xs, ys, colors, labels, notes = [], [], [], [], []
    for original, display in method_rows:
        row = df[df["method"].eq(original)].iloc[0]
        xs.append(row["n_features"])
        ys.append(row["macro_auc"])
        labels.append(display)

        if original == "Alpha diversity combined":
            colors.append(FIG_METHOD_COLORS["Alpha diversity combined"])
            note = (
                f"{int(row['n_features'])} features\n"
                f"{row['feature_reduction_vs_taxa']:.0f}× reduction\n"
                f"{format_time(row['cv_time_sec'])}"
            )
        elif original == res["focal_method"]:
            colors.append(FIG_METHOD_COLORS["Minimal low-order bandwidth"])
            note = (
                f"{int(row['n_features'])} features\n"
                f"{row['feature_reduction_vs_taxa']:.0f}× reduction\n"
                f"{format_time(row['cv_time_sec'])}"
            )
        else:
            colors.append(FIG_METHOD_COLORS["Full taxon abundance"])
            note = (
                f"{int(row['n_features'])} features\n"
                f"1× reference\n"
                f"{format_time(row['cv_time_sec'])}"
            )
        notes.append(note)

    ax.scatter(xs, ys, s=[58, 72, 58], c=colors, edgecolor="white", lw=0.85, zorder=4)
    ax.plot(xs, ys, lw=0.75, color=COL["grey"], alpha=0.55, zorder=2)

    for x, y, lab, note in zip(xs, ys, labels, notes):
        offset = 1.08 if lab != "Full taxon abundance" else 0.88
        ha = "left" if lab != "Full taxon abundance" else "right"
        ax.text(x * offset, y, f"{lab}\n{note}", fontsize=6.25, ha=ha, va="center", color=COL["dark"])

    ax.set_xscale("log")
    ax.set_xlabel("Number of input features (log scale)")
    ax.set_ylabel("Macro-average ROC AUC")
    ax.set_title("Spectral compression balances accuracy and cost")
    style_axis(ax, grid=True, grid_axis="both")

def plot_focal_summary(ax, res: Dict[str, object]) -> None:
    df = res["benchmark"]
    focal = df[df["method"].eq(res["focal_method"])].iloc[0]
    full = df[df["method"].eq(res["full_method"])].iloc[0]
    alpha = df[df["method"].eq("Alpha diversity combined")].iloc[0]

    ax.set_axis_off()
    ax.text(0.02, 0.95, "0.5% low-order spectral summary", transform=ax.transAxes,
            fontsize=8.0, fontweight="bold", color=COL["dark"], va="top")

    lines = [
        ("Retained modes", f"{int(focal['n_modes'])} / {int(res['n_modes_total'])}"),
        ("Real-valued features", f"{int(focal['n_features'])}"),
        ("Feature reduction", f"{focal['feature_reduction_vs_taxa']:.0f}× vs full abundance"),
        ("Accuracy", f"{focal['accuracy']:.3f}"),
        ("Macro-AUC", f"{focal['macro_auc']:.3f}"),
        ("CV time", f"{format_time(focal['cv_time_sec'])}"),
    ]

    y = 0.79
    for key, val in lines:
        ax.text(0.04, y, key, transform=ax.transAxes, fontsize=6.6, color=COL["grey"], va="center")
        ax.text(0.56, y, val, transform=ax.transAxes, fontsize=7.1, color=COL["dark"],
                fontweight="bold", va="center")
        y -= 0.105

    ax.text(0.04, 0.12,
            f"Reference: alpha-combined AUC={alpha['macro_auc']:.3f}; full abundance AUC={full['macro_auc']:.3f}.",
            transform=ax.transAxes, fontsize=6.2, color=COL["dark"], va="bottom", wrap=True)


def create_figure(res: Dict[str, object]) -> None:
    # Slightly narrower and less elongated than the previous wide layout.
    # Four panels are arranged in a balanced 2 × 2 grid; the last panel keeps the label "e"
    # because panel d has been intentionally removed.
    fig = plt.figure(figsize=(8.00, 7.25), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0],
                  wspace=0.34, hspace=0.50)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])

    panel_label(ax_a, "a")
    plot_performance_vs_fraction(ax_a, res)

    panel_label(ax_b, "b")
    plot_macro_roc(ax_b, res)

    panel_label(ax_c, "c")
    plot_per_biome_roc(ax_c, res)

    panel_label(ax_e, "e")
    plot_tradeoff(ax_e, res)

    fig.suptitle("Low-order spectral compression preserves biome identity with 0.5% Fourier modes",
                 y=0.985, fontsize=11.2, color=COL["dark"])
    fig.subplots_adjust(left=0.065, right=0.975, bottom=0.095, top=0.895)

    out_png = OUT_DIR / "figure_low_order_spectral_compression_application_v2_nature_0p5pct_no_acc_tradeoff.png"
    out_pdf = OUT_DIR / "figure_low_order_spectral_compression_application_v2_nature_0p5pct_no_acc_tradeoff.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=SAVE_DPI)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("[saved]", out_png)
    print("[saved]", out_pdf)


# =============================================================================
# 6. Output tables and report
# =============================================================================

def write_outputs(res: Dict[str, object]) -> None:
    res["benchmark"].to_csv(OUT_DIR / "low_order_compression_benchmark.csv", index=False)
    res["roc_curves"].to_csv(OUT_DIR / "macro_roc_curves.csv", index=False)
    res["roc_aucs"].to_csv(OUT_DIR / "macro_roc_aucs.csv", index=False)
    res["per_biome_curves"].to_csv(OUT_DIR / "per_biome_roc_curves_0p5pct.csv", index=False)
    res["per_biome_aucs"].to_csv(OUT_DIR / "per_biome_roc_aucs_0p5pct.csv", index=False)
    res["taxa_order_df"].to_csv(OUT_DIR / "taxa_order_used.csv", index=False)

    selected = res["selected_row"]
    full = res["benchmark"][res["benchmark"]["method"].eq(res["full_method"])].iloc[0]
    alpha_comb = res["benchmark"][res["benchmark"]["method"].eq("Alpha diversity combined")].iloc[0]

    report = []
    report.append("Low-order spectral compression application (revised)")
    report.append("=" * 60)
    report.append("")
    report.append("Core design")
    report.append("-----------")
    report.append("1. Use the first k LOWEST-frequency Fourier modes only; no supervised top-mode selection.")
    report.append("2. Use FULL TAXON ABUNDANCE (raw abundance matrix) as the high-dimensional reference.")
    report.append("3. Select the minimal low-order fraction that exceeds both the best alpha-diversity accuracy and macro-AUC.")
    report.append("4. Additionally report per-biome ROC for the explicit 0.5% low-order bandwidth.")
    report.append("")
    report.append("Dataset")
    report.append("-------")
    report.append(f"ABU_KEY = {ABU_KEY}")
    report.append(f"Top biomes = {', '.join(res['biome_order'])}")
    report.append(f"Samples = {TOP_N_BIOMES} biomes × {N_SAMPLES_PER_BIOME} samples")
    report.append(f"Taxon features = {res['n_taxa']}")
    report.append(f"Non-zero Fourier modes = {res['n_modes_total']}")
    report.append("")
    report.append("Minimal low-order bandwidth")
    report.append("--------------------------")
    report.append(f"Selected method = {res['selected_method']}  (selection retained for audit)")
    report.append(f"Selection rule = {res['selected_reason']}")
    report.append(f"Mode fraction = {selected['mode_fraction']:.6g}")
    report.append(f"Number of modes = {int(selected['n_modes'])}")
    report.append(f"Number of real-valued features = {int(selected['n_features'])}")
    report.append(f"Feature reduction vs full taxon abundance = {selected['feature_reduction_vs_taxa']:.2f}×")
    report.append(f"Accuracy = {selected['accuracy']:.4f}")
    report.append(f"Macro-AUC = {selected['macro_auc']:.4f}")
    report.append(f"CV time = {format_time(selected['cv_time_sec'])}")
    report.append("")
    report.append("Reference comparisons")
    report.append("---------------------")
    report.append(f"Alpha diversity combined: ACC={alpha_comb['accuracy']:.4f}, AUC={alpha_comb['macro_auc']:.4f}, time={format_time(alpha_comb['cv_time_sec'])}")
    report.append(f"Full taxon abundance: ACC={full['accuracy']:.4f}, AUC={full['macro_auc']:.4f}, time={format_time(full['cv_time_sec'])}")
    report.append(f"Full abundance time / selected low-order time = {full['cv_time_sec'] / selected['cv_time_sec']:.2f}×")
    report.append("")
    report.append("Full benchmark")
    report.append("--------------")
    report.append(res["benchmark"].to_string(index=False))

    out = OUT_DIR / "low_order_compression_selected_summary.txt"
    out.write_text("\n".join(report), encoding="utf-8")
    print("[saved]", out)


# =============================================================================
# 7. Main
# =============================================================================

def main() -> None:
    res = run_analysis()
    create_figure(res)
    write_outputs(res)
    print("[done]")


if __name__ == "__main__":
    main()
