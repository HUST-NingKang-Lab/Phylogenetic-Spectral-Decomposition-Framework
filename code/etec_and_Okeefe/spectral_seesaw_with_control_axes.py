#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset-specific phylogeny-ordered spectral response axes
=========================================================

Rationale
---------
This script follows a revised analysis strategy:

1) Discover the perturbation--recovery spectral axis from the full ETEC genus set.
2) Discover the reciprocal diet-swap spectral axis from the full O'Keefe genus set.
3) Do NOT force the two datasets to share the same genera for axis discovery.
4) Compare the two axes at the level of spectral-frequency loading profiles,
   rather than requiring the same taxonomic poles.
5) Validate the positive/negative axis-derived poles within each dataset using
   directional pole-balance and positive-vs-negative response relationships.

Main story this script is designed to test
------------------------------------------
Different perturbations need not share the same individual genera. Instead,
each perturbation can be represented by a dataset-specific phylogenetic spectral
axis: ETEC defines an infection-related perturbation--recovery axis, whereas
O'Keefe defines a reciprocal westernization--restoration diet axis. These axes
can then be compared through their spectral-frequency organization and their
axis-derived positive/negative taxonomic poles.

Outputs
-------
By default, this script writes only publication-style figures:
    - figure_dataset_specific_spectral_axes.pdf/png
    - figure_spectral_pole_seesaw_validation.pdf/png

Set SAVE_TABLES = True to export intermediate CSV files.
"""

from __future__ import annotations

from pathlib import Path
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.decomposition import PCA

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =============================================================================
# 0. User parameters
# =============================================================================

# Local project paths. The script will fall back to /mnt/data filenames if needed.
ETEC_ABU_PATH = Path("/home/zhangyuli/傅里叶非时序/data/etec/etec16s_abundance_genus.csv")
ETEC_META_PATH = Path("/home/zhangyuli/傅里叶非时序/data/etec/etec16s_metadata.csv")

OK_ABU_PATH = Path("/home/zhangyuli/傅里叶非时序/data/Okeefe/OKeefe_dietswap_abundance.csv")
OK_META_PATH = Path("/home/zhangyuli/傅里叶非时序/data/Okeefe/OKeefe_dietswap_metadata.csv")

PHYLO_PATH = Path("/home/zhangyuli/傅里叶非时序/data/phylogeny.csv")

OUT_DIR = Path("/home/zhangyuli/傅里叶非时序/figures/etec and okeefe/other_axes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ETEC design
ETEC_BASELINE_DAY = -1
ETEC_CHALLENGE_DAY = 0
ETEC_ACUTE_DAYS = [1, 2, 3, 4, 5, 6, 7, 9]
ETEC_RECOVERY_DAYS = [28, 84]

# O'Keefe design
OK_SAMPLE_COL = "sample"
OK_SUBJECT_COL = "subject"
OK_NATIONALITY_COL = "nationality"   # expected: AAM / AFR
OK_GROUP_COL = "group"               # expected: HE / DI / ED
OK_BASELINE_GROUP = "HE"
OK_INTERVENTION_GROUP = "DI"

# Spectral transform settings
PSEUDOCOUNT = 1e-6
FMAX = 0.45
WINDOW = "hann"  # "hann" or "none"
MIN_PREVALENCE = 0.02
MIN_TOTAL_COUNT = 10
MIN_TAXA = 20

# Candidate pole extraction
POLE_QUANTILE = 0.15

# Optional table export
SAVE_TABLES = False

RANDOM_SEED = 20260428
rng = np.random.default_rng(RANDOM_SEED)


# =============================================================================
# 1. Nature-like plotting style
# =============================================================================

def set_nature_style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.dpi": 160,
    })


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax, label: str) -> None:
    ax.text(-0.15, 1.08, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


set_nature_style()

COL = {
    "etec": "#D55E00",
    "recovery": "#0072B2",
    "afr": "#CC79A7",
    "aam": "#009E73",
    "dark": "#222222",
    "grey": "#BDBDBD",
    "lightgrey": "#D9D9D9",
    "pos": "#009E73",
    "neg": "#CC79A7",
    "freq_etec": "#D55E00",
    "freq_ok": "#6A51A3",
}


# =============================================================================
# 2. Utility functions: paths, taxa, data loading
# =============================================================================

TAXON_CORRECTIONS = {
    "Allistipes": "Alistipes",
    "Klebisiella": "Klebsiella",
    "Fusobacteria": "Fusobacterium",
    "Escherichia/Shigella": "Escherichia",
    "Clostridiales": "Clostridium",
}


def resolve_path(primary: Path, fallback_name: str) -> Path:
    if primary.exists():
        return primary
    fallback = Path("/mnt/data") / fallback_name
    if fallback.exists():
        return fallback
    return primary


def clean_taxon_label(x: str) -> str:
    x = str(x).strip().strip('"').strip("'")
    if x.lower() in {"nan", "none", "uncultured", "outgrouping", "incertae", ""}:
        return ""
    if ";" in x:
        x = x.split(";")[-1]
    for pref in ["sk__", "k__", "p__", "c__", "o__", "f__", "g__", "s__"]:
        if x.startswith(pref):
            x = x[len(pref):]
    x = x.replace(" et rel.", "").replace(" et rel", "")
    x = re.sub(r"\s+cluster.*$", "", x, flags=re.I)
    x = re.sub(r"\s+group.*$", "", x, flags=re.I)
    x = re.sub(r"\s+sensu.*$", "", x, flags=re.I)
    x = x.strip()
    if " " in x:
        x = x.split()[0]
    return TAXON_CORRECTIONS.get(x, x)


def read_abundance(path: Path) -> pd.DataFrame:
    """Read taxa x samples CSV and return samples x cleaned taxa."""
    df = pd.read_csv(path, index_col=0)
    df.index = [clean_taxon_label(t) for t in df.index]
    df = df.loc[[t != "" for t in df.index]]
    df = df.groupby(df.index).sum()
    abu = df.T
    abu.index = abu.index.astype(str)
    abu = abu.apply(pd.to_numeric, errors="coerce").fillna(0)
    return abu


def read_etec_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    first = meta.columns[0]
    meta = meta.rename(columns={first: "sample"})
    meta["sample"] = meta["sample"].astype(str)
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    meta["Day"] = pd.to_numeric(meta["Day"], errors="coerce")

    def phase(day):
        if day == ETEC_BASELINE_DAY:
            return "Baseline"
        if day == ETEC_CHALLENGE_DAY:
            return "Challenge"
        if day in ETEC_ACUTE_DAYS:
            return "Acute"
        if day in ETEC_RECOVERY_DAYS:
            return "Recovery"
        return "Other"

    meta["phase"] = meta["Day"].apply(phase)
    return meta.set_index("sample")


def read_okeefe_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    meta[OK_SAMPLE_COL] = meta[OK_SAMPLE_COL].astype(str)
    meta[OK_SUBJECT_COL] = meta[OK_SUBJECT_COL].astype(str)
    meta[OK_NATIONALITY_COL] = meta[OK_NATIONALITY_COL].astype(str)
    meta[OK_GROUP_COL] = meta[OK_GROUP_COL].astype(str)
    if "timepoint" in meta.columns:
        meta["timepoint"] = pd.to_numeric(meta["timepoint"], errors="coerce")
    return meta.set_index(OK_SAMPLE_COL)


def read_phylogenetic_order(path: Path) -> list[str]:
    phy = pd.read_csv(path)
    order, seen = [], set()
    for raw in phy.iloc[:, 0].astype(str):
        g = clean_taxon_label(raw)
        if g and g not in seen:
            order.append(g)
            seen.add(g)
    return order


def align_samples(abu: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = abu.index.intersection(meta.index)
    return abu.loc[common].copy(), meta.loc[common].copy()


def filter_taxa(abu: pd.DataFrame) -> pd.DataFrame:
    prev = (abu > 0).mean(axis=0)
    total = abu.sum(axis=0)
    keep = (prev >= MIN_PREVALENCE) & (total >= MIN_TOTAL_COUNT)
    return abu.loc[:, keep].copy()


def ordered_taxa_for_dataset(abu: pd.DataFrame, phy_order: list[str], dataset_name: str) -> list[str]:
    taxa = [t for t in phy_order if t in abu.columns]
    if len(taxa) < MIN_TAXA:
        raise ValueError(f"{dataset_name}: only {len(taxa)} taxa matched phylogenetic order.")
    return taxa


# =============================================================================
# 3. Spectral transform and axes
# =============================================================================

def clr_transform(abu: pd.DataFrame) -> pd.DataFrame:
    abu = abu.loc[abu.sum(axis=1) > 0].copy()
    rel = abu.div(abu.sum(axis=1), axis=0).fillna(0) + PSEUDOCOUNT
    log_rel = np.log(rel)
    return log_rel.sub(log_rel.mean(axis=1), axis=0)


def fft_features(clr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = clr.shape[1]
    if WINDOW == "hann":
        window = np.hanning(n)
    elif WINDOW == "none":
        window = np.ones(n)
    else:
        raise ValueError("WINDOW must be 'hann' or 'none'.")

    freq = np.fft.rfftfreq(n, d=1.0)
    Z = np.fft.rfft(clr.values.astype(float) * window[None, :], axis=1)
    freq_df = pd.DataFrame({
        "freq_id": [f"f{i:03d}" for i in range(len(freq))],
        "freq_index": np.arange(len(freq)),
        "freq": freq,
    })
    keep = freq_df[(freq_df["freq"] > 0) & (freq_df["freq"] <= FMAX)].copy()
    idx = keep["freq_index"].values
    Zk = Z[:, idx]
    feat = np.concatenate([Zk.real, Zk.imag], axis=1)
    cols = [f"Re_{fid}" for fid in keep["freq_id"]] + [f"Im_{fid}" for fid in keep["freq_id"]]
    feat_df = pd.DataFrame(feat, index=clr.index, columns=cols)
    return feat_df, freq_df, keep


def robust_scale(X: pd.DataFrame) -> pd.Series:
    med = X.median(axis=0)
    mad = (X - med).abs().median(axis=0)
    std = X.std(axis=0)
    scale = mad.copy()
    bad = (~np.isfinite(scale)) | (scale < 1e-12)
    scale.loc[bad] = std.loc[bad]
    scale[(~np.isfinite(scale)) | (scale < 1e-12)] = 1.0
    return scale


def normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v * np.nan


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a.dot(b) / den) if den > 1e-12 else np.nan


def score_delta(delta_scaled: pd.DataFrame, axis: pd.Series) -> pd.Series:
    return pd.Series(delta_scaled[axis.index].values @ axis.values, index=delta_scaled.index)


# =============================================================================
# 4. ETEC-specific analysis
# =============================================================================

def etec_baseline_deltas(X: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = meta[meta["Day"].eq(ETEC_BASELINE_DAY)]
    base_subjects = set(baseline["SubjectID"])
    rows, ids = [], []
    for sid in X.index:
        subj = meta.loc[sid, "SubjectID"]
        if subj not in base_subjects:
            continue
        b_ids = baseline[baseline["SubjectID"].eq(subj)].index
        base = X.loc[b_ids].mean(axis=0).values
        rows.append(X.loc[sid].values - base)
        ids.append(sid)
    return pd.DataFrame(rows, index=ids, columns=X.columns), meta.loc[ids].copy()


def mean_by_subject(df: pd.DataFrame, meta: pd.DataFrame, subject_col: str, mask: pd.Series) -> pd.DataFrame:
    ids = meta.index[mask]
    sub = df.loc[ids].copy()
    sub["__subject__"] = meta.loc[ids, subject_col].values
    return sub.groupby("__subject__").mean()


def build_etec_analysis(abu: pd.DataFrame, meta: pd.DataFrame, taxa_order: list[str]) -> dict:
    abu = abu.loc[:, taxa_order].copy()
    clr = clr_transform(abu)
    meta = meta.loc[clr.index].copy()
    X, freq_df, keep_freq = fft_features(clr)
    delta_raw, meta2 = etec_baseline_deltas(X, meta)
    scale = robust_scale(delta_raw.loc[~meta2["phase"].eq("Baseline")])
    delta = delta_raw / scale.loc[delta_raw.columns]

    acute = mean_by_subject(delta, meta2, "SubjectID", meta2["phase"].eq("Acute"))
    recovery = mean_by_subject(delta, meta2, "SubjectID", meta2["phase"].eq("Recovery"))
    shared = acute.index.intersection(recovery.index)
    rec_minus_acute = recovery.loc[shared] - acute.loc[shared]

    u_acute = normalize_vec(acute.mean(axis=0).values)
    u_recovery = normalize_vec(rec_minus_acute.mean(axis=0).values)
    axis = normalize_vec(u_acute - u_recovery)  # acute direction plus reversed recovery direction
    axis_s = pd.Series(axis, index=delta.columns, name="ETEC_disruption_axis")

    scores = meta2.copy()
    scores["axis_score"] = score_delta(delta, axis_s)

    response_points = []
    for name, mat in {
        "ETEC acute−baseline": acute,
        "ETEC recovery−acute": rec_minus_acute,
    }.items():
        tmp = mat.copy()
        tmp["response"] = name
        tmp["subject"] = tmp.index
        response_points.append(tmp.reset_index(drop=True))
    rp = pd.concat(response_points, ignore_index=True)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pc = pca.fit_transform(rp[delta.columns].values)
    rp["PC1"], rp["PC2"] = pc[:, 0], pc[:, 1]

    response_cos = cosine(u_acute, u_recovery)

    weights = inverse_fft_axis(axis_s, scale, keep_freq, freq_df, len(taxa_order), taxa_order)
    pos, neg = extract_poles(weights, POLE_QUANTILE)
    pole_balance_df = pole_balance(clr, pos, neg).join(meta)
    subject_validation = validate_poles_etec(clr, meta, pos, neg)

    return {
        "name": "ETEC",
        "taxa_order": taxa_order,
        "clr": clr,
        "meta": meta,
        "delta": delta,
        "scale": scale,
        "freq_df": freq_df,
        "keep_freq": keep_freq,
        "axis": axis_s,
        "axis_score": scores,
        "response_points": rp,
        "pca_explained": pca.explained_variance_ratio_,
        "u_acute": pd.Series(u_acute, index=delta.columns),
        "u_recovery": pd.Series(u_recovery, index=delta.columns),
        "response_cosine": response_cos,
        "weights": weights,
        "pos_taxa": pos,
        "neg_taxa": neg,
        "pole_balance": pole_balance_df,
        "subject_validation": subject_validation,
    }


# =============================================================================
# 5. O'Keefe-specific analysis
# =============================================================================

def okeefe_he_deltas(X: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = meta[meta[OK_GROUP_COL].eq(OK_BASELINE_GROUP)]
    base_subjects = set(baseline[OK_SUBJECT_COL])
    rows, ids = [], []
    for sid in X.index:
        subj = meta.loc[sid, OK_SUBJECT_COL]
        if subj not in base_subjects:
            continue
        h_ids = baseline[baseline[OK_SUBJECT_COL].eq(subj)].index
        base = X.loc[h_ids].mean(axis=0).values
        rows.append(X.loc[sid].values - base)
        ids.append(sid)
    return pd.DataFrame(rows, index=ids, columns=X.columns), meta.loc[ids].copy()


def build_okeefe_analysis(abu: pd.DataFrame, meta: pd.DataFrame, taxa_order: list[str]) -> dict:
    # Main O'Keefe analysis uses only HE and DI.
    meta = meta[meta[OK_GROUP_COL].isin([OK_BASELINE_GROUP, OK_INTERVENTION_GROUP])].copy()
    abu = abu.loc[meta.index, taxa_order].copy()
    clr = clr_transform(abu)
    meta = meta.loc[clr.index].copy()
    X, freq_df, keep_freq = fft_features(clr)
    delta_raw, meta2 = okeefe_he_deltas(X, meta)
    scale = robust_scale(delta_raw.loc[~meta2[OK_GROUP_COL].eq(OK_BASELINE_GROUP)])
    delta = delta_raw / scale.loc[delta_raw.columns]

    response_mats = {}
    for nat in ["AFR", "AAM"]:
        mat = mean_by_subject(
            delta,
            meta2,
            OK_SUBJECT_COL,
            meta2[OK_NATIONALITY_COL].eq(nat) & meta2[OK_GROUP_COL].eq(OK_INTERVENTION_GROUP),
        )
        response_mats[nat] = mat

    u_afr = normalize_vec(response_mats["AFR"].mean(axis=0).values)
    u_aam = normalize_vec(response_mats["AAM"].mean(axis=0).values)
    axis = normalize_vec(u_afr - u_aam)  # westernization-like direction positive, restorative direction negative
    axis_s = pd.Series(axis, index=delta.columns, name="OKeefe_westernization_axis")

    scores = meta2.copy()
    scores["axis_score"] = score_delta(delta, axis_s)

    response_points = []
    for nat, mat in response_mats.items():
        tmp = mat.copy()
        tmp["response"] = f"{nat} DI−HE"
        tmp["subject"] = tmp.index
        response_points.append(tmp.reset_index(drop=True))
    rp = pd.concat(response_points, ignore_index=True)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pc = pca.fit_transform(rp[delta.columns].values)
    rp["PC1"], rp["PC2"] = pc[:, 0], pc[:, 1]

    response_cos = cosine(u_afr, u_aam)

    weights = inverse_fft_axis(axis_s, scale, keep_freq, freq_df, len(taxa_order), taxa_order)
    pos, neg = extract_poles(weights, POLE_QUANTILE)
    pole_balance_df = pole_balance(clr, pos, neg).join(meta)
    subject_validation = validate_poles_okeefe(clr, meta, pos, neg)

    return {
        "name": "OKeefe",
        "taxa_order": taxa_order,
        "clr": clr,
        "meta": meta,
        "delta": delta,
        "scale": scale,
        "freq_df": freq_df,
        "keep_freq": keep_freq,
        "axis": axis_s,
        "axis_score": scores,
        "response_points": rp,
        "pca_explained": pca.explained_variance_ratio_,
        "u_afr": pd.Series(u_afr, index=delta.columns),
        "u_aam": pd.Series(u_aam, index=delta.columns),
        "response_cosine": response_cos,
        "weights": weights,
        "pos_taxa": pos,
        "neg_taxa": neg,
        "pole_balance": pole_balance_df,
        "subject_validation": subject_validation,
    }


# =============================================================================
# 6. Inverse FFT, poles and validation
# =============================================================================

def inverse_fft_axis(axis: pd.Series,
                     scale: pd.Series,
                     keep_freq: pd.DataFrame,
                     freq_df: pd.DataFrame,
                     n_taxa: int,
                     taxa_order: list[str]) -> pd.DataFrame:
    raw_axis = axis / scale.loc[axis.index]
    n_keep = len(keep_freq)
    re_vals = raw_axis.iloc[:n_keep].values
    im_vals = raw_axis.iloc[n_keep:2*n_keep].values

    z = np.zeros(len(freq_df), dtype=complex)
    id_to_idx = dict(zip(freq_df["freq_id"], freq_df["freq_index"]))
    for fid, re_val, im_val in zip(keep_freq["freq_id"], re_vals, im_vals):
        z[id_to_idx[fid]] = re_val + 1j * im_val

    w = np.fft.irfft(z, n=n_taxa)
    w = (w - np.mean(w)) / (np.std(w) + 1e-12)
    return pd.DataFrame({
        "taxon_order": np.arange(1, n_taxa + 1),
        "taxon": taxa_order,
        "axis_weight": w,
    })


def extract_poles(weight_df: pd.DataFrame, q: float) -> tuple[list[str], list[str]]:
    hi = weight_df["axis_weight"].quantile(1 - q)
    lo = weight_df["axis_weight"].quantile(q)
    pos = weight_df.loc[weight_df["axis_weight"] >= hi, "taxon"].tolist()
    neg = weight_df.loc[weight_df["axis_weight"] <= lo, "taxon"].tolist()
    return pos, neg


def pole_balance(clr: pd.DataFrame, pos_taxa: list[str], neg_taxa: list[str]) -> pd.DataFrame:
    pos = [t for t in pos_taxa if t in clr.columns]
    neg = [t for t in neg_taxa if t in clr.columns]
    out = pd.DataFrame(index=clr.index)
    out["positive_pole_mean_clr"] = clr[pos].mean(axis=1)
    out["negative_pole_mean_clr"] = clr[neg].mean(axis=1)
    out["pole_balance"] = out["positive_pole_mean_clr"] - out["negative_pole_mean_clr"]
    return out


def validate_poles_etec(clr: pd.DataFrame, meta: pd.DataFrame, pos: list[str], neg: list[str]) -> pd.DataFrame:
    rows = []
    for subj, msub in meta.groupby("SubjectID"):
        b = msub[msub["Day"].eq(ETEC_BASELINE_DAY)].index
        a = msub[msub["phase"].eq("Acute")].index
        r = msub[msub["phase"].eq("Recovery")].index
        if len(b) and len(a):
            dp = clr.loc[a, pos].mean(axis=0).mean() - clr.loc[b, pos].mean(axis=0).mean()
            dn = clr.loc[a, neg].mean(axis=0).mean() - clr.loc[b, neg].mean(axis=0).mean()
            rows.append({"dataset": "ETEC", "response": "acute−baseline", "subject": subj,
                         "delta_positive": dp, "delta_negative": dn,
                         "delta_balance": dp - dn, "oriented_balance": dp - dn})
        if len(a) and len(r):
            dp = clr.loc[r, pos].mean(axis=0).mean() - clr.loc[a, pos].mean(axis=0).mean()
            dn = clr.loc[r, neg].mean(axis=0).mean() - clr.loc[a, neg].mean(axis=0).mean()
            # Recovery is expected to reverse the acute direction, so flip sign.
            rows.append({"dataset": "ETEC", "response": "recovery−acute", "subject": subj,
                         "delta_positive": dp, "delta_negative": dn,
                         "delta_balance": dp - dn, "oriented_balance": -(dp - dn)})
    return pd.DataFrame(rows)


def validate_poles_okeefe(clr: pd.DataFrame, meta: pd.DataFrame, pos: list[str], neg: list[str]) -> pd.DataFrame:
    rows = []
    for subj, msub in meta.groupby(OK_SUBJECT_COL):
        h = msub[msub[OK_GROUP_COL].eq(OK_BASELINE_GROUP)].index
        d = msub[msub[OK_GROUP_COL].eq(OK_INTERVENTION_GROUP)].index
        if not (len(h) and len(d)):
            continue
        nat = str(msub[OK_NATIONALITY_COL].iloc[0])
        dp = clr.loc[d, pos].mean(axis=0).mean() - clr.loc[h, pos].mean(axis=0).mean()
        dn = clr.loc[d, neg].mean(axis=0).mean() - clr.loc[h, neg].mean(axis=0).mean()
        # AFR DI−HE is expected to be westernization-like, positive direction.
        # AAM DI−HE is expected to be restorative-like, opposite direction.
        sign = +1 if nat == "AFR" else -1
        rows.append({"dataset": "OKeefe", "response": f"{nat} DI−HE", "subject": subj,
                     "nationality": nat, "delta_positive": dp, "delta_negative": dn,
                     "delta_balance": dp - dn, "oriented_balance": sign * (dp - dn)})
    return pd.DataFrame(rows)


def summarize_validation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for resp, sub in df.groupby("response"):
        rho, p = spearmanr(sub["delta_positive"], sub["delta_negative"]) if len(sub) >= 3 else (np.nan, np.nan)
        try:
            stat, p_greater = wilcoxon(sub["oriented_balance"], alternative="greater", zero_method="wilcox")
        except Exception:
            stat, p_greater = np.nan, np.nan
        rows.append({
            "response": resp,
            "n": len(sub),
            "median_delta_positive": np.nanmedian(sub["delta_positive"]),
            "median_delta_negative": np.nanmedian(sub["delta_negative"]),
            "median_oriented_balance": np.nanmedian(sub["oriented_balance"]),
            "wilcoxon_p_oriented_greater": p_greater,
            "spearman_pos_neg_rho": rho,
            "spearman_pos_neg_p": p,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 7. Spectral frequency loading profiles
# =============================================================================

def axis_frequency_profile(axis: pd.Series, scale: pd.Series, keep_freq: pd.DataFrame) -> pd.DataFrame:
    raw = axis / scale.loc[axis.index]
    n_keep = len(keep_freq)
    re_vals = raw.iloc[:n_keep].values
    im_vals = raw.iloc[n_keep:2*n_keep].values
    amp = np.sqrt(re_vals ** 2 + im_vals ** 2)
    amp = amp / (amp.sum() + 1e-12)
    return pd.DataFrame({
        "freq": keep_freq["freq"].values,
        "loading": amp,
    })


def band_energy(profile: pd.DataFrame) -> pd.DataFrame:
    bands = [
        ("low", 0.0, 0.10),
        ("mid", 0.10, 0.25),
        ("high", 0.25, FMAX + 1e-9),
    ]
    rows = []
    for name, lo, hi in bands:
        val = profile.loc[(profile["freq"] > lo) & (profile["freq"] <= hi), "loading"].sum()
        rows.append({"band": name, "energy": val})
    return pd.DataFrame(rows)


def profile_similarity(profile_a: pd.DataFrame, profile_b: pd.DataFrame) -> float:
    grid = np.linspace(0.001, min(FMAX, profile_a["freq"].max(), profile_b["freq"].max()), 200)
    ya = np.interp(grid, profile_a["freq"], profile_a["loading"])
    yb = np.interp(grid, profile_b["freq"], profile_b["loading"])
    return cosine(ya, yb)



# =============================================================================
# 8. Integrated publication figure
# =============================================================================

from matplotlib.patches import Rectangle

# Palette from the user's reference document.
# Warm colors encode perturbation / positive pole; cool teal encodes recovery / negative pole.
PALETTE = {
    "sand": "#B7B5A0",
    "teal": "#44757A",
    "plum": "#452A3D",
    "red": "#D44C3C",
    "coral": "#DD6C4C",
    "peach": "#E5855D",
    "cream": "#EEDFB7",
    "rose": "#B66065",
    "bluegrey": "#A8BCCC",
}

COL.update({
    "etec": PALETTE["red"],
    "recovery": PALETTE["teal"],
    "afr": PALETTE["rose"],
    "aam": PALETTE["teal"],
    "dark": PALETTE["plum"],
    "grey": PALETTE["sand"],
    "lightgrey": "#D6D2C4",
    "pos": PALETTE["coral"],
    "neg": PALETTE["teal"],
    "pos_soft": PALETTE["peach"],
    "neg_soft": PALETTE["bluegrey"],
    "cream": PALETTE["cream"],
    "sand": PALETTE["sand"],
    # direct aliases used by plotting functions
    "peach": PALETTE["peach"],
    "teal": PALETTE["teal"],
    "bluegrey": PALETTE["bluegrey"],
    # unified palette for c/d panels (different from panel a colors)
    "taxa_axis_line": "#4C3A48",
    "taxa_pos_fill": "#B987A5",
    "taxa_neg_fill": "#86A9A6",
    "taxa_pos_bg": "#E6D8E2",
    "taxa_neg_bg": "#D9E7E3",
    # shared taxa annotation colors on c/d
    "shared_dorea": "#A55D7A",
    "shared_phasco": "#C28C5B",
    "shared_faecal": "#4F8C84",
    # extra accent colors for e/f/g
    "balance_recovery": PALETTE["bluegrey"],
    "fg_etec_acute": "#D77A61",
    "fg_etec_recovery": "#6F95A0",
    "fg_afr": "#C989AB",
    "fg_aam": "#5E9F8B",
    "arrow_muted": "#7C6A75",
})

# Output folder is defined by OUT_DIR above.
OUT_DIR.mkdir(parents=True, exist_ok=True)


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "p=n.s."
    if p < 1e-4:
        return f"p={p:.1e}"
    if p < 1e-3:
        return f"p={p:.1e}"
    return f"p={p:.3g}"


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left", color=COL["dark"])


def style_axis(ax, grid: bool = True) -> None:
    despine(ax)
    ax.tick_params(axis="both", colors=COL["dark"], labelcolor=COL["dark"], width=0.65, length=3)
    ax.spines["left"].set_color(COL["dark"])
    ax.spines["bottom"].set_color(COL["dark"])
    if grid:
        ax.grid(axis="y", color="#EEECE5", lw=0.55, zorder=0)
    ax.set_axisbelow(True)


def plot_etec_trajectory_publication(ax, etec: dict) -> None:
    sc = etec["axis_score"].copy()
    # biological phase shading
    ax.axvspan(1, 9, color=COL["cream"], alpha=0.35, lw=0, zorder=0)
    ax.axvspan(28, 84, color=COL["neg_soft"], alpha=0.12, lw=0, zorder=0)
    ax.axvline(0, color=COL["dark"], lw=0.75, ls=":", alpha=0.65, zorder=1)

    # individual trajectories
    for subj, sub in sc.sort_values("Day").groupby("SubjectID"):
        ax.plot(sub["Day"], sub["axis_score"], color=COL["sand"], lw=0.75,
                alpha=0.45, zorder=1)

    # summary trajectory
    summ = sc.groupby("Day")["axis_score"].agg(
        median="median",
        q1=lambda x: np.quantile(x, 0.25),
        q3=lambda x: np.quantile(x, 0.75),
    ).reset_index().sort_values("Day")
    ax.fill_between(summ["Day"], summ["q1"], summ["q3"], color=COL["peach"],
                    alpha=0.30, lw=0, zorder=2)
    ax.plot(summ["Day"], summ["median"], color=COL["etec"], lw=2.6, zorder=4)
    ax.scatter(summ["Day"], summ["median"], s=12, color=COL["etec"],
               edgecolor="white", lw=0.35, zorder=5)

    ax.axhline(0, color=COL["dark"], lw=0.75, ls="--", alpha=0.65)
    ylim = ax.get_ylim()
    ax.text(5, ylim[1] - 0.08*(ylim[1]-ylim[0]), "acute\nphase", ha="center", va="top",
            fontsize=6, color=COL["dark"])
    ax.text(56, ylim[1] - 0.08*(ylim[1]-ylim[0]), "late\nrecovery", ha="center", va="top",
            fontsize=6, color=COL["teal"])

    ax.set_xlabel("Day relative to ETEC challenge")
    ax.set_ylabel("ETEC spectral-axis score")
    ax.set_title("ETEC perturbation follows a full-genus spectral axis", pad=4, color=COL["dark"])
    style_axis(ax)


def plot_okeefe_diet_response_publication(ax, ok: dict) -> None:
    sc = ok["axis_score"].copy()
    group_info = [("AFR", COL["afr"], -0.08, "westernization-like"),
                  ("AAM", COL["aam"],  0.08, "fiber-rich / restorative")]
    for nat, color, offset, label_text in group_info:
        sub = sc[sc[OK_NATIONALITY_COL].eq(nat)]
        summ = sub.groupby([OK_SUBJECT_COL, OK_GROUP_COL])["axis_score"].mean().reset_index()
        wide = summ.pivot(index=OK_SUBJECT_COL, columns=OK_GROUP_COL, values="axis_score")
        x = np.array([0, 1]) + offset
        for _, row in wide.iterrows():
            if OK_BASELINE_GROUP in row.index and OK_INTERVENTION_GROUP in row.index:
                ax.plot(x, [row[OK_BASELINE_GROUP], row[OK_INTERVENTION_GROUP]],
                        color=color, alpha=0.25, lw=0.8, zorder=1)
                ax.scatter(x, [row[OK_BASELINE_GROUP], row[OK_INTERVENTION_GROUP]],
                           color=color, alpha=0.18, s=8, edgecolor="none", zorder=1)
        meds = np.array([wide[OK_BASELINE_GROUP].median(), wide[OK_INTERVENTION_GROUP].median()])
        q1 = np.array([wide[OK_BASELINE_GROUP].quantile(0.25), wide[OK_INTERVENTION_GROUP].quantile(0.25)])
        q3 = np.array([wide[OK_BASELINE_GROUP].quantile(0.75), wide[OK_INTERVENTION_GROUP].quantile(0.75)])
        ax.fill_between(x, q1, q3, color=color, alpha=0.12, lw=0, zorder=2)
        ax.plot(x, meds, color=color, lw=2.7, label=nat, zorder=4)
        ax.scatter(x, meds, color=color, s=34, edgecolor="white", linewidth=0.5, zorder=5)

    ax.axhline(0, color=COL["dark"], lw=0.75, ls="--", alpha=0.65)
    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([OK_BASELINE_GROUP, OK_INTERVENTION_GROUP])
    ax.set_ylabel("O'Keefe diet-axis score")
    ax.set_title("Reciprocal diet swap moves in opposite spectral directions", pad=4, color=COL["dark"])
    ax.legend(frameon=False, loc="upper left", handlelength=1.7, borderaxespad=0.2)

    # directional labels
    y0, y1 = ax.get_ylim()
    try:
        afr_wide = sc[sc[OK_NATIONALITY_COL].eq("AFR")].groupby([OK_SUBJECT_COL, OK_GROUP_COL])["axis_score"].mean().reset_index().pivot(index=OK_SUBJECT_COL, columns=OK_GROUP_COL, values="axis_score")
        aam_wide = sc[sc[OK_NATIONALITY_COL].eq("AAM")].groupby([OK_SUBJECT_COL, OK_GROUP_COL])["axis_score"].mean().reset_index().pivot(index=OK_SUBJECT_COL, columns=OK_GROUP_COL, values="axis_score")
        ax.annotate("AFR: westernization-like", xy=(1.02, afr_wide[OK_INTERVENTION_GROUP].median()),
                    xytext=(0.58, y1 - 0.12*(y1-y0)), textcoords="data", fontsize=6,
                    color=COL["afr"], arrowprops=dict(arrowstyle="-", lw=0.6, color=COL["afr"], alpha=0.8))
        ax.annotate("AAM: fiber-rich", xy=(1.08, aam_wide[OK_INTERVENTION_GROUP].median()),
                    xytext=(0.60, y0 + 0.13*(y1-y0)), textcoords="data", fontsize=6,
                    color=COL["aam"], arrowprops=dict(arrowstyle="-", lw=0.6, color=COL["aam"], alpha=0.8))
    except Exception:
        pass
    style_axis(ax)


def annotate_taxon(ax, w: pd.DataFrame, taxon: str, color: str, direction: str = "auto", dx: float = 2.0, dy_scale: float = 0.15) -> None:
    hit = w[w["taxon"].eq(taxon)]
    if hit.empty:
        return
    x = float(hit["taxon_order"].iloc[0])
    y = float(hit["axis_weight"].iloc[0])
    ax.scatter([x], [y], s=34, color=color, edgecolor="white", lw=0.55, zorder=6)
    ylim = ax.get_ylim()
    yr = ylim[1] - ylim[0]
    if direction == "positive" or (direction == "auto" and y >= 0):
        xytext = (x + dx, y + dy_scale * yr)
        va = "bottom"
    else:
        xytext = (x + dx, y - dy_scale * yr)
        va = "top"
    ax.annotate(taxon, xy=(x, y), xytext=xytext, fontsize=6.1,
                color=COL["dark"], ha="left", va=va,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78),
                arrowprops=dict(arrowstyle="-", color=color, lw=0.65, alpha=0.9),
                zorder=7)


def plot_axis_taxa_publication(ax, res: dict, title: str, xlabel: str, dataset: str = "etec") -> None:
    """Project a dataset-specific spectral axis back to the taxon-order axis.

    Panels c and d intentionally use the same color system so that the two
    datasets read as parallel analyses. These colors are independent from
    the orange/teal colors used in panels a and b.
    """
    w = res["weights"].copy()

    line_c = COL["taxa_axis_line"]
    pos_c = COL["taxa_pos_fill"]
    neg_c = COL["taxa_neg_fill"]
    bg_pos = COL["taxa_pos_bg"]
    bg_neg = COL["taxa_neg_bg"]

    ax.plot(w["taxon_order"], w["axis_weight"], color=line_c, lw=1.4, zorder=3)
    hi = w["axis_weight"].quantile(1 - POLE_QUANTILE)
    lo = w["axis_weight"].quantile(POLE_QUANTILE)

    ax.axhspan(hi, w["axis_weight"].max()*1.05, color=bg_pos, alpha=0.26, lw=0, zorder=0)
    ax.axhspan(w["axis_weight"].min()*1.05, lo, color=bg_neg, alpha=0.24, lw=0, zorder=0)
    ax.axhline(hi, color=pos_c, lw=0.85, ls=":", alpha=0.95)
    ax.axhline(lo, color=neg_c, lw=0.85, ls=":", alpha=0.95)

    ax.fill_between(
        w["taxon_order"], 0, w["axis_weight"],
        where=w["axis_weight"] >= hi,
        color=pos_c, alpha=0.78, lw=0, zorder=2
    )
    ax.fill_between(
        w["taxon_order"], 0, w["axis_weight"],
        where=w["axis_weight"] <= lo,
        color=neg_c, alpha=0.80, lw=0, zorder=2
    )
    ax.axhline(0, color=COL["dark"], lw=0.8, alpha=0.7)

    # Shared taxa highlighted in both datasets.
    if dataset.lower().startswith("e"):
        annotate_taxon(ax, w, "Dorea", COL["shared_dorea"], direction="positive", dx=1.6, dy_scale=0.14)
        annotate_taxon(ax, w, "Phascolarctobacterium", COL["shared_phasco"], direction="positive", dx=2.0, dy_scale=0.16)
        annotate_taxon(ax, w, "Faecalibacterium", COL["shared_faecal"], direction="negative", dx=2.2, dy_scale=0.12)
    else:
        annotate_taxon(ax, w, "Dorea", COL["shared_dorea"], direction="positive", dx=1.8, dy_scale=0.14)
        annotate_taxon(ax, w, "Phascolarctobacterium", COL["shared_phasco"], direction="positive", dx=2.0, dy_scale=0.12)
        annotate_taxon(ax, w, "Faecalibacterium", COL["shared_faecal"], direction="negative", dx=2.4, dy_scale=0.14)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Inverse-FFT axis weight")
    ax.set_title(title, pad=5, color=COL["dark"])
    style_axis(ax, grid=False)


def oriented_pole_pvalue(values: pd.Series) -> float:
    try:
        _, p = wilcoxon(values, alternative="greater", zero_method="wilcox")
        return float(p)
    except Exception:
        return np.nan


def plot_oriented_balance_publication(ax, etec: dict, ok: dict) -> None:
    df = pd.concat([etec["subject_validation"], ok["subject_validation"]], ignore_index=True)
    order = ["acute−baseline", "recovery−acute", "AFR DI−HE", "AAM DI−HE"]
    labels = ["ETEC\nacute", "ETEC\nrecovery", "AFR\nDI−HE", "AAM\nDI−HE"]
    colors = [COL["etec"], COL["balance_recovery"], COL["afr"], "#6FAE9D"]

    y_all = df["oriented_balance"].dropna().values
    ymin, ymax = np.nanmin(y_all), np.nanmax(y_all)
    pad = 0.12 * (ymax - ymin + 1e-9)
    ax.set_ylim(ymin - pad, ymax + pad)

    for i, resp in enumerate(order):
        sub = df[df["response"].eq(resp)].copy()
        if sub.empty:
            continue
        x = np.full(len(sub), i) + rng.normal(0, 0.045, len(sub))
        ax.scatter(x, sub["oriented_balance"], s=22, color=colors[i], alpha=0.65,
                   edgecolor="white", linewidth=0.35, zorder=3)
        med = np.nanmedian(sub["oriented_balance"])
        q1 = np.nanquantile(sub["oriented_balance"], 0.25)
        q3 = np.nanquantile(sub["oriented_balance"], 0.75)
        ax.plot([i - 0.20, i + 0.20], [med, med], color=colors[i], lw=2.4, solid_capstyle="round", zorder=5)
        ax.plot([i, i], [q1, q3], color=colors[i], lw=1.5, solid_capstyle="round", zorder=4)
        p = oriented_pole_pvalue(sub["oriented_balance"])
        ax.text(i, ax.get_ylim()[0] + 0.035*(ax.get_ylim()[1]-ax.get_ylim()[0]), format_p(p),
                ha="center", va="bottom", fontsize=5.8, color=COL["dark"])

    ax.axhline(0, color=COL["dark"], lw=0.75, ls="--", alpha=0.65)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Oriented pole-balance response")
    ax.set_title("Pole balance shifts in expected directions", pad=4, color=COL["dark"])
    style_axis(ax)


def add_reciprocal_background(ax) -> None:
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax.add_patch(Rectangle((0, y0), x1 - 0, 0 - y0, facecolor=COL["cream"], alpha=0.17, lw=0, zorder=0))
    ax.add_patch(Rectangle((x0, 0), 0 - x0, y1 - 0, facecolor=COL["cream"], alpha=0.17, lw=0, zorder=0))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)


def plot_pos_vs_neg_publication(ax, df: pd.DataFrame, title: str, colors: dict) -> None:
    for resp, sub in df.groupby("response"):
        color = colors.get(resp, COL["grey"])
        ax.scatter(sub["delta_positive"], sub["delta_negative"], s=36,
                   color=color, alpha=0.78, edgecolor="white", linewidth=0.38,
                   label=resp, zorder=3)

    allx = df["delta_positive"].values.astype(float)
    ally = df["delta_negative"].values.astype(float)
    rho, p = spearmanr(allx, ally) if len(df) >= 3 else (np.nan, np.nan)
    if np.isfinite(allx).sum() > 2:
        slope, intercept = np.polyfit(allx, ally, 1)
        xs = np.linspace(np.nanmin(allx), np.nanmax(allx), 100)
        ax.plot(xs, slope * xs + intercept, color=COL["arrow_muted"], lw=1.65, alpha=0.95, zorder=2)

    xr = np.nanmax(allx) - np.nanmin(allx)
    yr = np.nanmax(ally) - np.nanmin(ally)
    ax.set_xlim(np.nanmin(allx) - 0.10*xr - 1e-9, np.nanmax(allx) + 0.10*xr + 1e-9)
    ax.set_ylim(np.nanmin(ally) - 0.10*yr - 1e-9, np.nanmax(ally) + 0.10*yr + 1e-9)
    add_reciprocal_background(ax)
    ax.axhline(0, color="#E1DED3", lw=0.85, zorder=1)
    ax.axvline(0, color="#E1DED3", lw=0.85, zorder=1)

    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax.annotate("reciprocal\nshift", xy=(x1 - 0.12*(x1-x0), y0 + 0.18*(y1-y0)),
                xytext=(x0 + 0.12*(x1-x0), y1 - 0.20*(y1-y0)),
                fontsize=5.8, color=COL["dark"], ha="center", va="center",
                arrowprops=dict(arrowstyle="->", lw=1.0, color=COL["arrow_muted"], alpha=0.8),
                zorder=5)

    ax.set_xlabel("Δ positive pole")
    ax.set_ylabel("Δ negative pole")
    ax.set_title(f"{title}\nSpearman ρ={rho:.2f}, {format_p(p)}", pad=4, color=COL["dark"])
    ax.legend(frameon=False, loc="best", handletextpad=0.25, labelspacing=0.25)
    style_axis(ax, grid=False)


def create_integrated_publication_figure(etec: dict, ok: dict) -> None:
    fig = plt.figure(figsize=(9.4, 9.8))
    gs = GridSpec(3, 6, figure=fig, hspace=0.70, wspace=0.62,
                  height_ratios=[1.03, 1.03, 1.08])

    ax = fig.add_subplot(gs[0, 0:3]); panel_label(ax, "a")
    plot_etec_trajectory_publication(ax, etec)

    ax = fig.add_subplot(gs[0, 3:6]); panel_label(ax, "b")
    plot_okeefe_diet_response_publication(ax, ok)

    ax = fig.add_subplot(gs[1, 0:3]); panel_label(ax, "c")
    plot_axis_taxa_publication(ax, etec, "ETEC spectral axis projected to taxa", "ETEC phylogenetic genus order", dataset="etec")

    ax = fig.add_subplot(gs[1, 3:6]); panel_label(ax, "d")
    plot_axis_taxa_publication(ax, ok, "O'Keefe spectral axis projected to taxa", "O'Keefe phylogenetic genus order", dataset="okeefe")

    ax = fig.add_subplot(gs[2, 0:2]); panel_label(ax, "e")
    plot_oriented_balance_publication(ax, etec, ok)

    ax = fig.add_subplot(gs[2, 2:4]); panel_label(ax, "f")
    plot_pos_vs_neg_publication(
        ax,
        etec["subject_validation"],
        "ETEC pole responses",
        {"acute−baseline": COL["fg_etec_acute"], "recovery−acute": COL["fg_etec_recovery"]},
    )

    ax = fig.add_subplot(gs[2, 4:6]); panel_label(ax, "g")
    plot_pos_vs_neg_publication(
        ax,
        ok["subject_validation"],
        "O'Keefe pole responses",
        {"AFR DI−HE": COL["fg_afr"], "AAM DI−HE": COL["fg_aam"]},
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_spectral_seesaw_integrated_nature.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure_spectral_seesaw_integrated_nature.png", bbox_inches="tight")
    plt.close(fig)


def maybe_save_tables(etec: dict, ok: dict) -> None:
    if not SAVE_TABLES:
        return
    etec["axis_score"].reset_index().rename(columns={"index": "sample"}).to_csv(OUT_DIR / "ETEC_axis_sample_scores.csv", index=False)
    ok["axis_score"].reset_index().rename(columns={"index": "sample"}).to_csv(OUT_DIR / "OKeefe_axis_sample_scores.csv", index=False)
    etec["weights"].to_csv(OUT_DIR / "ETEC_axis_taxon_weights.csv", index=False)
    ok["weights"].to_csv(OUT_DIR / "OKeefe_axis_taxon_weights.csv", index=False)
    etec["subject_validation"].to_csv(OUT_DIR / "ETEC_pole_validation_subject_level.csv", index=False)
    ok["subject_validation"].to_csv(OUT_DIR / "OKeefe_pole_validation_subject_level.csv", index=False)
    summarize_validation(etec["subject_validation"]).to_csv(OUT_DIR / "ETEC_pole_validation_summary.csv", index=False)
    summarize_validation(ok["subject_validation"]).to_csv(OUT_DIR / "OKeefe_pole_validation_summary.csv", index=False)



# =============================================================================
# 9. Control-axis specificity analysis
# =============================================================================

# Number of non-target axes to draw beside the main perturbation/recovery axis.
# These axes are residual PCs after removing the main axis, so they represent
# real data variation that is orthogonal to the target biological direction.
CONTROL_N_AXES = 4


def orthogonalize_to_axis(vec: np.ndarray, ref_axis: np.ndarray) -> np.ndarray:
    """Remove the component of vec that lies along ref_axis, then normalize."""
    v = np.asarray(vec, dtype=float)
    r = normalize_vec(np.asarray(ref_axis, dtype=float))
    v = v - np.dot(v, r) * r
    return normalize_vec(v)


def build_residual_pc_control_axes(res: dict,
                                   n_axes: int = CONTROL_N_AXES,
                                   prefix: str = "residual PC") -> list[dict]:
    """Build control axes from residual spectral variation after removing target axis.

    The first returned axis is always the original target axis. The remaining
    axes are principal components of the residual matrix:

        X_residual = X - (X · v_target) v_target

    Therefore, these control axes are orthogonal to the main perturbation axis,
    but still represent dominant directions of real variation in the dataset.
    """
    target = res["axis"].copy()
    v = normalize_vec(target.values)
    X = res["delta"].loc[:, target.index].values.astype(float)

    # Remove the target-axis component from every sample.
    X_resid = X - np.outer(X @ v, v)

    n_possible = min(n_axes, X_resid.shape[0] - 1, X_resid.shape[1])
    axes = [{
        "label": "target perturbation axis",
        "short_label": "Target",
        "axis": target,
        "kind": "target",
        "explained": np.nan,
        "cosine_to_target": 1.0,
    }]
    if n_possible < 1:
        return axes

    pca = PCA(n_components=n_possible, random_state=RANDOM_SEED)
    pca.fit(X_resid)

    for i, comp in enumerate(pca.components_, start=1):
        comp_orth = orthogonalize_to_axis(comp, v)
        axis_s = pd.Series(comp_orth, index=target.index, name=f"{prefix}{i}")
        axes.append({
            "label": f"{prefix}{i} orthogonal to target",
            "short_label": f"Residual PC{i}",
            "axis": axis_s,
            "kind": "control",
            "explained": float(pca.explained_variance_ratio_[i - 1]),
            "cosine_to_target": cosine(axis_s.values, target.values),
        })
    return axes


def score_samples_on_axis(res: dict, axis: pd.Series, score_name: str = "axis_score") -> pd.DataFrame:
    """Project all baseline-centered spectral displacements onto one axis."""
    meta = res["axis_score"].drop(columns=["axis_score"], errors="ignore").copy()
    common = meta.index.intersection(res["delta"].index)
    meta = meta.loc[common].copy()
    axis = axis.loc[res["delta"].columns]
    meta[score_name] = score_delta(res["delta"].loc[common, axis.index], axis)
    return meta


def summarize_etec_axis_shape(etec: dict, axis_record: dict) -> dict:
    """Quantify whether an ETEC axis shows acute displacement followed by recovery."""
    sc = score_samples_on_axis(etec, axis_record["axis"])
    baseline = sc.loc[sc["phase"].eq("Baseline"), "axis_score"].median()
    acute = sc.loc[sc["phase"].eq("Acute"), "axis_score"].median()
    recovery = sc.loc[sc["phase"].eq("Recovery"), "axis_score"].median()
    acute_shift = acute - baseline
    recovery_offset = recovery - baseline
    return_fraction = (acute - recovery) / (acute_shift + 1e-12)
    shape_index = (abs(acute_shift) - abs(recovery_offset)) / (abs(acute_shift) + 1e-12)
    return {
        "dataset": "ETEC",
        "axis": axis_record["short_label"],
        "kind": axis_record["kind"],
        "residual_variance_explained": axis_record["explained"],
        "cosine_to_target": axis_record["cosine_to_target"],
        "baseline_median": baseline,
        "acute_median": acute,
        "recovery_median": recovery,
        "acute_shift": acute_shift,
        "recovery_offset_from_baseline": recovery_offset,
        "return_fraction": return_fraction,
        "shape_index": shape_index,
    }


def summarize_okeefe_axis_shape(ok: dict, axis_record: dict) -> dict:
    """Quantify whether an O'Keefe axis shows reciprocal AFR and AAM DI responses."""
    sc = score_samples_on_axis(ok, axis_record["axis"])
    rows = []
    out = {
        "dataset": "OKeefe",
        "axis": axis_record["short_label"],
        "kind": axis_record["kind"],
        "residual_variance_explained": axis_record["explained"],
        "cosine_to_target": axis_record["cosine_to_target"],
    }
    for nat in ["AFR", "AAM"]:
        sub = sc[sc[OK_NATIONALITY_COL].eq(nat)]
        summ = sub.groupby([OK_SUBJECT_COL, OK_GROUP_COL])["axis_score"].mean().reset_index()
        wide = summ.pivot(index=OK_SUBJECT_COL, columns=OK_GROUP_COL, values="axis_score")
        if OK_BASELINE_GROUP in wide.columns and OK_INTERVENTION_GROUP in wide.columns:
            shift = (wide[OK_INTERVENTION_GROUP] - wide[OK_BASELINE_GROUP]).median()
            out[f"{nat}_HE_median"] = wide[OK_BASELINE_GROUP].median()
            out[f"{nat}_DI_median"] = wide[OK_INTERVENTION_GROUP].median()
            out[f"{nat}_DI_minus_HE"] = shift
        else:
            out[f"{nat}_HE_median"] = np.nan
            out[f"{nat}_DI_median"] = np.nan
            out[f"{nat}_DI_minus_HE"] = np.nan
    afr_shift = out.get("AFR_DI_minus_HE", np.nan)
    aam_shift = out.get("AAM_DI_minus_HE", np.nan)
    out["reciprocal_strength"] = afr_shift - aam_shift
    out["opposition_index"] = -afr_shift * aam_shift
    out["opposite_sign"] = bool(np.sign(afr_shift) != np.sign(aam_shift)) if np.isfinite(afr_shift) and np.isfinite(aam_shift) else False
    return out


def plot_etec_axis_score(ax, etec: dict, axis_record: dict, row_index: int) -> None:
    """Plot ETEC trajectory along target or control axis."""
    sc = score_samples_on_axis(etec, axis_record["axis"])
    is_target = axis_record["kind"] == "target"
    main_color = COL["etec"] if is_target else "#7A7180"
    fill_color = COL["peach"] if is_target else "#C8C1C8"
    line_alpha = 0.42 if is_target else 0.28

    ax.axvspan(1, 9, color=COL["cream"], alpha=0.30 if is_target else 0.16, lw=0, zorder=0)
    ax.axvspan(28, 84, color=COL["neg_soft"], alpha=0.10 if is_target else 0.06, lw=0, zorder=0)
    ax.axvline(0, color=COL["dark"], lw=0.65, ls=":", alpha=0.55, zorder=1)

    for _, sub in sc.sort_values("Day").groupby("SubjectID"):
        ax.plot(sub["Day"], sub["axis_score"], color=COL["sand"], lw=0.65,
                alpha=line_alpha, zorder=1)

    summ = sc.groupby("Day")["axis_score"].agg(
        median="median",
        q1=lambda x: np.quantile(x, 0.25),
        q3=lambda x: np.quantile(x, 0.75),
    ).reset_index().sort_values("Day")
    ax.fill_between(summ["Day"], summ["q1"], summ["q3"], color=fill_color,
                    alpha=0.28 if is_target else 0.22, lw=0, zorder=2)
    ax.plot(summ["Day"], summ["median"], color=main_color, lw=2.3 if is_target else 1.8, zorder=4)
    ax.scatter(summ["Day"], summ["median"], s=12 if is_target else 9,
               color=main_color, edgecolor="white", lw=0.35, zorder=5)
    ax.axhline(0, color=COL["dark"], lw=0.65, ls="--", alpha=0.55)

    title = axis_record["short_label"] if is_target else f"{axis_record['short_label']} ({100*axis_record['explained']:.1f}% residual var.)"
    ax.set_title(title, pad=3, color=COL["dark"], fontsize=7.2)
    if row_index == 0:
        ax.set_title("ETEC: target axis vs orthogonal residual axes\n" + title, pad=3, color=COL["dark"], fontsize=7.2)
    ax.set_xlabel("Day relative to ETEC challenge")
    ax.set_ylabel("Projected score")
    style_axis(ax)


def plot_okeefe_axis_score(ax, ok: dict, axis_record: dict, row_index: int) -> None:
    """Plot O'Keefe HE-to-DI response along target or control axis."""
    sc = score_samples_on_axis(ok, axis_record["axis"])
    is_target = axis_record["kind"] == "target"
    group_info = [
        ("AFR", COL["afr"] if is_target else "#9E8C9A", -0.08),
        ("AAM", COL["aam"] if is_target else "#8DA39D", 0.08),
    ]
    for nat, color, offset in group_info:
        sub = sc[sc[OK_NATIONALITY_COL].eq(nat)]
        summ = sub.groupby([OK_SUBJECT_COL, OK_GROUP_COL])["axis_score"].mean().reset_index()
        wide = summ.pivot(index=OK_SUBJECT_COL, columns=OK_GROUP_COL, values="axis_score")
        if OK_BASELINE_GROUP not in wide.columns or OK_INTERVENTION_GROUP not in wide.columns:
            continue
        x = np.array([0, 1]) + offset
        for _, row in wide.iterrows():
            ax.plot(x, [row[OK_BASELINE_GROUP], row[OK_INTERVENTION_GROUP]],
                    color=color, alpha=0.25 if is_target else 0.18, lw=0.75, zorder=1)
            ax.scatter(x, [row[OK_BASELINE_GROUP], row[OK_INTERVENTION_GROUP]],
                       color=color, alpha=0.18, s=8, edgecolor="none", zorder=1)
        meds = np.array([wide[OK_BASELINE_GROUP].median(), wide[OK_INTERVENTION_GROUP].median()])
        q1 = np.array([wide[OK_BASELINE_GROUP].quantile(0.25), wide[OK_INTERVENTION_GROUP].quantile(0.25)])
        q3 = np.array([wide[OK_BASELINE_GROUP].quantile(0.75), wide[OK_INTERVENTION_GROUP].quantile(0.75)])
        ax.fill_between(x, q1, q3, color=color, alpha=0.12 if is_target else 0.08, lw=0, zorder=2)
        ax.plot(x, meds, color=color, lw=2.3 if is_target else 1.8, label=nat, zorder=4)
        ax.scatter(x, meds, color=color, s=30 if is_target else 22,
                   edgecolor="white", linewidth=0.45, zorder=5)

    ax.axhline(0, color=COL["dark"], lw=0.65, ls="--", alpha=0.55)
    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([OK_BASELINE_GROUP, OK_INTERVENTION_GROUP])
    ax.set_ylabel("Projected score")
    title = axis_record["short_label"] if is_target else f"{axis_record['short_label']} ({100*axis_record['explained']:.1f}% residual var.)"
    ax.set_title(title, pad=3, color=COL["dark"], fontsize=7.2)
    if row_index == 0:
        ax.set_title("O'Keefe: target axis vs orthogonal residual axes\n" + title, pad=3, color=COL["dark"], fontsize=7.2)
    if row_index == 0:
        ax.legend(frameon=False, loc="best", handlelength=1.6)
    style_axis(ax)


def create_control_axis_specificity_figure(etec: dict,
                                            ok: dict,
                                            n_control_axes: int = CONTROL_N_AXES) -> pd.DataFrame:
    """Create a figure showing that not every axis recapitulates panels a and b."""
    etec_axes = build_residual_pc_control_axes(etec, n_axes=n_control_axes, prefix="residual PC")
    ok_axes = build_residual_pc_control_axes(ok, n_axes=n_control_axes, prefix="residual PC")
    n_rows = max(len(etec_axes), len(ok_axes))

    fig = plt.figure(figsize=(8.7, 1.65 * n_rows + 0.9))
    gs = GridSpec(n_rows, 2, figure=fig, hspace=0.72, wspace=0.42)

    for i in range(n_rows):
        if i < len(etec_axes):
            ax = fig.add_subplot(gs[i, 0])
            if i == 0:
                panel_label(ax, "a")
            plot_etec_axis_score(ax, etec, etec_axes[i], row_index=i)
        if i < len(ok_axes):
            ax = fig.add_subplot(gs[i, 1])
            if i == 0:
                panel_label(ax, "b")
            plot_okeefe_axis_score(ax, ok, ok_axes[i], row_index=i)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_control_axes_specificity.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure_control_axes_specificity.png", bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for rec in etec_axes:
        summary_rows.append(summarize_etec_axis_shape(etec, rec))
    for rec in ok_axes:
        summary_rows.append(summarize_okeefe_axis_shape(ok, rec))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "control_axes_specificity_summary.csv", index=False)
    return summary


# =============================================================================
# 9. Main
# =============================================================================

def main() -> None:
    etec_abu_path = resolve_path(ETEC_ABU_PATH, "etec16s_abundance_genus(1).csv")
    etec_meta_path = resolve_path(ETEC_META_PATH, "etec16s_metadata(1).csv")
    ok_abu_path = resolve_path(OK_ABU_PATH, "OKeefe_dietswap_abundance(1).csv")
    ok_meta_path = resolve_path(OK_META_PATH, "OKeefe_dietswap_metadata(1).csv")

    print("Loading data...")
    etec_abu = filter_taxa(read_abundance(etec_abu_path))
    etec_meta = read_etec_metadata(etec_meta_path)
    etec_abu, etec_meta = align_samples(etec_abu, etec_meta)

    ok_abu = filter_taxa(read_abundance(ok_abu_path))
    ok_meta = read_okeefe_metadata(ok_meta_path)
    ok_abu, ok_meta = align_samples(ok_abu, ok_meta)

    phy_order = read_phylogenetic_order(PHYLO_PATH)
    etec_taxa = ordered_taxa_for_dataset(etec_abu, phy_order, "ETEC")
    ok_taxa = ordered_taxa_for_dataset(ok_abu, phy_order, "O'Keefe")

    print(f"ETEC taxa used: {len(etec_taxa)}")
    print(f"O'Keefe taxa used: {len(ok_taxa)}")

    print("Building ETEC full-genus spectral axis...")
    etec = build_etec_analysis(etec_abu, etec_meta, etec_taxa)

    print("Building O'Keefe diet-specific spectral axis...")
    ok = build_okeefe_analysis(ok_abu, ok_meta, ok_taxa)

    print("Drawing integrated figure...")
    create_integrated_publication_figure(etec, ok)

    print("Drawing target-vs-control axis specificity figure...")
    control_summary = create_control_axis_specificity_figure(etec, ok, n_control_axes=CONTROL_N_AXES)

    maybe_save_tables(etec, ok)

    print("\nDone.")
    print(f"Output directory: {OUT_DIR}")
    print("Figure:")
    print(" - figure_spectral_seesaw_integrated_nature.pdf")
    print(" - figure_spectral_seesaw_integrated_nature.png")
    print(" - figure_control_axes_specificity.pdf")
    print(" - figure_control_axes_specificity.png")
    print(" - control_axes_specificity_summary.csv")

    print("\nKey diagnostics:")
    print(f"ETEC acute vs recovery cosine: {etec['response_cosine']:.3f}")
    print(f"O'Keefe AFR diet vs AAM diet cosine: {ok['response_cosine']:.3f}")

    print("\nETEC pole validation summary:")
    print(summarize_validation(etec["subject_validation"]).to_string(index=False))
    print("\nO'Keefe pole validation summary:")
    print(summarize_validation(ok["subject_validation"]).to_string(index=False))

    print("\nTop ETEC positive pole taxa:")
    print(", ".join(etec["weights"].sort_values("axis_weight", ascending=False).head(10)["taxon"].tolist()))
    print("Top ETEC negative pole taxa:")
    print(", ".join(etec["weights"].sort_values("axis_weight", ascending=True).head(10)["taxon"].tolist()))

    print("\nTop O'Keefe positive pole taxa:")
    print(", ".join(ok["weights"].sort_values("axis_weight", ascending=False).head(10)["taxon"].tolist()))
    print("Top O'Keefe negative pole taxa:")
    print(", ".join(ok["weights"].sort_values("axis_weight", ascending=True).head(10)["taxon"].tolist()))


if __name__ == "__main__":
    main()
