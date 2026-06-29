#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMP body-site spectral phase attractor law with taxonomic back-projection
=======================================================================

Main biological question
------------------------
Do body sites act as low-order spectral phase attractors, and can the
phase-shifted spectral waves be traced back to concrete genera / higher taxa?

This script is designed for your HMP V35 genus table:
    abundance.csv: rows = genera, columns = samples
    metadata.csv : sample metadata with SITE, RSID, VISITNO

Core outputs
------------
1) Low-vs-high phase locking:
   - Rayleigh circular-uniformity tests by site and mode
   - PPC/PLV by site and mode
   - band-level comparison: low modes should be more phase-locked than high modes

2) Phase-attractor modes:
   - score = within-site PPC × between-site circular separation
   - label-permutation null

3) Taxonomic back-projection:
   - reconstruct body-site-specific locked waves from mean complex FFT coefficients
   - identify genera and clades anchoring each site's phase position
   - identify pairwise body-site phase-shift taxa/clades
   - optional phylum/class/order/family-level summaries if a taxonomy table is provided

4) Longitudinal attractor:
   - same subject + same site phase distance versus other controls

Important note
--------------
If TAXONOMY_PATH is unavailable, the script still runs and reports genus-level
anchors, but phylum/class clade conclusions are skipped. For strong ecological
interpretation such as Firmicutes vs Bacteroidetes, provide a taxonomy table
with at least columns containing genus and phylum.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import itertools
import math
import warnings
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2, kruskal, mannwhitneyu, spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_curve, auc

warnings.filterwarnings("ignore")


# =============================================================================
# 0. Paths and user parameters -- edit here only
# =============================================================================

ABUNDANCE_PATH = r"/home/zhangyuli/傅里叶非时序/data/HMP_site/abundance.csv"
METADATA_PATH = r"/home/zhangyuli/傅里叶非时序/data/HMP_site/metadata.csv"

# Your lineage file. One row per genus lineage, e.g.
# k__Bacteria;p__Firmicutes;c__Clostridia;o__...;f__...;g__Blautia
# This file is used BOTH for phylogenetic genus ordering and for
# phylum/class/order/family back-projection.
PHYLOGENY_PATH = r"/home/zhangyuli/傅里叶非时序/data/phylogeny.csv"

# Optional. Strongly recommended for clade-level interpretation.
# Expected: one row per genus/feature, with columns such as Genus, Phylum, Class,
# Order, Family. If empty or missing, genus-level back-projection still runs.
TAXONOMY_PATH = r""  # e.g. r"/home/zhangyuli/傅里叶非时序/data/HMP_site/taxonomy.csv"

# Optional. One-column CSV/TXT containing genus names in phylogenetic/taxonomic order.
# If empty, the script will sort by taxonomy if TAXONOMY_PATH is provided;
# otherwise it keeps the input abundance order.
GENUS_ORDER_PATH = r""

OUT_DIR = Path(r"/home/zhangyuli/傅里叶非时序/figures/HMP_site/phase_taxon")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Metadata columns in your current file.
SAMPLE_ID_COL = "sample"     # your metadata has column 'sample'
BODY_SITE_COL = "SITE"       # Gastrointestinal Tract / Oral / Skin / Airways / Urogenital Tract
SUBJECT_ID_COL = "RSID"
VISIT_COL = "VISITNO"

# Map raw HMP SITE labels to clean ecological habitats.
CURATE_BODY_SITE_LABELS = True
SITE_MAP = {
    "Gastrointestinal Tract": "Gut",
    "Gastrointestinal tract": "Gut",
    "GI": "Gut",
    "Stool": "Gut",
    "Oral": "Oral",
    "Skin": "Skin",
    "Airways": "Airway",
    "Airway": "Airway",
    "Anterior nares": "Airway",
    "Urogenital Tract": "Vagina",
    "Urogenital tract": "Vagina",
    "Vagina": "Vagina",
}
PREFERRED_BODY_SITES = ["Gut", "Oral", "Skin", "Airway", "Vagina"]

# Sampling and filtering
RANDOM_STATE = 20260520
MIN_SAMPLES_PER_SITE = 80
SAMPLES_PER_SITE = 220       # set None to use all, but balanced is cleaner
MIN_TOTAL_COUNT_PER_GENUS = 20
MIN_PREVALENCE = 0.002
PSEUDOCOUNT = 1e-9
USE_HANNING_WINDOW = True

# Frequency/mode settings. Modes are FFT indices k, excluding k=0.
MAX_MODE_INDEX = 64
LOW_MAX_K = 12
BOX_SPLIT_K = 32  # more even split for the low-vs-high boxplot
MID_MAX_K = 32
TOP_ATTRACTOR_MODES = 3
MODE_FOR_BACKPROJECTION = "auto"  # "auto" or integer, e.g. 4
BAND_FOR_TAXON_BACKPROJECTION = "low"  # "single" or "low". Low band is usually more ecological.

# Statistics
N_LABEL_PERMUTATIONS = 300
N_RANDOM_PHASE_PERMUTATIONS = 300
N_CV_SPLITS = 5

# Taxonomic back-projection
PEAK_WINDOW_FRACTION = 0.015   # window around wave peaks; 1.5% of ordered genera
TOP_ANCHOR_GENERA = 20
TOP_ANCHOR_CLADES = 12
PAIRWISE_MODE = "auto"         # use selected backprojection mode
PAIRWISE_TOP_GENERA = 15

# Faster testing switch
FAST_TEST = False
if FAST_TEST:
    SAMPLES_PER_SITE = 80
    MAX_MODE_INDEX = 32
    N_LABEL_PERMUTATIONS = 50
    N_RANDOM_PHASE_PERMUTATIONS = 50
    TOP_ANCHOR_GENERA = 10


# =============================================================================
# 1. Plot style
# =============================================================================

COL = {
    "dark": "#3A2634",
    "grey": "#AFA99B",
    "light": "#E8E3DB",
    "gut": "#5D8790",
    "oral": "#D75A49",
    "skin": "#6B5570",
    "airway": "#6FAE8D",
    "vagina": "#D8A24C",
    "null": "#BDB7AA",
}
SITE_COLORS = {
    "Gut": COL["gut"],
    "Oral": COL["oral"],
    "Skin": COL["skin"],
    "Airway": COL["airway"],
    "Vagina": COL["vagina"],
}
CMAP_PPC = LinearSegmentedColormap.from_list("ppc", ["#F4E7C6", "#E6774E", "#7B1E3C"], N=256)
CMAP_BLUE_RED = LinearSegmentedColormap.from_list("br", ["#4D8796", "#F7F4EC", "#C84E3A"], N=256)


def set_nature_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.dpi": 160,
    })


def style_axis(ax, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COL["dark"])
    ax.spines["bottom"].set_color(COL["dark"])
    ax.tick_params(colors=COL["dark"], length=3, width=0.65)
    if grid:
        ax.grid(axis="y", color=COL["light"], lw=0.55, alpha=0.9)
        ax.set_axisbelow(True)


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", va="top", ha="left", color=COL["dark"])


set_nature_style()


# =============================================================================
# 2. Circular statistics
# =============================================================================


def circ_mean(angles: np.ndarray) -> float:
    z = np.nanmean(np.exp(1j * angles))
    return float(np.angle(z))


def circ_r(angles: np.ndarray) -> float:
    angles = np.asarray(angles)
    angles = angles[np.isfinite(angles)]
    if len(angles) == 0:
        return np.nan
    return float(np.abs(np.mean(np.exp(1j * angles))))


def circ_dist(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    return np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))))


def ppc_from_angles(angles: np.ndarray) -> float:
    angles = np.asarray(angles)
    angles = angles[np.isfinite(angles)]
    n = len(angles)
    if n < 2:
        return np.nan
    r = circ_r(angles)
    ppc = (n * r * r - 1.0) / (n - 1.0)
    return float(ppc)


def rayleigh_test(angles: np.ndarray) -> Tuple[float, float, float]:
    """Approximate Rayleigh test for non-uniformity.
    Returns Rbar, z, p.
    """
    angles = np.asarray(angles)
    angles = angles[np.isfinite(angles)]
    n = len(angles)
    if n < 5:
        return np.nan, np.nan, np.nan
    r = circ_r(angles)
    z = n * r * r
    # Zar approximation, good for moderate n.
    p = np.exp(-z) * (
        1 + (2*z - z*z) / (4*n)
        - (24*z - 132*z*z + 76*z**3 - 9*z**4) / (288*n*n)
    )
    p = float(min(max(p, 0.0), 1.0))
    return float(r), float(z), p


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    pf = p[finite]
    if len(pf) == 0:
        return q
    order = np.argsort(pf)
    ranked = pf[order]
    m = len(ranked)
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    tmp = np.empty_like(adj)
    tmp[order] = adj
    q[finite] = tmp
    return q


# =============================================================================
# 3. Data loading and cleaning
# =============================================================================


def clean_genus(x: str) -> str:
    s = str(x).strip().strip('"').strip("'")
    if ";" in s:
        s = s.split(";")[-1]
    for pref in ["g__", "Genus:", "genus__", "D_5__", "k__", "p__", "c__", "o__", "f__", "s__"]:
        if s.startswith(pref):
            s = s[len(pref):]
    s = s.strip()
    if s in {"", "nan", "None", "NA", "Unclassified", "uncultured"}:
        return "Unclassified"
    return s


def load_abundance() -> pd.DataFrame:
    path = Path(ABUNDANCE_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find abundance file: {path}")
    abu = pd.read_csv(path, index_col=0)
    abu.index = [clean_genus(x) for x in abu.index]
    abu = abu.groupby(abu.index).sum()
    # columns are samples; make strings and numeric
    abu.columns = abu.columns.astype(str)
    abu = abu.apply(pd.to_numeric, errors="coerce").fillna(0)
    return abu


def load_metadata() -> pd.DataFrame:
    path = Path(METADATA_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find metadata file: {path}")
    meta = pd.read_csv(path)
    if SAMPLE_ID_COL not in meta.columns:
        # Try common fallback
        if "SampleID" in meta.columns:
            meta = meta.rename(columns={"SampleID": SAMPLE_ID_COL})
        else:
            raise ValueError(f"Cannot find sample column {SAMPLE_ID_COL}. Available: {list(meta.columns)}")
    if BODY_SITE_COL not in meta.columns:
        raise ValueError(f"Cannot find body-site column {BODY_SITE_COL}. Available: {list(meta.columns)}")
    meta[SAMPLE_ID_COL] = meta[SAMPLE_ID_COL].astype(str)
    meta = meta.set_index(SAMPLE_ID_COL, drop=False)
    raw = meta[BODY_SITE_COL].astype(str)
    if CURATE_BODY_SITE_LABELS:
        meta["body_site_raw"] = raw
        meta["body_site"] = raw.map(lambda x: SITE_MAP.get(x, x))
    else:
        meta["body_site_raw"] = raw
        meta["body_site"] = raw
    if SUBJECT_ID_COL in meta.columns:
        meta[SUBJECT_ID_COL] = meta[SUBJECT_ID_COL].astype(str)
    if VISIT_COL in meta.columns:
        meta[VISIT_COL] = pd.to_numeric(meta[VISIT_COL], errors="coerce")
    return meta


def parse_lineage_record(raw: str) -> Dict[str, str]:
    """Parse one semicolon lineage string into taxonomic ranks.

    Accepts strings such as:
        k__Bacteria;p__Firmicutes;c__Clostridia;o__...;f__...;g__Faecalibacterium
    or plain six-level strings separated by semicolons.
    """
    parts = [str(x).strip().strip('"').strip("'") for x in str(raw).split(";")]
    parts = [p for p in parts if p and p.lower() not in {"nan", "none"}]

    out = {"phylum": "Unknown", "class": "Unknown", "order": "Unknown", "family": "Unknown", "genus": "Unclassified"}

    prefix_map = {
        "p__": "phylum", "phylum__": "phylum", "D_1__": "phylum",
        "c__": "class", "class__": "class", "D_2__": "class",
        "o__": "order", "order__": "order", "D_3__": "order",
        "f__": "family", "family__": "family", "D_4__": "family",
        "g__": "genus", "genus__": "genus", "D_5__": "genus",
    }

    # Prefix-aware parsing.
    for p in parts:
        for pref, rank in prefix_map.items():
            if p.startswith(pref):
                val = p[len(pref):].strip()
                if val:
                    out[rank] = val
                break

    # Positional fallback: k, p, c, o, f, g.
    if out["genus"] == "Unclassified" and len(parts) >= 6:
        vals = [re.sub(r"^[a-z]__", "", x).strip() for x in parts]
        out["phylum"] = vals[1] if len(vals) > 1 and vals[1] else out["phylum"]
        out["class"] = vals[2] if len(vals) > 2 and vals[2] else out["class"]
        out["order"] = vals[3] if len(vals) > 3 and vals[3] else out["order"]
        out["family"] = vals[4] if len(vals) > 4 and vals[4] else out["family"]
        out["genus"] = clean_genus(vals[5])

    out["genus"] = clean_genus(out["genus"])
    for r in ["phylum", "class", "order", "family"]:
        out[r] = str(out[r]).replace("p__", "").replace("c__", "").replace("o__", "").replace("f__", "").strip()
        if out[r] in {"", "nan", "None"}:
            out[r] = "Unknown"
    return out


def load_lineage_file(path: Path, genera: List[str]) -> Optional[pd.DataFrame]:
    """Load phylogeny.csv-style lineage file and return taxonomy with phylo_order."""
    if not path.exists() or not path.is_file():
        return None
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] == 0:
        return None
    records = []
    for i, val in enumerate(raw.iloc[:, 0].astype(str)):
        rec = parse_lineage_record(val)
        rec["phylo_order"] = i
        records.append(rec)
    tax = pd.DataFrame(records)
    tax = tax[tax["genus"].notna() & ~tax["genus"].isin(["", "Unclassified"])]
    tax = tax.drop_duplicates("genus", keep="first").set_index("genus")

    out = pd.DataFrame(index=pd.Index(genera, name="genus"))
    out = out.join(tax, how="left")
    for col in ["phylum", "class", "order", "family"]:
        out[col] = out[col].fillna("Unknown")
    out["phylo_order"] = out["phylo_order"].fillna(1e12).astype(float)
    return out.reset_index()


def load_table_taxonomy(path: Path, genera: List[str]) -> Optional[pd.DataFrame]:
    """Load table-style taxonomy with genus/phylum/class/order/family columns."""
    if not path.exists() or not path.is_file():
        return None
    tax = pd.read_csv(path)

    # If this is actually a one-column lineage table, parse as lineage.
    if tax.shape[1] == 1 and tax.iloc[:, 0].astype(str).str.contains(";").mean() > 0.5:
        return load_lineage_file(path, genera)

    genus_candidates = ["Genus", "genus", "GENUS", "taxon", "Taxon", "feature", "Feature", "name", "Name"]
    genus_col = next((c for c in genus_candidates if c in tax.columns), None)
    if genus_col is None:
        genus_col = tax.columns[0]
    tax["genus"] = tax[genus_col].map(clean_genus)

    col_map = {}
    for target, cands in {
        "phylum": ["Phylum", "phylum", "PHYLUM", "Rank2", "rank2", "D_1", "D_1__"],
        "class": ["Class", "class", "CLASS", "Rank3", "rank3", "D_2", "D_2__"],
        "order": ["Order", "order", "ORDER", "Rank4", "rank4", "D_3", "D_3__"],
        "family": ["Family", "family", "FAMILY", "Rank5", "rank5", "D_4", "D_4__"],
    }.items():
        found = next((c for c in cands if c in tax.columns), None)
        if found is not None:
            col_map[found] = target
    tax = tax.rename(columns=col_map)
    for col in ["phylum", "class", "order", "family"]:
        if col not in tax.columns:
            tax[col] = "Unknown"
        tax[col] = tax[col].astype(str).str.replace(r"^[a-z]__", "", regex=True).replace({"nan": "Unknown", "": "Unknown"})
    if "phylo_order" not in tax.columns:
        tax["phylo_order"] = np.arange(len(tax))
    tax = tax[["genus", "phylum", "class", "order", "family", "phylo_order"]].drop_duplicates("genus")
    tax = tax.set_index("genus")

    out = pd.DataFrame(index=pd.Index(genera, name="genus"))
    out = out.join(tax, how="left")
    for col in ["phylum", "class", "order", "family"]:
        out[col] = out[col].fillna("Unknown")
    out["phylo_order"] = out["phylo_order"].fillna(1e12).astype(float)
    return out.reset_index()


def load_taxonomy(genera: List[str]) -> Optional[pd.DataFrame]:
    # Priority: TAXONOMY_PATH if set; otherwise PHYLOGENY_PATH.
    tax_path = str(TAXONOMY_PATH).strip()
    phy_path = str(PHYLOGENY_PATH).strip()

    if tax_path not in {"", "."}:
        tax = load_table_taxonomy(Path(tax_path), genera)
        if tax is not None:
            return tax

    if phy_path not in {"", "."}:
        tax = load_lineage_file(Path(phy_path), genera)
        if tax is not None:
            return tax

    return None


def choose_order(abu: pd.DataFrame, tax: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    genera = list(abu.index)

    # External genus order has highest priority. If empty, use PHYLOGENY_PATH order.
    order_str = str(GENUS_ORDER_PATH).strip()
    phy_str = str(PHYLOGENY_PATH).strip()

    order_source_path = None
    if order_str not in {"", "."} and Path(order_str).exists() and Path(order_str).is_file():
        order_source_path = Path(order_str)
    elif phy_str not in {"", "."} and Path(phy_str).exists() and Path(phy_str).is_file():
        order_source_path = Path(phy_str)

    if order_source_path is not None:
        odf = pd.read_csv(order_source_path, header=None)
        order = [clean_genus(x) for x in odf.iloc[:, 0].astype(str)]
        ordered = [g for g in order if g in abu.index]
        leftovers = [g for g in abu.index if g not in ordered]
        if tax is not None and "phylo_order" in tax.columns:
            t_left = tax.set_index("genus").reindex(leftovers).reset_index()
            t_left["phylo_order"] = t_left["phylo_order"].fillna(1e12)
            leftovers = t_left.sort_values(["phylo_order", "genus"])["genus"].tolist()
        else:
            leftovers = sorted(leftovers)
        final_order = ordered + leftovers
        source = f"lineage_order:{order_source_path}"
    elif tax is not None and "phylo_order" in tax.columns:
        t = tax.set_index("genus").loc[genera].reset_index()
        t["present_order"] = np.arange(len(t))
        t["phylo_order"] = t["phylo_order"].fillna(1e12)
        t = t.sort_values(["phylo_order", "phylum", "class", "order", "family", "genus", "present_order"])
        final_order = t["genus"].tolist()
        source = "taxonomy_phylo_order"
    elif tax is not None:
        t = tax.set_index("genus").loc[genera].reset_index()
        t["present_order"] = np.arange(len(t))
        t = t.sort_values(["phylum", "class", "order", "family", "genus", "present_order"])
        final_order = t["genus"].tolist()
        source = "taxonomy_sorted_order"
    else:
        final_order = genera
        source = "input_order_no_taxonomy"

    abu2 = abu.loc[final_order]
    if tax is None:
        tax2 = pd.DataFrame({"genus": final_order, "phylum": "Unknown", "class": "Unknown", "order": "Unknown", "family": "Unknown", "phylo_order": np.arange(len(final_order))})
    else:
        tax2 = tax.set_index("genus").reindex(final_order).reset_index()
        for col in ["phylum", "class", "order", "family"]:
            tax2[col] = tax2[col].fillna("Unknown")
        if "phylo_order" not in tax2.columns:
            tax2["phylo_order"] = np.arange(len(tax2))
    tax2["taxon_order"] = np.arange(1, len(final_order) + 1)
    return abu2, tax2, source


def filter_and_balance(abu: pd.DataFrame, meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Align samples
    shared = [s for s in abu.columns.astype(str) if s in meta.index]
    abu = abu.loc[:, shared]
    meta = meta.loc[shared].copy()

    # Keep preferred sites with enough samples
    if PREFERRED_BODY_SITES:
        meta = meta[meta["body_site"].isin(PREFERRED_BODY_SITES)].copy()
        abu = abu.loc[:, meta.index]
    counts_available = meta["body_site"].value_counts().reindex(PREFERRED_BODY_SITES).dropna()
    keep_sites = counts_available[counts_available >= MIN_SAMPLES_PER_SITE].index.tolist()
    meta = meta[meta["body_site"].isin(keep_sites)].copy()
    abu = abu.loc[:, meta.index]

    # Filter genera
    total = abu.sum(axis=1)
    prevalence = (abu > 0).mean(axis=1)
    keep = (total >= MIN_TOTAL_COUNT_PER_GENUS) & (prevalence >= MIN_PREVALENCE)
    abu = abu.loc[keep]

    # Balanced sampling
    rng = np.random.default_rng(RANDOM_STATE)
    if SAMPLES_PER_SITE is None:
        selected = meta.index.to_numpy()
    else:
        n_per = min(SAMPLES_PER_SITE, meta["body_site"].value_counts().min())
        selected = []
        for site in keep_sites:
            ids = meta.index[meta["body_site"].eq(site)].to_numpy()
            selected.extend(rng.choice(ids, size=n_per, replace=False).tolist())
        selected = np.array(selected, dtype=object)
        rng.shuffle(selected)
    meta_bal = meta.loc[selected].copy()
    abu_bal = abu.loc[:, meta_bal.index].copy()

    counts = pd.DataFrame({
        "body_site": keep_sites,
        "available": [int(counts_available.get(s, 0)) for s in keep_sites],
        "balanced": [int((meta_bal["body_site"] == s).sum()) for s in keep_sites],
    })
    return abu_bal, meta_bal, counts


def clr_transform_sample_by_taxon(abu_genus_x_sample: pd.DataFrame) -> pd.DataFrame:
    # Return samples x genera
    abu = abu_genus_x_sample.T.copy()
    abu = abu.loc[abu.sum(axis=1) > 0]
    rel = abu.div(abu.sum(axis=1), axis=0)
    logrel = np.log(rel + PSEUDOCOUNT)
    clr = logrel.sub(logrel.mean(axis=1), axis=0)
    return clr.astype(np.float32)


# =============================================================================
# 4. Spectral transform and mode statistics
# =============================================================================


def compute_fft(clr: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = clr.values.astype(np.float32)
    n = X.shape[1]
    if USE_HANNING_WINDOW:
        X = X * np.hanning(n).astype(np.float32)[None, :]
    Z = np.fft.rfft(X, axis=1)
    max_k = min(MAX_MODE_INDEX, Z.shape[1] - 1)
    modes = np.arange(1, max_k + 1)
    freq = np.fft.rfftfreq(n, d=1.0)[modes]
    return Z[:, modes], modes, freq


def phase_table(Z: np.ndarray, modes: np.ndarray, sample_ids: List[str], meta: pd.DataFrame) -> pd.DataFrame:
    ph = np.angle(Z)
    records = []
    for j, k in enumerate(modes):
        tmp = pd.DataFrame({
            "sample": sample_ids,
            "mode": int(k),
            "phase": ph[:, j],
            "amplitude": np.abs(Z[:, j]),
            "body_site": meta.loc[sample_ids, "body_site"].values,
        })
        if SUBJECT_ID_COL in meta.columns:
            tmp["subject"] = meta.loc[sample_ids, SUBJECT_ID_COL].astype(str).values
        if VISIT_COL in meta.columns:
            tmp["visit"] = meta.loc[sample_ids, VISIT_COL].values
        records.append(tmp)
    return pd.concat(records, ignore_index=True)


def compute_site_mode_stats(ptab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (site, k), sub in ptab.groupby(["body_site", "mode"]):
        angles = sub["phase"].values
        r, z, p = rayleigh_test(angles)
        rows.append({
            "body_site": site,
            "mode": int(k),
            "n": len(sub),
            "mean_phase": circ_mean(angles),
            "R": r,
            "PLV": r,
            "PPC": ppc_from_angles(angles),
            "rayleigh_z": z,
            "rayleigh_p": p,
        })
    out = pd.DataFrame(rows)
    out["rayleigh_q"] = benjamini_hochberg(out["rayleigh_p"].values)
    out["minus_log10_q"] = -np.log10(out["rayleigh_q"].clip(lower=1e-300))
    out["band"] = out["mode"].map(mode_band)
    return out


def mode_band(k: int) -> str:
    if k <= LOW_MAX_K:
        return "low"
    if k <= MID_MAX_K:
        return "mid"
    return "high"


def circular_between_site_distance(means: Dict[str, float], weights: Dict[str, float]) -> float:
    sites = list(means.keys())
    if len(sites) < 2:
        return np.nan
    vals = []
    wts = []
    for a, b in itertools.combinations(sites, 2):
        vals.append(float(circ_dist(means[a], means[b])))
        wts.append(math.sqrt(max(weights.get(a, 0), 0) * max(weights.get(b, 0), 0)))
    vals = np.array(vals)
    wts = np.array(wts)
    if np.sum(wts) <= 1e-12:
        return float(np.mean(vals))
    return float(np.average(vals, weights=wts))


def phase_attractor_scores(ptab: pd.DataFrame, modes: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in modes:
        subk = ptab[ptab["mode"].eq(k)]
        means = {}
        ppcs = {}
        for site, ss in subk.groupby("body_site"):
            means[site] = circ_mean(ss["phase"].values)
            ppcs[site] = max(ppc_from_angles(ss["phase"].values), 0)
        within_ppc = float(np.nanmean(list(ppcs.values())))
        between = circular_between_site_distance(means, ppcs)
        score = within_ppc * (between / np.pi)
        rows.append({
            "mode": int(k),
            "band": mode_band(int(k)),
            "within_mean_PPC": within_ppc,
            "between_phase_distance": between,
            "phase_attractor_score": score,
        })
    return pd.DataFrame(rows)


def label_permutation_for_scores(ptab: pd.DataFrame, modes: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    obs = phase_attractor_scores(ptab, modes)
    null_records = []
    # preserve group sizes by permuting labels across samples per mode consistently
    base = ptab[["sample", "body_site"]].drop_duplicates().set_index("sample")
    sample_ids = base.index.to_numpy()
    labels = base["body_site"].to_numpy()
    for p in range(N_LABEL_PERMUTATIONS):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        mapping = dict(zip(sample_ids, shuffled))
        tmp = ptab.copy()
        tmp["body_site"] = tmp["sample"].map(mapping)
        ns = phase_attractor_scores(tmp, modes)
        ns["perm"] = p
        null_records.append(ns)
    null = pd.concat(null_records, ignore_index=True)
    summ = null.groupby("mode")["phase_attractor_score"].agg(
        null_median="median",
        null_q05=lambda x: np.quantile(x, 0.05),
        null_q95=lambda x: np.quantile(x, 0.95),
    ).reset_index()
    # empirical one-sided p: null >= observed
    pvals = []
    for _, row in obs.iterrows():
        vals = null.loc[null["mode"].eq(row["mode"]), "phase_attractor_score"].values
        pval = (np.sum(vals >= row["phase_attractor_score"]) + 1) / (len(vals) + 1)
        pvals.append(pval)
    obs["label_perm_p"] = pvals
    obs["label_perm_q"] = benjamini_hochberg(np.array(pvals))
    out = obs.merge(summ, on="mode", how="left")
    return out


def amplitude_association(Z: np.ndarray, modes: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    from sklearn.feature_selection import f_classif
    amp = np.abs(Z)
    fvals, pvals = f_classif(amp, y)
    return pd.DataFrame({"mode": modes.astype(int), "amplitude_F": fvals, "amplitude_p": pvals})


# =============================================================================
# 5. Taxonomic back-projection
# =============================================================================


def mean_complex_by_site(Z: np.ndarray, modes: np.ndarray, sample_ids: List[str], meta: pd.DataFrame) -> Dict[str, np.ndarray]:
    site_to_vec = {}
    for site in PREFERRED_BODY_SITES:
        ids = [i for i, s in enumerate(sample_ids) if meta.loc[s, "body_site"] == site]
        if len(ids) == 0:
            continue
        site_to_vec[site] = np.mean(Z[ids, :], axis=0)
    return site_to_vec


def reconstruct_wave_from_modes(mean_coeff: np.ndarray,
                                modes: np.ndarray,
                                n_taxa: int,
                                selected_modes: List[int]) -> np.ndarray:
    full = np.zeros(n_taxa // 2 + 1, dtype=complex)
    mode_to_pos = {int(k): i for i, k in enumerate(modes)}
    for k in selected_modes:
        if k in mode_to_pos and k < len(full):
            full[k] = mean_coeff[mode_to_pos[k]]
    x = np.fft.irfft(full, n=n_taxa)
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    return x


def selected_backprojection_modes(score_df: pd.DataFrame) -> List[int]:
    if BAND_FOR_TAXON_BACKPROJECTION == "low":
        return list(range(1, LOW_MAX_K + 1))
    if MODE_FOR_BACKPROJECTION == "auto":
        # choose highest low-mode score if possible
        sub = score_df[score_df["mode"].le(LOW_MAX_K)].copy()
        if sub.empty:
            sub = score_df.copy()
        return [int(sub.sort_values("phase_attractor_score", ascending=False).iloc[0]["mode"])]
    return [int(MODE_FOR_BACKPROJECTION)]


def site_mean_clr_by_genus(clr: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    tmp = clr.copy()
    tmp["body_site"] = meta.loc[clr.index, "body_site"].values
    means = tmp.groupby("body_site").mean(numeric_only=True)
    return means


def find_local_peaks(x: np.ndarray, top_n: int = 10, sign: str = "positive") -> np.ndarray:
    arr = x if sign == "positive" else -x
    # local maxima
    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] >= arr[i-1] and arr[i] >= arr[i+1]:
            peaks.append(i)
    if not peaks:
        peaks = list(np.argsort(arr)[-top_n:])
    peaks = np.array(peaks, dtype=int)
    peaks = peaks[np.argsort(arr[peaks])[::-1]]
    return peaks[:top_n]


def compute_taxon_anchor_tables(site_waves: Dict[str, np.ndarray],
                                clr: pd.DataFrame,
                                meta: pd.DataFrame,
                                tax_order: pd.DataFrame,
                                selected_modes: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute genus and clade anchors for each site.

    Anchor score combines:
    - positive locked waveform position
    - site specificity in mean CLR relative to other sites

    This prevents labeling arbitrary sine-wave peaks that are not actually enriched
    in the site.
    """
    # The reconstructed wave is built on the genus order used for FFT.
    # Some HMP genus tables may still contain extra filtered genera or duplicated names after
    # taxonomy/order matching. Therefore we force all taxon-level summaries to use exactly
    # the same genus vector as site_waves. This prevents shape mismatches such as
    # wave=(280,) while site specificity=(314,).
    n_wave = len(next(iter(site_waves.values()))) if site_waves else len(tax_order)
    tax_order = tax_order.iloc[:n_wave].copy().reset_index(drop=True)
    n = len(tax_order)
    window = max(3, int(round(n * PEAK_WINDOW_FRACTION)))
    genera = tax_order["genus"].astype(str).tolist()
    site_means = site_mean_clr_by_genus(clr, meta)
    # Collapse duplicated genus columns if any, then reindex to the FFT genus order.
    site_means.columns = site_means.columns.astype(str)
    if site_means.columns.duplicated().any():
        site_means = site_means.T.groupby(level=0).mean().T
    site_means = site_means.reindex(columns=genera).fillna(0.0)

    genus_records = []
    clade_records = []
    peak_records = []

    for site, wave in site_waves.items():
        if site not in site_means.index:
            continue
        other_sites = [s for s in site_means.index if s != site]
        specificity = site_means.loc[site] - site_means.loc[other_sites].median(axis=0)
        spec_z = (specificity - specificity.mean()) / (specificity.std() + 1e-12)
        wave = np.asarray(wave, dtype=float)[:n]
        wave_pos = np.maximum(wave, 0)
        wave_z = (wave_pos - wave_pos.mean()) / (wave_pos.std() + 1e-12)
        spec_vec = spec_z.reindex(genera).fillna(0.0).values.astype(float)
        anchor = wave_z * spec_vec
        # positive means wave peak coincides with site-enriched genera
        df = tax_order.copy()
        df["body_site"] = site
        df["selected_modes"] = "+".join(map(str, selected_modes))
        df["wave_value"] = wave
        df["site_specificity_clr"] = specificity.values
        df["anchor_score"] = anchor
        df["rank_anchor"] = (-df["anchor_score"]).rank(method="first")
        genus_records.append(df.sort_values("anchor_score", ascending=False).head(TOP_ANCHOR_GENERA))

        peaks = find_local_peaks(wave, top_n=5, sign="positive")
        for pi, p in enumerate(peaks):
            lo = max(0, p - window)
            hi = min(n, p + window + 1)
            sub = df.iloc[lo:hi].copy()
            # dominant genera around the peak by anchor score
            top_sub = sub.sort_values("anchor_score", ascending=False).head(8)
            peak_records.append({
                "body_site": site,
                "selected_modes": "+".join(map(str, selected_modes)),
                "peak_rank": pi + 1,
                "peak_position": int(p + 1),
                "peak_genus_at_center": genera[p],
                "window_start": int(lo + 1),
                "window_end": int(hi),
                "top_anchor_genera_in_window": ";".join(top_sub["genus"].astype(str).tolist()),
                "top_anchor_scores_in_window": ";".join([f"{v:.3g}" for v in top_sub["anchor_score"].values]),
                "dominant_phylum_in_window": top_sub["phylum"].mode().iloc[0] if "phylum" in top_sub else "Unknown",
                "dominant_family_in_window": top_sub["family"].mode().iloc[0] if "family" in top_sub else "Unknown",
            })

        # Clade summaries
        for level in ["phylum", "class", "order", "family"]:
            if level not in df.columns:
                continue
            for clade, sub in df.groupby(level):
                if clade == "Unknown" or len(sub) < 2:
                    continue
                clade_records.append({
                    "body_site": site,
                    "taxonomic_level": level,
                    "clade": clade,
                    "n_genera": len(sub),
                    "mean_wave": float(sub["wave_value"].mean()),
                    "mean_specificity_clr": float(sub["site_specificity_clr"].mean()),
                    "mean_anchor_score": float(sub["anchor_score"].mean()),
                    "sum_positive_anchor_score": float(np.maximum(sub["anchor_score"], 0).sum()),
                    "top_genera": ";".join(sub.sort_values("anchor_score", ascending=False).head(6)["genus"].astype(str).tolist()),
                })

    genus_df = pd.concat(genus_records, ignore_index=True) if genus_records else pd.DataFrame()
    clade_df = pd.DataFrame(clade_records)
    peak_df = pd.DataFrame(peak_records)
    if not clade_df.empty:
        clade_df["rank_within_site_level"] = clade_df.groupby(["body_site", "taxonomic_level"])["sum_positive_anchor_score"].rank(ascending=False, method="first")
    return genus_df, clade_df, peak_df


def pairwise_phase_shift_backprojection(site_waves: Dict[str, np.ndarray],
                                        clr: pd.DataFrame,
                                        meta: pd.DataFrame,
                                        tax_order: pd.DataFrame,
                                        selected_modes: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_wave = len(next(iter(site_waves.values()))) if site_waves else len(tax_order)
    tax_order = tax_order.iloc[:n_wave].copy().reset_index(drop=True)
    genera = tax_order["genus"].astype(str).tolist()
    site_means = site_mean_clr_by_genus(clr, meta)
    site_means.columns = site_means.columns.astype(str)
    if site_means.columns.duplicated().any():
        site_means = site_means.T.groupby(level=0).mean().T
    site_means = site_means.reindex(columns=genera).fillna(0.0)
    rows_g = []
    rows_c = []
    for a, b in itertools.combinations(site_waves.keys(), 2):
        # wave shift from B to A
        wa = np.asarray(site_waves[a], dtype=float)[:len(genera)]
        wb = np.asarray(site_waves[b], dtype=float)[:len(genera)]
        dw = wa - wb
        ds = site_means.loc[a] - site_means.loc[b]
        ds_z = (ds - ds.mean()) / (ds.std() + 1e-12)
        dw_z = (dw - dw.mean()) / (dw.std() + 1e-12)
        shift_score = dw_z * ds_z.reindex(genera).fillna(0.0).values.astype(float)
        df = tax_order.copy()
        df["site_A"] = a
        df["site_B"] = b
        df["selected_modes"] = "+".join(map(str, selected_modes))
        df["wave_shift_A_minus_B"] = dw
        df["clr_specificity_A_minus_B"] = ds.values
        df["phase_shift_taxon_score"] = shift_score
        topA = df.sort_values("phase_shift_taxon_score", ascending=False).head(PAIRWISE_TOP_GENERA)
        topB = df.sort_values("phase_shift_taxon_score", ascending=True).head(PAIRWISE_TOP_GENERA).copy()
        topA["direction"] = f"{a}>{b}"
        topB["direction"] = f"{b}>{a}"
        topB["phase_shift_taxon_score"] = -topB["phase_shift_taxon_score"]
        rows_g.extend([topA, topB])
        for level in ["phylum", "class", "order", "family"]:
            for clade, sub in df.groupby(level):
                if clade == "Unknown" or len(sub) < 2:
                    continue
                val = float(sub["phase_shift_taxon_score"].mean())
                rows_c.append({
                    "site_A": a,
                    "site_B": b,
                    "taxonomic_level": level,
                    "clade": clade,
                    "n_genera": len(sub),
                    "mean_shift_score_A_minus_B": val,
                    "abs_mean_shift_score": abs(val),
                    "direction": f"{a}>{b}" if val > 0 else f"{b}>{a}",
                    "top_genera": ";".join(sub.assign(abs_score=sub["phase_shift_taxon_score"].abs()).sort_values("abs_score", ascending=False).head(6)["genus"].astype(str).tolist()),
                })
    genus_shift = pd.concat(rows_g, ignore_index=True) if rows_g else pd.DataFrame()
    clade_shift = pd.DataFrame(rows_c)
    return genus_shift, clade_shift


# =============================================================================
# 6. Longitudinal and classification supporting analyses
# =============================================================================


def phase_feature_matrix(Z: np.ndarray, modes: np.ndarray, selected_modes: List[int], kind: str) -> np.ndarray:
    pos = [np.where(modes == k)[0][0] for k in selected_modes if k in modes]
    sub = Z[:, pos]
    if kind == "amplitude":
        return np.abs(sub).astype(np.float32)
    if kind == "phase":
        ph = np.angle(sub)
        return np.concatenate([np.cos(ph), np.sin(ph)], axis=1).astype(np.float32)
    if kind == "amp_phase":
        ph = np.angle(sub)
        amp = np.abs(sub)
        return np.concatenate([amp, np.cos(ph), np.sin(ph)], axis=1).astype(np.float32)
    if kind == "complex":
        return np.concatenate([sub.real, sub.imag], axis=1).astype(np.float32)
    raise ValueError(kind)


def classification_benchmark(Z: np.ndarray, modes: np.ndarray, meta: pd.DataFrame, sample_ids: List[str], selected_modes: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y_raw = meta.loc[sample_ids, "body_site"].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    cv = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    cm_sum = np.zeros((len(le.classes_), len(le.classes_)), dtype=float)
    for kind in ["amplitude", "phase", "amp_phase", "complex"]:
        X = phase_feature_matrix(Z, modes, selected_modes, kind)
        preds = np.empty_like(y)
        for tr, te in cv.split(X, y):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto"))
            clf.fit(X[tr], y[tr])
            preds[te] = clf.predict(X[te])
        bal = balanced_accuracy_score(y, preds)
        rows.append({"feature": kind, "balanced_accuracy": bal, "n_features": X.shape[1]})
        if kind == "phase":
            cm_sum = confusion_matrix(y, preds, labels=np.arange(len(le.classes_)), normalize="true")
    bench = pd.DataFrame(rows)
    cm = pd.DataFrame(cm_sum, index=le.classes_, columns=le.classes_)
    return bench, cm


def longitudinal_phase_distances(Z: np.ndarray, modes: np.ndarray, meta: pd.DataFrame, sample_ids: List[str], selected_modes: List[int]) -> pd.DataFrame:
    if SUBJECT_ID_COL not in meta.columns or VISIT_COL not in meta.columns:
        return pd.DataFrame()
    pos = [np.where(modes == k)[0][0] for k in selected_modes if k in modes]
    if not pos:
        return pd.DataFrame()
    phases = np.angle(Z[:, pos])
    index = pd.DataFrame({
        "sample": sample_ids,
        "body_site": meta.loc[sample_ids, "body_site"].values,
        "subject": meta.loc[sample_ids, SUBJECT_ID_COL].astype(str).values,
        "visit": meta.loc[sample_ids, VISIT_COL].values,
    })
    rng = np.random.default_rng(RANDOM_STATE)
    records = []

    def mean_phase_distance(i, j) -> float:
        return float(np.mean(circ_dist(phases[i, :], phases[j, :])))

    # same subject same site, different visits
    for (subj, site), ids in index.groupby(["subject", "body_site"]).groups.items():
        ids = list(ids)
        if len(ids) >= 2:
            for i, j in itertools.combinations(ids, 2):
                if index.loc[i, "visit"] != index.loc[j, "visit"]:
                    records.append({"comparison": "same subject-site", "distance": mean_phase_distance(i, j)})

    # different subject same site
    for site, ids in index.groupby("body_site").groups.items():
        ids = np.array(list(ids))
        for _ in range(min(2000, len(ids) * 4)):
            i, j = rng.choice(ids, size=2, replace=False)
            if index.loc[i, "subject"] != index.loc[j, "subject"]:
                records.append({"comparison": "different subject same site", "distance": mean_phase_distance(i, j)})

    # same subject different site
    for subj, ids in index.groupby("subject").groups.items():
        ids = np.array(list(ids))
        if len(np.unique(index.loc[ids, "body_site"])) >= 2:
            for _ in range(min(400, len(ids) * 4)):
                i, j = rng.choice(ids, size=2, replace=False)
                if index.loc[i, "body_site"] != index.loc[j, "body_site"]:
                    records.append({"comparison": "same subject different site", "distance": mean_phase_distance(i, j)})

    # random different site
    all_ids = np.arange(len(index))
    for _ in range(3000):
        i, j = rng.choice(all_ids, size=2, replace=False)
        if index.loc[i, "body_site"] != index.loc[j, "body_site"]:
            records.append({"comparison": "random different site", "distance": mean_phase_distance(i, j)})

    return pd.DataFrame(records)


# =============================================================================
# 7. Plotting
# =============================================================================


def plot_design(ax, counts: pd.DataFrame) -> None:
    df = counts.set_index("body_site").reindex(PREFERRED_BODY_SITES).dropna().reset_index()
    y = np.arange(len(df))[::-1]
    ax.barh(y, df["available"], color="#DDD8CD", height=0.72, label="available")
    ax.barh(y, df["balanced"], color="#6F8E8E", height=0.38, label="balanced")
    ax.set_yticks(y)
    ax.set_yticklabels(df["body_site"])
    ax.set_xlabel("Samples")
    ax.set_title("HMP body-site design")
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax)



def _permute_kruskal_p(groups: List[np.ndarray], n_perm: int = 3000) -> Tuple[float, float]:
    valid = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    if len(valid) < 2:
        return np.nan, np.nan
    obs_h, _ = kruskal(*valid)
    values = np.concatenate(valid)
    sizes = [len(g) for g in valid]
    rng = np.random.default_rng(RANDOM_STATE)
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(values)
        split = []
        st = 0
        for n in sizes:
            split.append(perm[st:st+n])
            st += n
        h, _ = kruskal(*split)
        ge += (h >= obs_h)
    p = (ge + 1) / (n_perm + 1)
    return float(obs_h), float(p)


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return 'NA'
    if p < 1e-4:
        return 'p<1e-4'
    return f'p={p:.3g}'


def _add_sig_bracket(ax, x1: float, x2: float, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color=COL['dark'], lw=0.8, clip_on=False)
    ax.text((x1+x2)/2, y+h*1.05, text, ha='center', va='bottom', color=COL['dark'], fontsize=6.1)


def plot_low_high_locking(ax, site_stats: pd.DataFrame) -> pd.DataFrame:
    low_vals = site_stats.loc[site_stats['mode'].le(BOX_SPLIT_K), 'PPC'].dropna().values
    high_vals = site_stats.loc[site_stats['mode'].gt(BOX_SPLIT_K), 'PPC'].dropna().values
    data = [low_vals, high_vals]
    labels = [f'Low-order\n(k≤{BOX_SPLIT_K})', f'High-order\n(k>{BOX_SPLIT_K})']
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.56)
    colors = ['#C6B1A6', '#547F9E']
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_edgecolor(COL['dark']); patch.set_alpha(0.88)
    for med in bp['medians']:
        med.set_color('white'); med.set_linewidth(1.5)
    ax.set_xticks([1, 2]); ax.set_xticklabels(labels)
    ax.set_ylabel('Within-site PPC')
    ax.set_title('Low-order modes are more strongly phase-locked')
    style_axis(ax)
    p_mw = np.nan
    if len(low_vals) and len(high_vals):
        _, p_mw = mannwhitneyu(low_vals, high_vals, alternative='two-sided')
    top = max([np.nanmax(v) if len(v) else 0 for v in data] + [0])
    base = top + 0.05
    _add_sig_bracket(ax, 1, 2, base, 0.025, _format_p(p_mw))
    H, p_perm = _permute_kruskal_p(data)
    ax.text(0.02, 0.98, f'Permutation p: {_format_p(p_perm)}', transform=ax.transAxes,
            ha='left', va='top', fontsize=6.2, color=COL['dark'])
    ymin = min(0, site_stats['PPC'].min() - 0.05)
    ax.set_ylim(bottom=ymin, top=base + 0.12)
    stats = pd.DataFrame([
        {'comparison': f'k<={BOX_SPLIT_K}_vs_k>{BOX_SPLIT_K}', 'p_value': p_mw, 'overall_kruskal_H': H, 'overall_permutation_p': p_perm}
    ])
    return stats


def plot_rayleigh_heatmap(ax, site_stats: pd.DataFrame) -> None:
    sub = site_stats[site_stats["mode"].le(LOW_MAX_K)].copy()
    mat = sub.pivot(index="body_site", columns="mode", values="minus_log10_q").reindex(PREFERRED_BODY_SITES)
    im = ax.imshow(mat.values, aspect="auto", cmap=CMAP_PPC, vmin=0, vmax=np.nanpercentile(mat.values, 95))
    ax.set_yticks(np.arange(mat.shape[0])); ax.set_yticklabels(mat.index)
    ax.set_xticks(np.arange(mat.shape[1])); ax.set_xticklabels([f"k={int(c)}" for c in mat.columns], rotation=45, ha="right")
    ax.set_title("Rayleigh test: within-site phases are non-uniform")
    ax.set_xlabel("Low-order spectral modes")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("-log10(q)")


def plot_score_permutation(ax, score_df: pd.DataFrame) -> None:
    ax.fill_between(score_df["mode"], score_df["null_q05"], score_df["null_q95"], color=COL["null"], alpha=0.35, label="label perm. 5–95%")
    ax.plot(score_df["mode"], score_df["null_median"], color=COL["null"], lw=1.0, label="null median")
    colors = [COL["oral"] if k <= LOW_MAX_K else "#CFC9BC" for k in score_df["mode"]]
    ax.scatter(score_df["mode"], score_df["phase_attractor_score"], c=colors, s=14, edgecolor="white", lw=0.25, zorder=3, label="observed")
    ax.axvline(LOW_MAX_K + 0.5, color=COL["dark"], ls=":", lw=0.8)
    ax.set_xlabel("FFT mode index")
    ax.set_ylabel("Phase-attractor score")
    ax.set_title("Body-site labels define non-random phase attractors")
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax)



def compute_mode_specific_anchor_labels(site_complex: Dict[str, np.ndarray],
                                       modes: np.ndarray,
                                       clr: pd.DataFrame,
                                       meta: pd.DataFrame,
                                       tax_order: pd.DataFrame,
                                       selected_modes: List[int]) -> pd.DataFrame:
    n_wave = len(tax_order)
    tax_order = tax_order.iloc[:n_wave].copy().reset_index(drop=True)
    genera = tax_order['genus'].astype(str).tolist()
    site_means = site_mean_clr_by_genus(clr, meta)
    site_means.columns = site_means.columns.astype(str)
    if site_means.columns.duplicated().any():
        site_means = site_means.T.groupby(level=0).mean().T
    site_means = site_means.reindex(columns=genera).fillna(0.0)
    rows = []
    for k in selected_modes:
        for site, coeff in site_complex.items():
            if site not in site_means.index:
                continue
            wave = reconstruct_wave_from_modes(coeff, modes, n_wave, [k])
            other_sites = [s for s in site_means.index if s != site]
            specificity = site_means.loc[site] - site_means.loc[other_sites].median(axis=0)
            spec_z = (specificity - specificity.mean()) / (specificity.std() + 1e-12)
            wave_pos = np.maximum(wave, 0)
            wave_z = (wave_pos - wave_pos.mean()) / (wave_pos.std() + 1e-12)
            anchor = wave_z * spec_z.reindex(genera).fillna(0.0).values.astype(float)
            df = tax_order.copy()
            df['mode'] = int(k)
            df['body_site'] = site
            df['wave_value'] = wave
            df['site_specificity_clr'] = specificity.values
            df['anchor_score'] = anchor
            top = df.sort_values('anchor_score', ascending=False).head(3)
            top_genera = top['genus'].astype(str).tolist()
            top_families = top['family'].astype(str).tolist() if 'family' in top.columns else ['Unknown'] * len(top)
            rows.append({
                'mode': int(k), 'body_site': site,
                'top_genus': top_genera[0] if len(top_genera) else 'NA',
                'top_two_genera': '/'.join(top_genera[:2]),
                'top_family': top_families[0] if len(top_families) else 'Unknown',
                'top_three_genera': ';'.join(top_genera),
                'top_anchor_score': float(top['anchor_score'].iloc[0]) if len(top) else np.nan,
            })
    return pd.DataFrame(rows)


def plot_polar_modes(fig, gs_cell, ptab: pd.DataFrame, selected: List[int], anchor_labels: pd.DataFrame) -> None:
    inner = gs_cell.subgridspec(1, len(selected), wspace=0.35)
    legend_handles = []
    for idx, k in enumerate(selected):
        ax = fig.add_subplot(inner[0, idx], projection='polar')
        subk = ptab[ptab['mode'].eq(k)]
        for sidx, site in enumerate(PREFERRED_BODY_SITES):
            ss = subk[subk['body_site'].eq(site)]
            if ss.empty:
                continue
            rng = np.random.default_rng(RANDOM_STATE + int(k) * 10 + sidx)
            vals = ss['phase'].values
            if len(vals) > 180:
                vals = rng.choice(vals, size=180, replace=False)
            radii = 0.90 + 0.07 * rng.normal(size=len(vals))
            radii = np.clip(radii, 0.78, 1.03)
            color = SITE_COLORS.get(site, 'grey')
            ax.scatter(vals, radii, s=9, color=color, alpha=0.22, edgecolor='white', linewidth=0.18)
            if idx == 0:
                legend_handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                                             markeredgecolor='white', markeredgewidth=0.3, markersize=5.5, label=site))
            mu = circ_mean(ss['phase'].values)
            r = max(circ_r(ss['phase'].values), 0.08)
            ax.plot([mu, mu], [0, r], color=color, lw=1.65)
            ax.scatter([mu], [r], s=24, color=color, edgecolor='white', lw=0.35, zorder=5)
        ax.set_title(f'k={k}', pad=4)
        ax.set_yticklabels([]); ax.set_xticklabels([])
        ax.grid(color=COL['light'], lw=0.6)
        ax.set_ylim(0, 1.12)
        if idx == 0 and legend_handles:
            ax.legend(handles=legend_handles, frameon=False, loc='upper left', bbox_to_anchor=(-0.18, 1.15), fontsize=5.6)

def plot_phylum_anchor_heatmap(ax, clade_df: pd.DataFrame) -> None:
    """Plot site-level taxonomic anchors.

    In clade_df, taxonomic ranks are stored in rows:
        taxonomic_level = phylum/class/order/family
        clade = the clade name

    The previous version incorrectly checked for a literal column named "phylum".
    That made panel g blank even when taxonomy was successfully parsed. Here we
    choose the most informative available rank, preferring family, then order,
    then class, then phylum.
    """
    required = {"body_site", "taxonomic_level", "clade", "sum_positive_anchor_score"}
    if clade_df.empty or not required.issubset(set(clade_df.columns)):
        ax.text(
            0.5, 0.5,
            "No usable site-level clade anchors",
            ha="center", va="center", transform=ax.transAxes
        )
        ax.set_axis_off()
        return

    preferred_levels = ["family", "order", "class", "phylum"]
    available = [
        lev for lev in preferred_levels
        if (clade_df["taxonomic_level"].astype(str).str.lower() == lev).any()
    ]
    if not available:
        ax.text(
            0.5, 0.5,
            "No phylum/class/order/family anchors",
            ha="center", va="center", transform=ax.transAxes
        )
        ax.set_axis_off()
        return

    level = available[0]
    sub = clade_df[clade_df["taxonomic_level"].astype(str).str.lower().eq(level)].copy()
    sub = sub[sub["body_site"].isin(PREFERRED_BODY_SITES)].copy()
    if sub.empty:
        ax.text(
            0.5, 0.5,
            f"No {level}-level anchors after site filtering",
            ha="center", va="center", transform=ax.transAxes
        )
        ax.set_axis_off()
        return

    # Select clades with the largest positive anchor mass across sites.
    top = (
        sub.groupby("clade")["sum_positive_anchor_score"]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_ANCHOR_CLADES)
        .index
    )
    mat = sub[sub["clade"].isin(top)].pivot_table(
        index="body_site",
        columns="clade",
        values="sum_positive_anchor_score",
        aggfunc="sum",
        fill_value=0.0,
    )
    mat = mat.reindex(PREFERRED_BODY_SITES).fillna(0.0)
    # Remove all-zero rows/columns after reindexing.
    mat = mat.loc[mat.sum(axis=1) > 0, mat.sum(axis=0) > 0]
    if mat.empty:
        ax.text(
            0.5, 0.5,
            f"No positive {level}-level anchor scores",
            ha="center", va="center", transform=ax.transAxes
        )
        ax.set_axis_off()
        return

    # Row-normalize so each body's strongest anchors are visible, then z-score by clade.
    M = mat.values.astype(float)
    M = M / (M.max(axis=1, keepdims=True) + 1e-12)
    Mz = (M - M.mean(axis=0, keepdims=True)) / (M.std(axis=0, keepdims=True) + 1e-12)

    im = ax.imshow(Mz, aspect="auto", cmap=CMAP_BLUE_RED, vmin=-2, vmax=2)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_title(f"Site phase attractors trace to {level}-level anchors", pad=4)
    ax.set_xlabel(f"{level.capitalize()} anchors")
    cbar = plt.colorbar(im, ax=ax, fraction=0.028, pad=0.010)
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.set_label("Row-scaled anchor z", fontsize=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

def plot_pairwise_clade_shift(ax, clade_shift: pd.DataFrame) -> None:
    if clade_shift.empty:
        ax.text(0.5, 0.5, "No clade-level shift table\ncheck PHYLOGENY_PATH/genus matching", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return
    # Use family-level if enough, else phylum-level
    level = "family" if (clade_shift["taxonomic_level"] == "family").sum() > 20 else "phylum"
    sub = clade_shift[clade_shift["taxonomic_level"].eq(level)].copy()
    # focus on pairwise shifts versus Gut or top absolute shifts
    sub["pair"] = sub["site_A"] + " vs " + sub["site_B"]
    top_clades = sub.groupby("clade")["abs_mean_shift_score"].sum().sort_values(ascending=False).head(12).index
    top_pairs = [p for p in sub["pair"].drop_duplicates().tolist() if "Gut" in p][:4]
    if len(top_pairs) < 4:
        top_pairs = sub["pair"].drop_duplicates().tolist()[:4]
    mat = sub[sub["clade"].isin(top_clades) & sub["pair"].isin(top_pairs)].pivot_table(index="clade", columns="pair", values="mean_shift_score_A_minus_B", aggfunc="mean", fill_value=0)
    im = ax.imshow(mat.values, aspect="auto", cmap=CMAP_BLUE_RED, vmin=-np.nanmax(np.abs(mat.values)), vmax=np.nanmax(np.abs(mat.values)))
    ax.set_yticks(np.arange(mat.shape[0])); ax.set_yticklabels(mat.index)
    ax.set_xticks(np.arange(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_title(f"Pairwise phase shifts trace to {level}-level anchors")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("A minus B shift score")



def plot_longitudinal(ax, dist_df: pd.DataFrame) -> None:
    if dist_df.empty:
        ax.text(0.5, 0.5, 'Longitudinal columns unavailable', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off(); return
    order = ['same subject-site', 'different subject same site', 'same subject different site']
    data = [dist_df.loc[dist_df['comparison'].eq(o), 'distance'].dropna().values for o in order]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.65)
    colors = ['#7FA3A8', '#B8C7D2', '#D77A61']
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_edgecolor(COL['dark']); patch.set_alpha(0.85)
    for med in bp['medians']:
        med.set_color(COL['dark']); med.set_linewidth(1.2)
    ax.set_xticks(range(1, len(order)+1))
    ax.set_xticklabels(['same\nsubject-site', 'different subject\nsame site', 'same subject\ndifferent site'])
    ax.set_ylabel('Mean phase distance')
    ax.set_title('Body-site phases are longitudinally stable attractors')
    style_axis(ax)
    top = max([np.nanmax(v) if len(v) else 0 for v in data] + [0])
    base = top + 0.08
    comps = [(0,1), (0,2), (1,2)]
    for idx, (i,j) in enumerate(comps):
        p = np.nan
        if len(data[i]) and len(data[j]):
            _, p = mannwhitneyu(data[i], data[j], alternative='two-sided')
        _add_sig_bracket(ax, i+1, j+1, base + idx*0.12, 0.03, _format_p(p))
    H, p_perm = _permute_kruskal_p(data)
    ax.text(0.02, 0.98, f'Overall permutation p: {_format_p(p_perm)}', transform=ax.transAxes,
            ha='left', va='top', fontsize=6.1, color=COL['dark'])
    ymin = min(0, np.nanmin(np.concatenate([d for d in data if len(d)])) - 0.05)
    ax.set_ylim(bottom=ymin, top=base + 0.46)


def plot_classification(ax, bench: pd.DataFrame) -> None:
    names = ["amplitude", "phase", "amp_phase", "complex"]
    labels = ["Amplitude", "Phase", "Amp+phase", "Complex"]
    df = bench.set_index("feature").loc[names].reset_index()
    x = np.arange(len(df))
    colors = [COL["gut"], COL["oral"], COL["skin"], COL["vagina"]]
    ax.bar(x, df["balanced_accuracy"], color=colors, edgecolor="white", lw=0.35)
    ax.axhline(1/len(PREFERRED_BODY_SITES), color=COL["dark"], ls="--", lw=0.8, alpha=0.6, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Phase carries body-site information")
    ax.set_ylim(0, min(1, max(0.35, df["balanced_accuracy"].max() + 0.12)))
    style_axis(ax)



def plot_backprojected_waves(ax, genus_anchor_df: pd.DataFrame, tax_order: pd.DataFrame) -> None:
    if genus_anchor_df.empty:
        ax.text(0.5, 0.5, 'No genus-level anchor table', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off(); return
    pos_map = {str(g): i for i, g in enumerate(tax_order['genus'].astype(str).tolist())}
    y_sites = {site: i for i, site in enumerate(PREFERRED_BODY_SITES[::-1])}
    for site, y in y_sites.items():
        ax.hlines(y, 0, len(pos_map)-1, color=COL['light'], lw=0.8, zorder=0)
        sub = genus_anchor_df[genus_anchor_df['body_site'].eq(site)].copy()
        if sub.empty:
            continue
        sub['position'] = sub['genus'].astype(str).map(pos_map)
        sub = sub.dropna(subset=['position']).sort_values('anchor_score', ascending=False).head(8)
        if sub.empty:
            continue
        sizes = 20 + 90 * (sub['anchor_score'] - sub['anchor_score'].min()) / (sub['anchor_score'].max() - sub['anchor_score'].min() + 1e-12)
        ax.scatter(sub['position'], np.full(len(sub), y), s=sizes, color=SITE_COLORS.get(site, 'grey'),
                   alpha=0.88, edgecolor='white', lw=0.35, zorder=3)
        top_lab = sub.head(5).sort_values('position').reset_index(drop=True)
        offsets = [0.18, 0.34, 0.20, 0.38, 0.26]
        xjit = [-2.2, -0.5, 1.4, 2.8, 4.0]
        for ii, (_, row) in enumerate(top_lab.iterrows()):
            yy = y + offsets[ii % len(offsets)]
            xx = row['position'] + xjit[ii % len(xjit)]
            ax.text(xx, yy, str(row['genus']), color=SITE_COLORS.get(site, 'grey'), fontsize=5.8,
                    rotation=28, ha='left', va='bottom')
    ax.set_xlim(0, max(1, len(pos_map)-1))
    ax.set_yticks(list(y_sites.values())); ax.set_yticklabels(list(y_sites.keys()))
    ax.set_xlabel('Ordered genus axis')
    ax.set_title('Anchor genera occupy distinct positions along the ordered genus axis')
    style_axis(ax)
    ax.grid(axis='x', color=COL['light'], lw=0.45, alpha=0.4)


def select_anchor_and_de_taxa(genus_anchor_df: pd.DataFrame,
                              clr: pd.DataFrame,
                              meta: pd.DataFrame,
                              top_n: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    anchor_rows, de_rows = [], []
    # clr is samples x genera
    feature_df = clr.copy()
    y_site = meta.loc[feature_df.index, 'body_site']
    for site in PREFERRED_BODY_SITES:
        sub = genus_anchor_df[genus_anchor_df['body_site'].eq(site)].copy()
        sub = sub.sort_values('anchor_score', ascending=False).head(top_n)
        anchor_g = [g for g in sub['genus'].astype(str).tolist() if g in feature_df.columns]
        for rank, g in enumerate(anchor_g, start=1):
            anchor_rows.append({'body_site': site, 'genus': g, 'rank': rank, 'set': 'anchor'})
        # differential taxa: one-vs-rest rank-sum on CLR, same feature number as anchors
        y = (y_site.values == site)
        stats = []
        for g in feature_df.columns.astype(str):
            vals_pos = feature_df.loc[y, g].values.astype(float)
            vals_neg = feature_df.loc[~y, g].values.astype(float)
            if len(vals_pos) < 2 or len(vals_neg) < 2:
                continue
            try:
                _, pval = mannwhitneyu(vals_pos, vals_neg, alternative='two-sided')
            except Exception:
                pval = np.nan
            eff = np.median(vals_pos) - np.median(vals_neg)
            stats.append((g, pval, eff))
        st = pd.DataFrame(stats, columns=['genus','p_value','effect']).replace([np.inf, -np.inf], np.nan).dropna()
        if st.empty:
            continue
        pvals = st['p_value'].values
        order_idx = np.argsort(pvals)
        qvals = np.empty_like(pvals, dtype=float)
        m = len(pvals)
        prev = 1.0
        for ii in range(m - 1, -1, -1):
            idx = order_idx[ii]
            q = min(prev, pvals[idx] * m / (ii + 1))
            qvals[idx] = q
            prev = q
        st['q_value'] = qvals
        n_select = max(top_n, len(anchor_g))
        st_pos = st[st['effect'] > 0].sort_values(['q_value','p_value','effect'], ascending=[True,True,False])
        if len(st_pos) < n_select:
            st_sel = pd.concat([st_pos, st[~st['genus'].isin(st_pos['genus'])].sort_values(['q_value','p_value','effect'], ascending=[True,True,False])]).head(n_select)
        else:
            st_sel = st_pos.head(n_select)
        for rank, row in enumerate(st_sel.itertuples(index=False), start=1):
            de_rows.append({'body_site': site, 'genus': row.genus, 'rank': rank, 'p_value': row.p_value,
                            'q_value': row.q_value, 'effect': row.effect, 'set': 'differential'})
    return pd.DataFrame(anchor_rows), pd.DataFrame(de_rows)


def compute_anchor_vs_de_roc(anchor_taxa_df: pd.DataFrame,
                             de_taxa_df: pd.DataFrame,
                             clr: pd.DataFrame,
                             meta: pd.DataFrame) -> Tuple[Dict[str, Dict[str, object]], pd.DataFrame]:
    # clr is samples x genera
    X_all = clr.copy()
    y_site = meta.loc[X_all.index, 'body_site']
    results = {}
    auc_rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for site in PREFERRED_BODY_SITES:
        y = (y_site.values == site).astype(int)
        results[site] = {}
        for label, tax_df in [('Anchor taxa', anchor_taxa_df), ('Differential taxa', de_taxa_df)]:
            genera = tax_df.loc[tax_df['body_site'].eq(site), 'genus'].astype(str).tolist()
            genera = [g for g in genera if g in X_all.columns]
            if len(genera) == 0 or len(np.unique(y)) < 2:
                continue
            X = X_all[genera].values.astype(float)
            y_true_all, y_prob_all = [], []
            aucs = []
            for tr, te in cv.split(X, y):
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
                clf.fit(X[tr], y[tr])
                prob = clf.predict_proba(X[te])[:, 1]
                y_true_all.extend(y[te].tolist())
                y_prob_all.extend(prob.tolist())
                fpr, tpr, _ = roc_curve(y[te], prob)
                aucs.append(auc(fpr, tpr))
            fpr_all, tpr_all, _ = roc_curve(np.array(y_true_all), np.array(y_prob_all))
            auc_all = auc(fpr_all, tpr_all)
            results[site][label] = {
                'genera': genera,
                'fpr': fpr_all,
                'tpr': tpr_all,
                'auc': auc_all,
                'cv_auc_mean': float(np.mean(aucs)),
                'cv_auc_sd': float(np.std(aucs)),
            }
            auc_rows.append({'body_site': site, 'feature_set': label, 'n_genera': len(genera),
                             'auc': auc_all, 'cv_auc_mean': float(np.mean(aucs)), 'cv_auc_sd': float(np.std(aucs)),
                             'genera': ';'.join(genera)})
    return results, pd.DataFrame(auc_rows)


def create_roc_comparison_supplementary(roc_results: Dict[str, Dict[str, object]], auc_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(12.5, 7.2))
    gs = GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.45)
    site_list = PREFERRED_BODY_SITES
    colors = {'Anchor taxa': '#8C5A3C', 'Differential taxa': '#547F9E'}
    for idx, site in enumerate(site_list):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.plot([0,1], [0,1], ls='--', lw=0.9, color=COL['muted'])
        for label in ['Anchor taxa', 'Differential taxa']:
            if site not in roc_results or label not in roc_results[site]:
                continue
            rr = roc_results[site][label]
            ax.plot(rr['fpr'], rr['tpr'], lw=1.7, color=colors[label],
                    label=f"{label} (AUC={rr['auc']:.2f})")
        ax.set_title(site)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        style_axis(ax)
        ax.legend(frameon=False, fontsize=6.0, loc='lower right')
    # summary panel
    ax = fig.add_subplot(gs[1, 2])
    if not auc_df.empty:
        piv = auc_df.pivot(index='body_site', columns='feature_set', values='auc').reindex(PREFERRED_BODY_SITES)
        x = np.arange(len(PREFERRED_BODY_SITES))
        w = 0.36
        if 'Anchor taxa' in piv.columns:
            ax.bar(x - w/2, piv['Anchor taxa'].values, width=w, color=colors['Anchor taxa'], label='Anchor taxa')
        if 'Differential taxa' in piv.columns:
            ax.bar(x + w/2, piv['Differential taxa'].values, width=w, color=colors['Differential taxa'], label='Differential taxa')
        ax.set_xticks(x); ax.set_xticklabels(PREFERRED_BODY_SITES, rotation=25, ha='right')
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel('AUC')
        ax.set_title('AUC summary')
        style_axis(ax)
        ax.legend(frameon=False, fontsize=6.2)
    fig.suptitle('Supplementary: ROC comparison of anchor genera vs differential taxa', y=0.99, fontsize=11, color=COL['dark'])
    fig.savefig(OUT_DIR / 'figure_HMP_phase_story_supplementary_anchor_vs_de_roc.png', bbox_inches='tight')
    fig.savefig(OUT_DIR / 'figure_HMP_phase_story_supplementary_anchor_vs_de_roc.pdf', bbox_inches='tight')
    plt.close(fig)


def create_figure(res: Dict[str, object]) -> None:
    fig = plt.figure(figsize=(15.2, 10.8))
    gs = GridSpec(3, 10, figure=fig, height_ratios=[1.0, 1.15, 1.2], hspace=0.9, wspace=0.85)

    ax = fig.add_subplot(gs[0, 0:4])
    panel_label(ax, 'b')
    band_stats = plot_low_high_locking(ax, res['site_stats'])
    band_stats.to_csv(OUT_DIR / 'band_ppc_pairwise_and_permutation_stats.csv', index=False)

    ax = fig.add_subplot(gs[0, 4:7])
    panel_label(ax, 'd')
    plot_score_permutation(ax, res['score_df'])

    ax = fig.add_subplot(gs[0, 7:10])
    panel_label(ax, 'i')
    plot_longitudinal(ax, res['longitudinal_dist'])

    dummy = fig.add_subplot(gs[1, :]); dummy.set_axis_off(); panel_label(dummy, 'e')
    plot_polar_modes(fig, gs[1, :], res['phase_table'], res['selected_polar_modes'], res['mode_anchor_labels'])

    ax = fig.add_subplot(gs[2, :5])
    panel_label(ax, 'g')
    plot_phylum_anchor_heatmap(ax, res['clade_anchor_df'])

    ax = fig.add_subplot(gs[2, 5:10])
    panel_label(ax, 'j')
    plot_backprojected_waves(ax, res['genus_anchor_df'], res['tax_order'])

    fig.suptitle('Taxonomically anchored body-site phase attractors', y=0.995, fontsize=12, color=COL['dark'])
    fig.savefig(OUT_DIR / 'figure_HMP_phase_story_main.png', bbox_inches='tight')
    fig.savefig(OUT_DIR / 'figure_HMP_phase_story_main.pdf', bbox_inches='tight')
    plt.close(fig)


def create_supplementary_figure(res: Dict[str, object]) -> None:
    fig = plt.figure(figsize=(8.6, 3.6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.45)
    ax = fig.add_subplot(gs[0,0])
    panel_label(ax, 'a')
    plot_design(ax, res['counts'])
    ax = fig.add_subplot(gs[0,1])
    panel_label(ax, 'f')
    plot_classification(ax, res['benchmark'])
    fig.suptitle('Supplementary: dataset design and classification benchmark', y=0.99, fontsize=10.8, color=COL['dark'])
    fig.savefig(OUT_DIR / 'figure_HMP_phase_story_supplementary_af.png', bbox_inches='tight')
    fig.savefig(OUT_DIR / 'figure_HMP_phase_story_supplementary_af.pdf', bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 8. Main analysis
# =============================================================================


def run_analysis() -> Dict[str, object]:
    print("Loading abundance and metadata...")
    abu_raw = load_abundance()
    meta_raw = load_metadata()
    abu, meta, counts = filter_and_balance(abu_raw, meta_raw)
    tax = load_taxonomy(list(abu.index))
    abu, tax_order, order_source = choose_order(abu, tax)
    print(f"Samples after balancing: {abu.shape[1]}; genera after filtering: {abu.shape[0]}; order source: {order_source}")

    clr = clr_transform_sample_by_taxon(abu)
    meta = meta.loc[clr.index].copy()
    sample_ids = list(clr.index)

    print("Computing FFT and phase statistics...")
    Z, modes, freq = compute_fft(clr)
    ptab = phase_table(Z, modes, sample_ids, meta)
    site_stats = compute_site_mode_stats(ptab)
    y = LabelEncoder().fit_transform(meta.loc[sample_ids, "body_site"].values)
    amp_assoc = amplitude_association(Z, modes, y)

    print("Computing phase-attractor scores and label permutation null...")
    score_df = label_permutation_for_scores(ptab, modes)
    score_df = score_df.merge(amp_assoc, on="mode", how="left")

    # Select modes for polar plots: high score and significant if possible, prioritizing low modes.
    cand = score_df[score_df["mode"].le(max(LOW_MAX_K, 12))].copy()
    if "label_perm_q" in cand.columns:
        cand = cand.sort_values(["label_perm_q", "phase_attractor_score"], ascending=[True, False])
    else:
        cand = cand.sort_values("phase_attractor_score", ascending=False)
    selected_polar = [k for k in [4, 5, 6] if k in modes]

    selected_back = selected_backprojection_modes(score_df)
    if PAIRWISE_MODE == "auto":
        pair_modes = selected_back if BAND_FOR_TAXON_BACKPROJECTION == "low" else selected_polar[:1]
    else:
        pair_modes = [int(PAIRWISE_MODE)]

    print(f"Selected polar modes: {selected_polar}")
    print(f"Selected backprojection modes: {selected_back}")

    print("Computing taxonomic back-projection...")
    site_complex = mean_complex_by_site(Z, modes, sample_ids, meta)
    site_waves = {
        site: reconstruct_wave_from_modes(coeff, modes, len(tax_order), selected_back)
        for site, coeff in site_complex.items()
    }
    genus_anchor_df, clade_anchor_df, peak_df = compute_taxon_anchor_tables(site_waves, clr, meta, tax_order, selected_back)
    mode_anchor_labels = compute_mode_specific_anchor_labels(site_complex, modes, clr, meta, tax_order, selected_polar)
    genus_shift_df, clade_shift_df = pairwise_phase_shift_backprojection(site_waves, clr, meta, tax_order, pair_modes)

    print("Computing classification and longitudinal attractor controls...")
    bench, cm = classification_benchmark(Z, modes, meta, sample_ids, selected_back if selected_back else list(range(1, LOW_MAX_K+1)))
    longitudinal = longitudinal_phase_distances(Z, modes, meta, sample_ids, selected_back if selected_back else list(range(1, LOW_MAX_K+1)))
    anchor_taxa_df, de_taxa_df = select_anchor_and_de_taxa(genus_anchor_df, clr, meta, top_n=8)
    roc_results, roc_auc_df = compute_anchor_vs_de_roc(anchor_taxa_df, de_taxa_df, clr, meta)

    # Statistical summaries for low vs high phase locking
    band_summary = site_stats.groupby("band").agg(
        mean_PPC=("PPC", "mean"),
        median_PPC=("PPC", "median"),
        mean_R=("R", "mean"),
        median_minus_log10_q=("minus_log10_q", "median"),
        n=("PPC", "count"),
    ).reset_index()
    low_vals = site_stats.loc[site_stats["band"].eq("low"), "PPC"].dropna().values
    high_vals = site_stats.loc[site_stats["band"].eq("high"), "PPC"].dropna().values
    if len(low_vals) and len(high_vals):
        u, p_low_gt_high = mannwhitneyu(low_vals, high_vals, alternative="greater")
    else:
        p_low_gt_high = np.nan
    band_summary["low_vs_high_mannwhitney_p"] = p_low_gt_high

    # Save outputs
    print("Saving tables...")
    counts.to_csv(OUT_DIR / "body_site_sample_counts.csv", index=False)
    tax_order.to_csv(OUT_DIR / "ordered_genera_with_taxonomy.csv", index=False)
    site_stats.to_csv(OUT_DIR / "site_mode_phase_locking_rayleigh_ppc.csv", index=False)
    score_df.to_csv(OUT_DIR / "phase_attractor_modes_with_label_permutation.csv", index=False)
    band_summary.to_csv(OUT_DIR / "low_mid_high_phase_locking_summary.csv", index=False)
    genus_anchor_df.to_csv(OUT_DIR / "backprojected_site_anchor_genera.csv", index=False)
    clade_anchor_df.to_csv(OUT_DIR / "backprojected_site_anchor_clades.csv", index=False)
    peak_df.to_csv(OUT_DIR / "backprojected_wave_peak_windows.csv", index=False)
    mode_anchor_labels.to_csv(OUT_DIR / "polar_mode_anchor_labels.csv", index=False)
    genus_shift_df.to_csv(OUT_DIR / "pairwise_phase_shift_anchor_genera.csv", index=False)
    clade_shift_df.to_csv(OUT_DIR / "pairwise_phase_shift_anchor_clades.csv", index=False)
    bench.to_csv(OUT_DIR / "phase_amplitude_classification_benchmark.csv", index=False)
    cm.to_csv(OUT_DIR / "phase_only_confusion_matrix.csv")
    longitudinal.to_csv(OUT_DIR / "longitudinal_phase_attractor_distances.csv", index=False)
    anchor_taxa_df.to_csv(OUT_DIR / "anchor_taxa_for_roc.csv", index=False)
    de_taxa_df.to_csv(OUT_DIR / "differential_taxa_for_roc.csv", index=False)
    roc_auc_df.to_csv(OUT_DIR / "anchor_vs_differential_roc_auc.csv", index=False)

    readme = f"""HMP phase attractor taxonomic back-projection outputs
=====================================================

Input abundance: {ABUNDANCE_PATH}
Input metadata: {METADATA_PATH}
Input taxonomy: {TAXONOMY_PATH if str(TAXONOMY_PATH).strip() else 'not provided'}
Input phylogeny: {PHYLOGENY_PATH if str(PHYLOGENY_PATH).strip() else 'not provided'}
Order source: {order_source}

Key files:
- figure_HMP_phase_taxon_backprojection.png/pdf
- site_mode_phase_locking_rayleigh_ppc.csv
- low_mid_high_phase_locking_summary.csv
- phase_attractor_modes_with_label_permutation.csv
- backprojected_site_anchor_genera.csv
- backprojected_site_anchor_clades.csv
- backprojected_wave_peak_windows.csv
- pairwise_phase_shift_anchor_genera.csv
- pairwise_phase_shift_anchor_clades.csv
- longitudinal_phase_attractor_distances.csv

Interpretation guide:
1. Low-frequency locking:
   Check low_mid_high_phase_locking_summary.csv. A strong law expects low modes
   to have larger PPC/Rayleigh effect than high modes.

2. Phase-attractor modes:
   Check phase_attractor_modes_with_label_permutation.csv. Strong modes have
   high phase_attractor_score and low label_perm_q.

3. Taxonomic anchors:
   backprojected_site_anchor_genera.csv lists genera where a site's locked wave
   coincides with site-enriched abundance. This is more interpretable than raw
   sine-wave peaks.

4. Clade anchors:
   backprojected_site_anchor_clades.csv summarizes positive anchor scores by
   phylum/class/order/family. Provide TAXONOMY_PATH for meaningful Firmicutes / 
   Bacteroidetes-level interpretation.

5. Pairwise shifts:
   pairwise_phase_shift_anchor_clades.csv identifies taxa/clades explaining
   phase shifts between body sites, such as Gut vs Oral or Gut vs Skin.
"""
    (OUT_DIR / "README_outputs.txt").write_text(readme, encoding="utf-8")

    return {
        "counts": counts,
        "tax_order": tax_order,
        "order_source": order_source,
        "clr": clr,
        "meta": meta,
        "Z": Z,
        "modes": modes,
        "phase_table": ptab,
        "site_stats": site_stats,
        "score_df": score_df,
        "selected_polar_modes": selected_polar,
        "selected_backprojection_modes": selected_back,
        "site_waves": site_waves,
        "genus_anchor_df": genus_anchor_df,
        "clade_anchor_df": clade_anchor_df,
        "peak_df": peak_df,
        "mode_anchor_labels": mode_anchor_labels,
        "genus_shift_df": genus_shift_df,
        "clade_shift_df": clade_shift_df,
        "benchmark": bench,
        "anchor_taxa_df": anchor_taxa_df,
        "de_taxa_df": de_taxa_df,
        "roc_results": roc_results,
        "roc_auc_df": roc_auc_df,
        "confusion_matrix": cm,
        "longitudinal_dist": longitudinal,
        "band_summary": band_summary,
    }


def main() -> None:
    res = run_analysis()
    print("Creating figure...")
    create_figure(res)
    create_supplementary_figure(res)
    create_roc_comparison_supplementary(res['roc_results'], res['roc_auc_df'])
    print(f"Done. Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
