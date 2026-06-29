from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, kruskal

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute MGnify phylogenetic spectral compressibility metrics."
    )
    parser.add_argument("--abundance", type=Path, default=Path("data/mgnify/abu.h5"))
    parser.add_argument("--abundance-key", default="genus")
    parser.add_argument("--metadata", type=Path, default=Path("data/mgnify/metadata.csv"))
    parser.add_argument("--phylogeny", type=Path, default=Path("data/phylogeny.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/04_spectral_compressibility/spectral_compressibility_metrics"),
    )
    parser.add_argument("--biome-col", default="level_3")
    parser.add_argument("--min-samples-per-biome", type=int, default=100)
    parser.add_argument("--top-n-biomes", type=int, default=8)
    parser.add_argument("--max-samples-per-biome", type=int, default=1200)
    parser.add_argument("--pseudocount", type=float, default=1e-9)
    parser.add_argument("--fmax", type=float, default=0.20)
    parser.add_argument("--energy-threshold", type=float, default=0.80)
    parser.add_argument("--richness-threshold", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def configure_matplotlib(dpi):
    mpl.rcParams.update(
        {
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
            "savefig.dpi": dpi,
            "figure.dpi": 160,
        }
    )


def style_axis(axis, grid_axis="both"):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#452A3D")
    axis.spines["bottom"].set_color("#452A3D")
    axis.tick_params(colors="#452A3D")

    if grid_axis == "y":
        axis.grid(axis="y", color="#E9E3DA", linewidth=0.65, alpha=0.85)
    elif grid_axis == "x":
        axis.grid(axis="x", color="#E9E3DA", linewidth=0.65, alpha=0.85)
    else:
        axis.grid(color="#E9E3DA", linewidth=0.65, alpha=0.85)

    axis.set_axisbelow(True)


def add_panel_label(axis, label):
    axis.text(
        -0.14,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="#452A3D",
    )


def format_p_value(value):
    if not np.isfinite(value):
        return "P=NA"
    if value < 1e-4:
        return "P<1e-4"
    if value < 0.001:
        return f"P={value:.1e}"
    return f"P={value:.3g}"


def normalize_taxon_name(value):
    value = str(value).strip()
    if value.startswith("sk__"):
        value = "k__" + value[4:]
    return value


def short_taxon_name(value):
    last = str(value).split(";")[-1]
    if "__" in last:
        last = last.split("__")[-1]
    return last if last else str(value)


def build_phylogenetic_name_map(phylogeny):
    phylogeny = phylogeny.iloc[:, 0].astype(str).str.split(";", expand=True)

    if phylogeny.shape[1] < 6:
        raise ValueError("The phylogeny table must contain semicolon-delimited taxonomic paths to genus level.")

    phylogeny.index = phylogeny[5]
    phylogeny = phylogeny[~phylogeny.index.duplicated(keep="first")]
    fullnames = phylogeny[0] + ";" + phylogeny[1] + ";" + phylogeny[2] + ";" + phylogeny[3] + ";" + phylogeny[4] + ";" + phylogeny[5]
    return fullnames


def load_mgnify_data(args):
    abundance = pd.read_hdf(args.abundance, args.abundance_key)
    metadata = pd.read_csv(args.metadata, low_memory=False)
    phylogeny = pd.read_csv(args.phylogeny, low_memory=False)

    metadata = metadata.rename(columns={"SampleID": "sample", "Env": "biome"})
    if "sample" not in metadata.columns:
        raise ValueError("Metadata must contain either 'SampleID' or 'sample'.")
    if "biome" not in metadata.columns:
        raise ValueError("Metadata must contain either 'Env' or 'biome'.")

    metadata["sample"] = metadata["sample"].astype(str)
    metadata = metadata.set_index("sample")

    environment_parts = metadata["biome"].astype(str).str.split(":", expand=True)
    for index in range(environment_parts.shape[1]):
        metadata[f"level_{index + 1}"] = environment_parts[index]

    abundance.index = abundance.index.astype(str)
    abundance.columns = [normalize_taxon_name(column) for column in abundance.columns]

    fullnames = build_phylogenetic_name_map(phylogeny)
    overlap = abundance.columns.intersection(fullnames.index)

    if len(overlap) == 0:
        raise ValueError("No overlapping taxa were found between the abundance table and phylogeny.")

    abundance = abundance.loc[:, overlap]
    abundance.columns = fullnames.loc[abundance.columns].values

    shared_samples = abundance.index.intersection(metadata.index)
    if len(shared_samples) == 0:
        raise ValueError("No overlapping samples were found between abundance and metadata.")

    abundance = abundance.loc[shared_samples]
    metadata = metadata.loc[shared_samples]

    if args.biome_col not in metadata.columns:
        raise ValueError(f"Biome column '{args.biome_col}' was not found in metadata.")

    metadata = metadata[metadata[args.biome_col].notna()]
    abundance = abundance.loc[metadata.index]

    biome_counts = metadata[args.biome_col].value_counts()
    biome_order = biome_counts[biome_counts >= args.min_samples_per_biome].index[:args.top_n_biomes].tolist()

    if len(biome_order) == 0:
        raise ValueError("No biome passed the minimum sample-size filter.")

    metadata = metadata[metadata[args.biome_col].isin(biome_order)].copy()
    abundance = abundance.loc[metadata.index].copy()

    ordered_columns = [column for column in fullnames.values if column in abundance.columns]
    abundance = abundance.loc[:, ordered_columns]
    abundance = abundance.loc[:, abundance.sum(axis=0) > 0]

    if args.max_samples_per_biome is not None and args.max_samples_per_biome > 0:
        random_generator = np.random.default_rng(args.random_state)
        retained_samples = []

        for biome in biome_order:
            sample_ids = metadata.index[metadata[args.biome_col].astype(str) == str(biome)].to_numpy()
            if len(sample_ids) > args.max_samples_per_biome:
                sample_ids = random_generator.choice(sample_ids, size=args.max_samples_per_biome, replace=False)
            retained_samples.extend(sample_ids.tolist())

        retained_samples = pd.Index(retained_samples).intersection(abundance.index)
        abundance = abundance.loc[retained_samples].copy()
        metadata = metadata.loc[retained_samples].copy()

    metadata["biome_plot"] = pd.Categorical(metadata[args.biome_col], categories=biome_order, ordered=True)

    taxa_order = pd.DataFrame(
        {
            "taxon_order": np.arange(1, len(abundance.columns) + 1),
            "taxon_fullname": abundance.columns,
            "taxon_short": [short_taxon_name(column) for column in abundance.columns],
        }
    )

    return abundance, metadata, taxa_order, biome_order


def first_fraction_reaching_threshold(cumulative_energy, threshold):
    indices = np.where(cumulative_energy >= threshold)[0]
    if len(indices) == 0:
        return 1.0
    return (indices[0] + 1) / len(cumulative_energy)


def estimate_spectral_slope(power, frequency, fmin, fmax):
    mask = (frequency >= fmin) & (frequency <= fmax) & np.isfinite(power) & (power > 0)

    if mask.sum() < 3:
        return np.nan, np.nan, np.nan

    x = np.log10(frequency[mask])
    y = np.log10(power[mask])
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual_sum = np.sum((y - fitted) ** 2)
    total_sum = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - residual_sum / total_sum if total_sum > 0 else np.nan
    return -slope, intercept, r_squared


def compute_compressibility_metrics(abundance, metadata, args):
    abundance = abundance.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    row_sums = abundance.sum(axis=1)
    abundance = abundance.loc[row_sums > 0].div(row_sums[row_sums > 0], axis=0)
    metadata = metadata.loc[abundance.index].copy()

    richness = (abundance > args.richness_threshold).sum(axis=1).astype(int)

    n_taxa = abundance.shape[1]
    window = np.hanning(n_taxa).astype(float)
    frequency = np.fft.rfftfreq(n_taxa, d=1.0)
    fmin = 2.0 / n_taxa
    frequency_mask = (frequency >= fmin) & (frequency <= args.fmax)
    active_frequency = frequency[frequency_mask]
    n_modes = int(frequency_mask.sum())

    if n_modes < 3:
        raise ValueError("Too few frequency modes were retained for compressibility analysis.")

    mode_fraction = np.arange(1, n_modes + 1) / n_modes
    records = []
    cumulative_curves = []
    sample_ids = []

    for sample_id in abundance.index:
        values = abundance.loc[sample_id].to_numpy(dtype=float)
        centered = np.log(values + args.pseudocount)
        centered = centered - centered.mean()
        transformed = centered * window
        power = (np.abs(np.fft.rfft(transformed)) ** 2) / n_taxa
        active_power = power[frequency_mask].astype(float)
        total_power = float(np.nansum(active_power))

        if total_power <= 0 or not np.isfinite(total_power):
            continue

        probability = active_power / total_power
        cumulative = np.cumsum(probability)

        c50 = first_fraction_reaching_threshold(cumulative, 0.50)
        c80 = first_fraction_reaching_threshold(cumulative, 0.80)
        c90 = first_fraction_reaching_threshold(cumulative, 0.90)

        entropy = -float(np.nansum(probability * np.log(probability + 1e-30)))
        effective_mode_number = float(np.exp(entropy))
        effective_dimension = effective_mode_number / n_modes
        normalized_entropy = entropy / np.log(n_modes)

        n10 = max(1, int(np.ceil(0.10 * n_modes)))
        n20 = max(1, int(np.ceil(0.20 * n_modes)))
        e10 = float(cumulative[n10 - 1])
        e20 = float(cumulative[n20 - 1])

        centroid = float(np.nansum(active_frequency * probability) / (active_frequency.max() + 1e-30))
        beta, intercept, r_squared = estimate_spectral_slope(power, frequency, fmin, args.fmax)

        records.append(
            {
                "sample": sample_id,
                "biome": str(metadata.loc[sample_id, "biome_plot"]),
                "richness": int(richness.loc[sample_id]),
                "log10_richness": float(np.log10(richness.loc[sample_id] + 1)),
                "C50": c50,
                "C80": c80,
                "C90": c90,
                "E10_low_order_energy": e10,
                "E20_low_order_energy": e20,
                "spectral_entropy_norm": normalized_entropy,
                "effective_spectral_dimension": effective_dimension,
                "effective_mode_number": effective_mode_number,
                "spectral_centroid_norm": centroid,
                "beta": beta,
                "beta_intercept": intercept,
                "beta_r2": r_squared,
                "n_active_modes": n_modes,
                "n_taxa": n_taxa,
                "fmin": fmin,
                "fmax": args.fmax,
            }
        )

        cumulative_curves.append(cumulative)
        sample_ids.append(sample_id)

    metrics = pd.DataFrame(records).set_index("sample")
    curves = np.vstack(cumulative_curves)

    tier_order = ["Low richness", "Mid richness", "High richness"]
    ranks = metrics["richness"].rank(method="first")
    metrics["richness_tier"] = pd.qcut(ranks, 3, labels=tier_order)
    metrics["log10_richness_resid"] = metrics["log10_richness"] - metrics.groupby("biome")["log10_richness"].transform("median")
    metrics["C80_resid"] = metrics["C80"] - metrics.groupby("biome")["C80"].transform("median")
    metrics["effective_spectral_dimension_resid"] = (
        metrics["effective_spectral_dimension"]
        - metrics.groupby("biome")["effective_spectral_dimension"].transform("median")
    )

    tier_by_sample = metrics.loc[sample_ids, "richness_tier"].astype(str).to_numpy()
    return metrics, curves, mode_fraction, tier_by_sample


def spearman_test(first_values, second_values):
    first_values = pd.Series(first_values, dtype=float)
    second_values = pd.Series(second_values, dtype=float)
    mask = first_values.notna() & second_values.notna()

    if mask.sum() < 5:
        return np.nan, np.nan

    rho, p_value = spearmanr(first_values[mask], second_values[mask])
    return float(rho), float(p_value)


def fit_line(first_values, second_values):
    x = np.asarray(first_values, dtype=float)
    y = np.asarray(second_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return None, None

    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    grid = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
    return grid, slope * grid + intercept


def make_main_figure(metrics, curves, mode_fraction, tier_by_sample, biome_order, output_dir, dpi):
    tier_order = ["Low richness", "Mid richness", "High richness"]
    tier_colors = {
        "Low richness": "#B7B5A0",
        "Mid richness": "#E5855D",
        "High richness": "#44757A",
    }
    biome_colors = {
        "Human": "#C56A48",
        "Aquatic": "#3E7897",
        "Mammals": "#A94955",
        "Terrestrial": "#9A7934",
        "Plants": "#5A8C5A",
        "Birds": "#7A3446",
        "Animal": "#B95B7A",
        "Wastewater": "#6F6087",
    }

    figure = plt.figure(figsize=(10.4, 6.8))
    grid = figure.add_gridspec(2, 3, wspace=0.34, hspace=0.47)

    axis = figure.add_subplot(grid[0, 0])
    add_panel_label(axis, "a")
    for tier in tier_order:
        indices = np.where(tier_by_sample == tier)[0]
        if len(indices) == 0:
            continue
        subset = curves[indices, :]
        median = np.nanmedian(subset, axis=0)
        q1 = np.nanquantile(subset, 0.25, axis=0)
        q3 = np.nanquantile(subset, 0.75, axis=0)
        axis.plot(mode_fraction, median, color=tier_colors[tier], linewidth=2.0, label=tier)
        axis.fill_between(mode_fraction, q1, q3, color=tier_colors[tier], alpha=0.14, linewidth=0)
    axis.axhline(0.80, color="#452A3D", linestyle="--", linewidth=0.8, alpha=0.55)
    axis.text(0.96, 0.825, "80% energy", ha="right", va="bottom", fontsize=6.6, color="#452A3D")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Fraction of frequency modes\naccumulated from low to high")
    axis.set_ylabel("Cumulative spectral energy")
    axis.set_title("Emergent spectral compressibility")
    axis.legend(frameon=False, loc="lower right", handlelength=1.8)
    style_axis(axis)

    axis = figure.add_subplot(grid[0, 1])
    add_panel_label(axis, "b")
    for biome in biome_order:
        subset = metrics[metrics["biome"] == biome]
        axis.scatter(
            subset["richness"],
            subset["C80"],
            s=14,
            c=biome_colors.get(biome, "#44757A"),
            alpha=0.28,
            edgecolor="none",
        )
    grid_x, fitted = fit_line(metrics["richness"], metrics["C80"])
    if grid_x is not None:
        axis.plot(grid_x, fitted, color="#452A3D", linewidth=1.5)
    rho_all, p_all = spearman_test(metrics["richness"], metrics["C80"])
    rho_res, p_res = spearman_test(metrics["log10_richness_resid"], metrics["C80_resid"])
    axis.text(
        0.03,
        0.96,
        f"Across samples: ρ={rho_all:.2f}, {format_p_value(p_all)}\nWithin-biome residuals: ρ={rho_res:.2f}, {format_p_value(p_res)}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        color="#452A3D",
    )
    axis.set_xlabel("Taxonomic richness\n(observed genera)")
    axis.set_ylabel("C80 compressibility index")
    axis.set_title("Richness and spectral compressibility")
    style_axis(axis)

    axis = figure.add_subplot(grid[0, 2])
    add_panel_label(axis, "c")
    for biome in biome_order:
        subset = metrics[metrics["biome"] == biome]
        axis.scatter(
            subset["log10_richness_resid"],
            subset["C80_resid"],
            s=14,
            c=biome_colors.get(biome, "#44757A"),
            alpha=0.28,
            edgecolor="none",
        )
    grid_x, fitted = fit_line(metrics["log10_richness_resid"], metrics["C80_resid"])
    if grid_x is not None:
        axis.plot(grid_x, fitted, color="#452A3D", linewidth=1.5)
    axis.axhline(0, color="#452A3D", linewidth=0.75, linestyle=":", alpha=0.55)
    axis.axvline(0, color="#452A3D", linewidth=0.75, linestyle=":", alpha=0.55)
    axis.text(
        0.03,
        0.96,
        f"Biome-centered:\nρ={rho_res:.2f}, {format_p_value(p_res)}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        color="#452A3D",
    )
    axis.set_xlabel("Richness residual\nwithin biome")
    axis.set_ylabel("C80 residual\nwithin biome")
    axis.set_title("Within-biome relationship")
    style_axis(axis)

    axis = figure.add_subplot(grid[1, 0])
    add_panel_label(axis, "d")
    positions = np.arange(1, 4)
    data = [
        metrics.loc[metrics["richness_tier"].astype(str) == tier, "C80"].dropna().to_numpy()
        for tier in tier_order
    ]
    box = axis.boxplot(
        data,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#452A3D", "linewidth": 1.2},
        boxprops={"color": "#452A3D", "linewidth": 0.9},
        whiskerprops={"color": "#452A3D", "linewidth": 0.9},
        capprops={"color": "#452A3D", "linewidth": 0.9},
    )
    for patch, tier in zip(box["boxes"], tier_order):
        patch.set_facecolor(tier_colors[tier])
        patch.set_alpha(0.42)
    _, p_kw = kruskal(*data)
    axis.text(0.03, 0.96, f"Kruskal {format_p_value(p_kw)}", transform=axis.transAxes, ha="left", va="top", fontsize=6.7)
    axis.text(0.97, 0.96, "lower = more compressible", transform=axis.transAxes, ha="right", va="top", fontsize=6.7, color="#7E746B")
    axis.set_xticks(positions)
    axis.set_xticklabels(["Low", "Mid", "High"])
    axis.set_xlabel("Richness tier")
    axis.set_ylabel("C80: modes required\nfor 80% spectral energy")
    axis.set_title("High-richness communities need fewer modes")
    axis.set_ylim(0.5, 1.0)
    style_axis(axis, grid_axis="y")

    axis = figure.add_subplot(grid[1, 1])
    add_panel_label(axis, "e")
    data_deff = [
        metrics.loc[metrics["richness_tier"].astype(str) == tier, "effective_spectral_dimension"].dropna().to_numpy()
        for tier in tier_order
    ]
    box = axis.boxplot(
        data_deff,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#452A3D", "linewidth": 1.2},
        boxprops={"color": "#452A3D", "linewidth": 0.9},
        whiskerprops={"color": "#452A3D", "linewidth": 0.9},
        capprops={"color": "#452A3D", "linewidth": 0.9},
    )
    for patch, tier in zip(box["boxes"], tier_order):
        patch.set_facecolor(tier_colors[tier])
        patch.set_alpha(0.42)
    _, p_kw_deff = kruskal(*data_deff)
    rho_deff, p_deff = spearman_test(metrics["richness"], metrics["effective_spectral_dimension"])
    axis.text(
        0.03,
        0.96,
        f"Kruskal {format_p_value(p_kw_deff)}\nρ richness={rho_deff:.2f}, {format_p_value(p_deff)}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(["Low", "Mid", "High"])
    axis.set_xlabel("Richness tier")
    axis.set_ylabel("Effective spectral dimension\n(normalized mode number)")
    axis.set_title("Frequency-domain dimensionality")
    axis.set_ylim(0.5, 1.0)
    style_axis(axis, grid_axis="y")

    axis = figure.add_subplot(grid[1, 2])
    add_panel_label(axis, "f")
    biome_statistics = []

    for biome in biome_order:
        subset = metrics[metrics["biome"] == biome]
        rho, p_value = spearman_test(subset["richness"], subset["C80"])
        biome_statistics.append({"biome": biome, "rho": rho, "p": p_value})

    biome_statistics = pd.DataFrame(biome_statistics)
    y_positions = np.arange(len(biome_order))[::-1]

    for y_position, biome in zip(y_positions, biome_order):
        row = biome_statistics[biome_statistics["biome"] == biome].iloc[0]
        axis.scatter(
            row["rho"],
            y_position,
            s=50,
            c=biome_colors.get(biome, "#44757A"),
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        axis.text(row["rho"] + 0.03, y_position, format_p_value(row["p"]), va="center", ha="left", fontsize=6.1, color="#7E746B")

    axis.axvline(0, color="#452A3D", linewidth=0.8, linestyle=":", alpha=0.65)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(biome_order)
    axis.set_xlim(-0.8, 0.45)
    axis.set_xlabel("Spearman ρ\nrichness vs C80")
    axis.set_title("Trends recur across biomes")
    style_axis(axis, grid_axis="x")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=biome_colors.get(biome, "#44757A"),
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5.6,
            label=biome,
        )
        for biome in biome_order
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=4,
        frameon=False,
        columnspacing=0.95,
        handletextpad=0.35,
    )

    figure.suptitle("Emergent spectral compressibility in microbial ecosystems", x=0.5, y=1.01, fontsize=11.2)
    figure.savefig(output_dir / "spectral_compressibility_metrics.png", bbox_inches="tight", dpi=dpi)
    figure.savefig(output_dir / "spectral_compressibility_metrics.pdf", bbox_inches="tight")
    plt.close(figure)

    return biome_statistics


def make_low_order_energy_figure(metrics, output_dir, dpi):
    tier_order = ["Low richness", "Mid richness", "High richness"]
    tier_colors = {
        "Low richness": "#B7B5A0",
        "Mid richness": "#E5855D",
        "High richness": "#44757A",
    }

    figure = plt.figure(figsize=(7.0, 3.2))
    grid = figure.add_gridspec(1, 2, wspace=0.35)
    variables = [
        ("E10_low_order_energy", "Energy captured by\nlowest 10% modes", "E10"),
        ("E20_low_order_energy", "Energy captured by\nlowest 20% modes", "E20"),
    ]

    for index, (column, ylabel, title_label) in enumerate(variables):
        axis = figure.add_subplot(grid[0, index])
        add_panel_label(axis, chr(ord("a") + index))
        positions = np.arange(1, 4)
        data = [
            metrics.loc[metrics["richness_tier"].astype(str) == tier, column].dropna().to_numpy()
            for tier in tier_order
        ]
        box = axis.boxplot(
            data,
            positions=positions,
            widths=0.56,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#452A3D", "linewidth": 1.2},
            boxprops={"color": "#452A3D", "linewidth": 0.9},
            whiskerprops={"color": "#452A3D", "linewidth": 0.9},
            capprops={"color": "#452A3D", "linewidth": 0.9},
        )
        for patch, tier in zip(box["boxes"], tier_order):
            patch.set_facecolor(tier_colors[tier])
            patch.set_alpha(0.42)
        _, p_value = kruskal(*data)
        axis.text(0.03, 0.96, f"Kruskal {format_p_value(p_value)}", transform=axis.transAxes, ha="left", va="top", fontsize=6.7)
        axis.set_xticks(positions)
        axis.set_xticklabels(["Low", "Mid", "High"])
        axis.set_xlabel("Richness tier")
        axis.set_ylabel(ylabel)
        axis.set_title(f"Low-order energy concentration ({title_label})")
        style_axis(axis, grid_axis="y")

    figure.savefig(output_dir / "low_order_energy_concentration.png", bbox_inches="tight", dpi=dpi)
    figure.savefig(output_dir / "low_order_energy_concentration.pdf", bbox_inches="tight")
    plt.close(figure)


def make_difference_curve_figure(curves, mode_fraction, tier_by_sample, output_dir, dpi):
    tier_order = ["Low richness", "Mid richness", "High richness"]
    difference_colors = {
        "High - Low": "#D44C3C",
        "High - Mid": "#B66065",
        "Mid - Low": "#44757A",
    }

    tier_curves = {}
    for tier in tier_order:
        indices = np.where(tier_by_sample == tier)[0]
        tier_curves[tier] = np.nanmedian(curves[indices, :], axis=0)

    difference_map = {
        "High - Low": tier_curves["High richness"] - tier_curves["Low richness"],
        "High - Mid": tier_curves["High richness"] - tier_curves["Mid richness"],
        "Mid - Low": tier_curves["Mid richness"] - tier_curves["Low richness"],
    }

    figure, axis = plt.subplots(figsize=(5.4, 3.6))

    for label, difference in difference_map.items():
        axis.plot(mode_fraction, difference, linewidth=2.0, color=difference_colors[label], label=label)

    axis.axhline(0, color="#452A3D", linewidth=0.85, linestyle=":", alpha=0.75)
    axis.axvline(0.10, color="#7E746B", linewidth=0.8, linestyle="--", alpha=0.6)
    axis.axvline(0.20, color="#7E746B", linewidth=0.8, linestyle="--", alpha=0.6)
    axis.text(0.10, 0.98, "10% modes", transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=6.3, color="#7E746B")
    axis.text(0.20, 0.98, "20% modes", transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=6.3, color="#7E746B")
    axis.set_xlim(0, 1)
    axis.set_xlabel("Fraction of frequency modes")
    axis.set_ylabel("Difference in cumulative energy")
    axis.set_title("Cumulative-energy differences among richness tiers")
    axis.legend(frameon=False, loc="best", handlelength=2.0)
    style_axis(axis)
    figure.savefig(output_dir / "richness_tier_difference_curves.png", bbox_inches="tight", dpi=dpi)
    figure.savefig(output_dir / "richness_tier_difference_curves.pdf", bbox_inches="tight")
    plt.close(figure)


def write_outputs(metrics, taxa_order, biome_statistics, output_dir, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.reset_index().to_csv(output_dir / "spectral_compressibility_metrics.csv", index=False)
    taxa_order.to_csv(output_dir / "taxa_phylogenetic_order.csv", index=False)
    biome_statistics.to_csv(output_dir / "biome_specific_richness_c80_correlations.csv", index=False)

    parameter_table = pd.DataFrame(
        {
            "parameter": [
                "abundance_key",
                "biome_col",
                "min_samples_per_biome",
                "top_n_biomes",
                "max_samples_per_biome",
                "pseudocount",
                "fmax",
                "energy_threshold",
                "richness_threshold",
                "random_state",
            ],
            "value": [
                args.abundance_key,
                args.biome_col,
                args.min_samples_per_biome,
                args.top_n_biomes,
                args.max_samples_per_biome,
                args.pseudocount,
                args.fmax,
                args.energy_threshold,
                args.richness_threshold,
                args.random_state,
            ],
        }
    )
    parameter_table.to_csv(output_dir / "analysis_parameters.csv", index=False)


def main():
    args = parse_arguments()
    configure_matplotlib(args.dpi)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    abundance, metadata, taxa_order, biome_order = load_mgnify_data(args)
    metrics, curves, mode_fraction, tier_by_sample = compute_compressibility_metrics(abundance, metadata, args)

    biome_statistics = make_main_figure(
        metrics=metrics,
        curves=curves,
        mode_fraction=mode_fraction,
        tier_by_sample=tier_by_sample,
        biome_order=biome_order,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    make_low_order_energy_figure(metrics, args.output_dir, args.dpi)
    make_difference_curve_figure(curves, mode_fraction, tier_by_sample, args.output_dir, args.dpi)
    write_outputs(metrics, taxa_order, biome_statistics, args.output_dir, args)

    print(f"Samples retained: {metrics.shape[0]}")
    print(f"Taxa retained: {int(metrics['n_taxa'].iloc[0])}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
