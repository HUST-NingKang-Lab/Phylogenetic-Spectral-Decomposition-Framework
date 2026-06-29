from pathlib import Path
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute phylogenetic spectral slopes for antibiotic perturbation and recovery analysis."
    )
    parser.add_argument(
        "--abundance",
        type=Path,
        default=Path("data/palleja/abundance_phylogeny_ordered.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/palleja/metadata.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/02_host_stability_transitions/antibiotic_perturbation_spectral_slope"),
    )
    parser.add_argument("--sample-col", default="sample_id")
    parser.add_argument("--group-col", default="label")
    parser.add_argument("--study-col", default="study_name")
    parser.add_argument(
        "--group-order",
        nargs="+",
        default=["pre_stable", "perturbation_unstable", "recovery_stable"],
    )
    parser.add_argument("--pseudocount", type=float, default=1e-9)
    parser.add_argument("--fmax", type=float, default=0.20)
    parser.add_argument("--min-samples-per-group", type=int, default=1)
    return parser.parse_args()


def configure_matplotlib():
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["ytick.major.size"] = 3
    plt.rcParams["savefig.dpi"] = 300


def cliffs_delta(first_values, second_values):
    first_values = np.asarray(first_values, dtype=float)
    second_values = np.asarray(second_values, dtype=float)

    if len(first_values) == 0 or len(second_values) == 0:
        return np.nan

    greater = np.sum(first_values[:, None] > second_values[None, :])
    smaller = np.sum(first_values[:, None] < second_values[None, :])
    return (greater - smaller) / (len(first_values) * len(second_values))


def estimate_spectral_slope(values, frequency, window, fmin, fmax):
    n_taxa = len(values)
    transformed = np.asarray(values, dtype=float) * window
    power = (np.abs(np.fft.rfft(transformed)) ** 2) / n_taxa
    mask = (frequency >= fmin) & (frequency <= fmax) & np.isfinite(power) & (power > 0)

    if mask.sum() < 3:
        return power, np.nan, np.nan, np.nan

    x = np.log10(frequency[mask])
    y = np.log10(power[mask])
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual_sum = np.sum((y - fitted) ** 2)
    total_sum = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - residual_sum / total_sum if total_sum > 0 else np.nan
    beta = -slope
    return power, float(beta), float(intercept), float(r_squared)


def load_abundance_table(path, sample_column):
    abundance = pd.read_csv(path, low_memory=False)

    if sample_column not in abundance.columns:
        raise ValueError(f"Sample column '{sample_column}' was not found in the abundance table.")

    abundance[sample_column] = abundance[sample_column].astype(str)
    abundance = abundance.set_index(sample_column)
    abundance = abundance.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return abundance


def load_metadata(path, sample_column, group_column, group_order):
    metadata = pd.read_csv(path, low_memory=False)

    if sample_column not in metadata.columns:
        raise ValueError(f"Sample column '{sample_column}' was not found in the metadata table.")
    if group_column not in metadata.columns:
        raise ValueError(f"Group column '{group_column}' was not found in the metadata table.")

    metadata[sample_column] = metadata[sample_column].astype(str)
    metadata = metadata.set_index(sample_column)
    metadata = metadata[metadata[group_column].isin(group_order)].copy()
    return metadata


def align_data(abundance, metadata, group_column, group_order, min_samples_per_group):
    shared_samples = abundance.index.intersection(metadata.index)

    if len(shared_samples) == 0:
        raise ValueError("No overlapping samples were found between abundance and metadata tables.")

    abundance = abundance.loc[shared_samples].copy()
    metadata = metadata.loc[shared_samples].copy()

    group_counts = metadata[group_column].value_counts()
    valid_groups = [
        group for group in group_order
        if group in group_counts.index and group_counts[group] >= min_samples_per_group
    ]

    if len(valid_groups) == 0:
        raise ValueError("No groups passed the minimum sample-size filter.")

    metadata = metadata[metadata[group_column].isin(valid_groups)].copy()
    abundance = abundance.loc[metadata.index].copy()

    row_sums = abundance.sum(axis=1)
    abundance = abundance.loc[row_sums > 0].div(row_sums[row_sums > 0], axis=0)
    metadata = metadata.loc[abundance.index].copy()
    abundance = abundance.loc[:, abundance.sum(axis=0) > 0].copy()

    if abundance.shape[1] < 4:
        raise ValueError(f"Too few taxa remained after filtering: {abundance.shape[1]}.")

    return abundance, metadata, valid_groups


def centered_log_ratio(abundance, pseudocount):
    abundance = abundance + pseudocount
    log_abundance = np.log(abundance)
    return log_abundance.sub(log_abundance.mean(axis=1), axis=0)


def compute_sample_spectra(clr_abundance, metadata, group_column, study_column, fmax):
    n_taxa = clr_abundance.shape[1]
    window = np.hanning(n_taxa)
    frequency = np.fft.rfftfreq(n_taxa, d=1.0)
    fmin = 2.0 / n_taxa
    frequency_mask = (frequency >= fmin) & (frequency <= fmax)

    if frequency_mask.sum() < 3:
        raise ValueError(
            f"Too few frequency bins in the fitting range. n_taxa={n_taxa}, fmin={fmin}, fmax={fmax}."
        )

    records = []
    power_rows = []

    for sample_id in clr_abundance.index:
        power, beta, intercept, r_squared = estimate_spectral_slope(
            clr_abundance.loc[sample_id].values,
            frequency=frequency,
            window=window,
            fmin=fmin,
            fmax=fmax,
        )
        power_rows.append(power)
        records.append(
            {
                "sample_id": sample_id,
                "group": metadata.loc[sample_id, group_column],
                "study_name": metadata.loc[sample_id, study_column] if study_column in metadata.columns else "NA",
                "beta": beta,
                "intercept": intercept,
                "r2": r_squared,
            }
        )

    slope_table = pd.DataFrame(records)
    power_table = pd.DataFrame(np.vstack(power_rows), index=clr_abundance.index, columns=frequency)
    return slope_table, power_table, frequency, frequency_mask, fmin


def summarize_group_spectra(power_table, slope_table, frequency, frequency_mask, group_order):
    selected_frequency = frequency[frequency_mask]
    indexed_slopes = slope_table.set_index("sample_id")
    records = []

    for group in group_order:
        sample_ids = indexed_slopes.index[indexed_slopes["group"] == group]

        if len(sample_ids) == 0:
            continue

        group_power = power_table.loc[sample_ids, selected_frequency]
        median = group_power.median(axis=0)
        lower_quartile = group_power.quantile(0.25, axis=0)
        upper_quartile = group_power.quantile(0.75, axis=0)

        fit_x = np.log10(selected_frequency)
        fit_y = np.log10(median.values)
        slope, intercept = np.polyfit(fit_x, fit_y, 1)

        records.append(
            pd.DataFrame(
                {
                    "group": group,
                    "frequency": selected_frequency,
                    "median_power": median.values,
                    "q1_power": lower_quartile.values,
                    "q3_power": upper_quartile.values,
                    "group_beta": -slope,
                    "group_intercept": intercept,
                }
            )
        )

    if len(records) == 0:
        return pd.DataFrame()

    return pd.concat(records, axis=0, ignore_index=True)


def summarize_group_slopes(slope_table, group_order):
    records = []

    for group in group_order:
        values = slope_table.loc[slope_table["group"] == group, "beta"].dropna().to_numpy()
        records.append(
            {
                "group": group,
                "n": len(values),
                "median_beta": np.median(values) if len(values) else np.nan,
                "mean_beta": np.mean(values) if len(values) else np.nan,
                "sd_beta": np.std(values, ddof=1) if len(values) > 1 else np.nan,
            }
        )

    return pd.DataFrame(records)


def compare_group_slopes(slope_table, group_order):
    group_values = {
        group: slope_table.loc[slope_table["group"] == group, "beta"].dropna().to_numpy()
        for group in group_order
    }

    non_empty_groups = [group for group in group_order if len(group_values[group]) > 0]

    if len(non_empty_groups) >= 2:
        kruskal_statistic, kruskal_p = kruskal(*[group_values[group] for group in non_empty_groups])
    else:
        kruskal_statistic, kruskal_p = np.nan, np.nan

    records = []

    for first_group, second_group in itertools.combinations(non_empty_groups, 2):
        first_values = group_values[first_group]
        second_values = group_values[second_group]
        p_value = mannwhitneyu(first_values, second_values, alternative="two-sided").pvalue
        effect_size = cliffs_delta(first_values, second_values)

        records.append(
            {
                "group1": first_group,
                "group2": second_group,
                "n1": len(first_values),
                "n2": len(second_values),
                "median1": np.median(first_values),
                "median2": np.median(second_values),
                "mean1": np.mean(first_values),
                "mean2": np.mean(second_values),
                "cliffs_delta_group1_vs_group2": effect_size,
                "mannwhitney_p": p_value,
            }
        )

    pairwise_table = pd.DataFrame(records)

    if not pairwise_table.empty:
        pairwise_table["p_adj_bonferroni"] = np.minimum(
            pairwise_table["mannwhitney_p"] * len(pairwise_table),
            1.0,
        )

    omnibus_table = pd.DataFrame(
        {
            "test": ["kruskal_wallis"],
            "statistic": [kruskal_statistic],
            "p_value": [kruskal_p],
        }
    )

    return omnibus_table, pairwise_table


def plot_spectral_slope_summary(output_dir, slope_table, group_spectra, group_order, omnibus_table):
    labels = {
        "pre_stable": "Pre-stable",
        "perturbation_unstable": "Perturbation-unstable",
        "recovery_stable": "Recovery-stable",
    }
    palette = {
        "pre_stable": "#4C78A8",
        "perturbation_unstable": "#D04A36",
        "recovery_stable": "#55A868",
    }

    available_groups = [group for group in group_order if group in slope_table["group"].unique()]
    figure = plt.figure(figsize=(13.5, 4.2))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.32)

    spectrum_axis = figure.add_subplot(grid[0, 0])

    for group in available_groups:
        subset = group_spectra[group_spectra["group"] == group]

        if subset.empty:
            continue

        frequency = subset["frequency"].to_numpy()
        median = subset["median_power"].to_numpy()
        lower_quartile = subset["q1_power"].to_numpy()
        upper_quartile = subset["q3_power"].to_numpy()
        beta = subset["group_beta"].iloc[0]
        intercept = subset["group_intercept"].iloc[0]

        color = palette.get(group, "#333333")
        label = labels.get(group, group)

        spectrum_axis.fill_between(frequency, lower_quartile, upper_quartile, color=color, alpha=0.16, linewidth=0)
        spectrum_axis.plot(frequency, median, color=color, linewidth=2.2, label=f"{label} β={beta:.2f}")
        fitted = 10 ** (intercept - beta * np.log10(frequency))
        spectrum_axis.plot(frequency, fitted, color=color, linewidth=1.2, linestyle="--", alpha=0.9)

    spectrum_axis.set_xscale("log")
    spectrum_axis.set_yscale("log")
    spectrum_axis.set_xlabel("Phylogenetic frequency")
    spectrum_axis.set_ylabel("Power spectral density")
    spectrum_axis.set_title("Group-level phylogenetic spectra")
    spectrum_axis.legend(frameon=False, loc="upper right", fontsize=9, handlelength=2.6)
    spectrum_axis.spines["top"].set_visible(False)
    spectrum_axis.spines["right"].set_visible(False)

    slope_axis = figure.add_subplot(grid[0, 1])
    positions = np.arange(1, len(available_groups) + 1)
    values_by_group = [
        slope_table.loc[slope_table["group"] == group, "beta"].dropna().to_numpy()
        for group in available_groups
    ]

    violins = slope_axis.violinplot(
        values_by_group,
        positions=positions,
        widths=0.8,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for index, body in enumerate(violins["bodies"]):
        body.set_facecolor(palette.get(available_groups[index], "#333333"))
        body.set_edgecolor("none")
        body.set_alpha(0.35)

    slope_axis.boxplot(
        values_by_group,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1},
        whiskerprops={"color": "black", "linewidth": 1},
        capprops={"color": "black", "linewidth": 1},
    )

    random_generator = np.random.default_rng(42)

    for index, group in enumerate(available_groups, start=1):
        values = slope_table.loc[slope_table["group"] == group, "beta"].dropna().to_numpy()
        jittered_x = random_generator.normal(loc=index, scale=0.06, size=len(values))
        slope_axis.scatter(
            jittered_x,
            values,
            s=14,
            alpha=0.55,
            color=palette.get(group, "#333333"),
            edgecolors="none",
        )

    slope_axis.set_xticks(positions)
    slope_axis.set_xticklabels([labels.get(group, group) for group in available_groups], rotation=15, ha="right")
    slope_axis.set_ylabel("Spectral slope (β)")
    slope_axis.set_title("Sample-level slope distribution")
    slope_axis.spines["top"].set_visible(False)
    slope_axis.spines["right"].set_visible(False)

    p_value = omnibus_table.loc[0, "p_value"] if not omnibus_table.empty else np.nan
    p_label = f"Kruskal–Wallis P={p_value:.2e}" if np.isfinite(p_value) else "Kruskal–Wallis P=NA"

    slope_axis.text(
        0.02,
        0.98,
        p_label,
        transform=slope_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    for axis, label in zip([spectrum_axis, slope_axis], ["a", "b"]):
        axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontsize=13, fontweight="bold", va="bottom")

    figure.savefig(output_dir / "antibiotic_perturbation_spectral_slope.png", bbox_inches="tight")
    figure.savefig(output_dir / "antibiotic_perturbation_spectral_slope.pdf", bbox_inches="tight")
    plt.close(figure)


def write_outputs(
    output_dir,
    abundance,
    slope_table,
    group_spectra,
    group_summary,
    omnibus_table,
    pairwise_table,
    sample_column,
    group_column,
    study_column,
    pseudocount,
    fmin,
    fmax,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    taxa_table = pd.DataFrame(
        {
            "taxon_order": np.arange(1, len(abundance.columns) + 1),
            "taxon_name": abundance.columns,
        }
    )

    taxa_table.to_csv(output_dir / "taxa_phylogenetic_order.csv", index=False)
    slope_table.to_csv(output_dir / "spectral_slope_per_sample.csv", index=False)
    group_spectra.to_csv(output_dir / "group_spectra_summary.csv", index=False)
    group_summary.to_csv(output_dir / "group_spectral_slope_summary.csv", index=False)
    omnibus_table.to_csv(output_dir / "group_spectral_slope_omnibus_test.csv", index=False)

    if not pairwise_table.empty:
        pairwise_table.to_csv(output_dir / "pairwise_group_spectral_slope_tests.csv", index=False)

    parameter_table = pd.DataFrame(
        {
            "parameter": [
                "sample_column",
                "group_column",
                "study_column",
                "pseudocount",
                "fmin",
                "fmax",
                "n_samples",
                "n_taxa",
                "taxonomy_level",
                "input_taxon_order",
            ],
            "value": [
                sample_column,
                group_column,
                study_column,
                pseudocount,
                fmin,
                fmax,
                abundance.shape[0],
                abundance.shape[1],
                "genus_or_species_resolved",
                "preordered",
            ],
        }
    )
    parameter_table.to_csv(output_dir / "analysis_parameters.csv", index=False)


def main():
    args = parse_arguments()
    configure_matplotlib()

    abundance = load_abundance_table(args.abundance, args.sample_col)
    metadata = load_metadata(args.metadata, args.sample_col, args.group_col, args.group_order)
    abundance, metadata, valid_groups = align_data(
        abundance=abundance,
        metadata=metadata,
        group_column=args.group_col,
        group_order=args.group_order,
        min_samples_per_group=args.min_samples_per_group,
    )

    clr_abundance = centered_log_ratio(abundance, args.pseudocount)

    slope_table, power_table, frequency, frequency_mask, fmin = compute_sample_spectra(
        clr_abundance=clr_abundance,
        metadata=metadata,
        group_column=args.group_col,
        study_column=args.study_col,
        fmax=args.fmax,
    )

    group_spectra = summarize_group_spectra(
        power_table=power_table,
        slope_table=slope_table,
        frequency=frequency,
        frequency_mask=frequency_mask,
        group_order=valid_groups,
    )

    group_summary = summarize_group_slopes(slope_table=slope_table, group_order=valid_groups)
    omnibus_table, pairwise_table = compare_group_slopes(slope_table=slope_table, group_order=valid_groups)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_spectral_slope_summary(
        output_dir=args.output_dir,
        slope_table=slope_table,
        group_spectra=group_spectra,
        group_order=valid_groups,
        omnibus_table=omnibus_table,
    )

    write_outputs(
        output_dir=args.output_dir,
        abundance=abundance,
        slope_table=slope_table,
        group_spectra=group_spectra,
        group_summary=group_summary,
        omnibus_table=omnibus_table,
        pairwise_table=pairwise_table,
        sample_column=args.sample_col,
        group_column=args.group_col,
        study_column=args.study_col,
        pseudocount=args.pseudocount,
        fmin=fmin,
        fmax=args.fmax,
    )

    print(f"Samples retained: {abundance.shape[0]}")
    print(f"Taxa retained: {abundance.shape[1]}")
    print(f"Groups retained: {', '.join(valid_groups)}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
