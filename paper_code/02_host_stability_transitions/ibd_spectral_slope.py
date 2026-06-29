from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--abundance", default="data/ibd/abundance.csv")
    parser.add_argument("--metadata", default="data/ibd/metadata.csv")
    parser.add_argument("--phylogeny", default="data/phylogeny.csv")
    parser.add_argument("--output-dir", default="outputs/02_host_stability_transitions/ibd_spectral_slope")
    parser.add_argument("--sample-col", default="sample_id")
    parser.add_argument("--group-col", default="label")
    parser.add_argument("--study-col", default="study_name")
    parser.add_argument("--healthy-label", default="healthy")
    parser.add_argument("--disease-label", default="disease")
    parser.add_argument("--healthy-name", default="Healthy")
    parser.add_argument("--disease-name", default="IBD")
    parser.add_argument("--pseudocount", type=float, default=1e-9)
    parser.add_argument("--fmax", type=float, default=0.20)
    parser.add_argument("--min-samples-per-group", type=int, default=30)
    parser.add_argument("--max-points", type=int, default=600)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def configure_plotting():
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
    sns.set_style("white")


def standardize_taxon_name(value):
    name = str(value).strip().replace("|", ";")
    if name.startswith("sk__"):
        name = "k__" + name[4:]
    return name


def collapse_to_genus(value):
    name = standardize_taxon_name(value)
    parts = [part for part in name.split(";") if part]
    retained = []
    for part in parts:
        if part.startswith("s__"):
            break
        retained.append(part)
        if part.startswith("g__"):
            break
    return ";".join(retained)


def load_abundance_table(path, sample_col):
    abundance = pd.read_csv(path, low_memory=False)
    if sample_col not in abundance.columns:
        raise ValueError(f"{sample_col} not found in abundance table")
    abundance[sample_col] = abundance[sample_col].astype(str)
    abundance = abundance.set_index(sample_col)
    abundance = abundance.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    abundance.columns = [collapse_to_genus(column) for column in abundance.columns]
    abundance = abundance.loc[:, abundance.columns.astype(str) != ""]
    abundance = abundance.T.groupby(level=0).sum().T
    return abundance


def load_metadata(path, sample_col, group_col, healthy_label, disease_label):
    metadata = pd.read_csv(path, low_memory=False)
    if sample_col not in metadata.columns:
        raise ValueError(f"{sample_col} not found in metadata")
    if group_col not in metadata.columns:
        raise ValueError(f"{group_col} not found in metadata")
    metadata[sample_col] = metadata[sample_col].astype(str)
    metadata = metadata.set_index(sample_col)
    metadata[group_col] = metadata[group_col].astype(str)
    metadata = metadata[metadata[group_col].isin([healthy_label, disease_label])].copy()
    return metadata


def load_phylogeny_order(path):
    phylogeny = pd.read_csv(path, low_memory=False)
    taxon_col = phylogeny.columns[0]
    order = phylogeny[taxon_col].astype(str).map(collapse_to_genus).drop_duplicates().tolist()
    return [taxon for taxon in order if taxon]


def align_taxa_to_phylogeny(abundance, phylogeny_order):
    overlap = abundance.columns.intersection(phylogeny_order)
    if len(overlap) == 0:
        raise ValueError("No overlapping taxa between abundance table and phylogeny order")
    abundance = abundance.loc[:, overlap]
    ordered_taxa = [taxon for taxon in phylogeny_order if taxon in abundance.columns]
    abundance = abundance[ordered_taxa]
    abundance = abundance.loc[:, abundance.sum(axis=0) > 0]
    return abundance


def align_samples(abundance, metadata, group_col, min_samples_per_group):
    shared_samples = abundance.index.intersection(metadata.index)
    abundance = abundance.loc[shared_samples]
    metadata = metadata.loc[shared_samples]
    group_counts = metadata[group_col].value_counts()
    retained_groups = group_counts[group_counts >= min_samples_per_group].index.tolist()
    metadata = metadata[metadata[group_col].isin(retained_groups)]
    abundance = abundance.loc[metadata.index]
    if metadata[group_col].nunique() < 2:
        raise ValueError("Fewer than two groups remain after filtering")
    return abundance, metadata


def relative_abundance(abundance):
    row_sums = abundance.sum(axis=1)
    abundance = abundance.loc[row_sums > 0]
    row_sums = row_sums.loc[abundance.index]
    abundance = abundance.div(row_sums, axis=0)
    abundance = abundance.loc[:, abundance.sum(axis=0) > 0]
    return abundance


def centered_log_ratio(abundance, pseudocount):
    abundance = abundance + pseudocount
    log_abundance = np.log(abundance)
    return log_abundance.sub(log_abundance.mean(axis=1), axis=0)


def fit_spectral_slope(power, frequency, fmin, fmax):
    mask = (frequency >= fmin) & (frequency <= fmax) & np.isfinite(power) & (power > 0)
    x = np.log10(frequency[mask])
    y = np.log10(power[mask])
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual_sum = np.sum((y - fitted) ** 2)
    total_sum = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - residual_sum / total_sum if total_sum > 0 else np.nan
    return -float(slope), float(intercept), float(r_squared)


def compute_power_spectrum(values, window):
    n_taxa = len(values)
    transformed = np.fft.rfft(np.asarray(values, dtype=float) * window)
    return (np.abs(transformed) ** 2) / n_taxa


def compute_sample_spectra(clr_abundance, metadata, group_col, study_col, frequency, window, fmin, fmax):
    records = []
    power_values = []
    for sample_id in clr_abundance.index:
        power = compute_power_spectrum(clr_abundance.loc[sample_id].values, window)
        spectral_slope, intercept, r_squared = fit_spectral_slope(power, frequency, fmin, fmax)
        study_name = metadata.loc[sample_id, study_col] if study_col in metadata.columns else "NA"
        records.append({
            "sample": sample_id,
            "group": metadata.loc[sample_id, group_col],
            "study_name": study_name,
            "spectral_slope": spectral_slope,
            "intercept": intercept,
            "r_squared": r_squared,
        })
        power_values.append(power)
    spectral_table = pd.DataFrame(records).set_index("sample")
    power_table = pd.DataFrame(np.vstack(power_values), index=clr_abundance.index, columns=frequency)
    return spectral_table, power_table


def summarize_group_spectra(power_table, spectral_table, group_order, frequency, fmin, fmax):
    summaries = {}
    for group in group_order:
        sample_ids = spectral_table.index[spectral_table["group"] == group]
        if len(sample_ids) == 0:
            continue
        group_power = power_table.loc[sample_ids, frequency]
        median = group_power.median(axis=0)
        lower = group_power.quantile(0.25, axis=0)
        upper = group_power.quantile(0.75, axis=0)
        spectral_slope, intercept, _ = fit_spectral_slope(median.values, frequency, fmin, fmax)
        summaries[group] = {
            "median": median,
            "lower": lower,
            "upper": upper,
            "spectral_slope": spectral_slope,
            "intercept": intercept,
        }
    return summaries


def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    greater = np.sum(x[:, None] > y[None, :])
    smaller = np.sum(x[:, None] < y[None, :])
    return float((greater - smaller) / (len(x) * len(y)))


def compare_groups(spectral_table, healthy_label, disease_label):
    healthy = spectral_table.loc[spectral_table["group"] == healthy_label, "spectral_slope"].dropna().values
    disease = spectral_table.loc[spectral_table["group"] == disease_label, "spectral_slope"].dropna().values
    if len(healthy) == 0 or len(disease) == 0:
        raise ValueError("Both groups must contain valid spectral slope values")
    p_value = mannwhitneyu(healthy, disease, alternative="two-sided").pvalue
    effect_size = cliffs_delta(disease, healthy)
    return healthy, disease, float(p_value), effect_size


def compute_study_effects(spectral_table, healthy_label, disease_label):
    records = []
    for study_name in sorted(spectral_table["study_name"].dropna().astype(str).unique()):
        subset = spectral_table[spectral_table["study_name"].astype(str) == study_name]
        if subset["group"].nunique() < 2:
            continue
        healthy = subset.loc[subset["group"] == healthy_label, "spectral_slope"].dropna().values
        disease = subset.loc[subset["group"] == disease_label, "spectral_slope"].dropna().values
        if len(healthy) == 0 or len(disease) == 0:
            continue
        records.append({
            "study_name": study_name,
            "cliffs_delta": cliffs_delta(disease, healthy),
            "mean_difference": float(np.mean(disease) - np.mean(healthy)),
            "p_value": float(mannwhitneyu(healthy, disease, alternative="two-sided").pvalue),
            "n_disease": int(len(disease)),
            "n_healthy": int(len(healthy)),
        })
    if len(records) == 0:
        return pd.DataFrame(columns=["study_name", "cliffs_delta", "mean_difference", "p_value", "n_disease", "n_healthy"])
    return pd.DataFrame(records).sort_values("cliffs_delta").reset_index(drop=True)


def save_taxon_order(abundance, output_dir):
    taxon_order = pd.DataFrame({
        "taxon_order": np.arange(1, abundance.shape[1] + 1),
        "taxon_fullname": abundance.columns,
    })
    taxon_order.to_csv(output_dir / "taxa_phylogeny_order.csv", index=False)


def plot_results(spectral_table, group_summaries, study_effects, plot_frequency, healthy_label, disease_label, healthy_name, disease_name, p_value, effect_size, output_dir, max_points, random_seed):
    display_names = {healthy_label: healthy_name, disease_label: disease_name}
    palette = {healthy_name: "#4C78A8", disease_name: "#D04A36"}
    spectral_table = spectral_table.copy()
    spectral_table["display_group"] = spectral_table["group"].map(display_names)
    display_order = [healthy_name, disease_name]

    fig = plt.figure(figsize=(11.5, 3.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.95, 0.85], wspace=0.35)

    ax1 = fig.add_subplot(grid[0, 0])
    for group in [healthy_label, disease_label]:
        if group not in group_summaries:
            continue
        label = display_names[group]
        summary = group_summaries[group]
        ax1.fill_between(plot_frequency, summary["lower"].values, summary["upper"].values, color=palette[label], alpha=0.16, linewidth=0)
        ax1.plot(plot_frequency, summary["median"].values, color=palette[label], lw=2.2, label=f'{label}  β={summary["spectral_slope"]:.2f}')
        fitted = 10 ** (summary["intercept"] - summary["spectral_slope"] * np.log10(plot_frequency))
        ax1.plot(plot_frequency, fitted, color=palette[label], lw=1.2, ls="--", alpha=0.9)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Phylogenetic frequency")
    ax1.set_ylabel("Power spectral density")
    ax1.legend(frameon=False, loc="upper right", fontsize=9, handlelength=2.6)
    ax1.set_title("Group-level phylogenetic spectra", pad=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(grid[0, 1])
    sns.violinplot(data=spectral_table, x="display_group", y="spectral_slope", order=display_order, palette=palette, inner=None, cut=0, linewidth=0, ax=ax2)
    sns.boxplot(data=spectral_table, x="display_group", y="spectral_slope", order=display_order, width=0.26, showcaps=False, boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1}, whiskerprops={"linewidth": 1}, medianprops={"color": "black", "linewidth": 1.2}, showfliers=False, ax=ax2)
    point_table = spectral_table.sample(n=min(len(spectral_table), max_points), random_state=random_seed)
    sns.stripplot(data=point_table, x="display_group", y="spectral_slope", order=display_order, palette=palette, alpha=0.45, size=2.2, jitter=0.22, ax=ax2)
    ax2.set_xlabel("")
    ax2.set_ylabel("Spectral slope (β)")
    ax2.set_title("Sample-level slope distribution", pad=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ymin, ymax = ax2.get_ylim()
    ax2.text(0.5, ymax - 0.04 * (ymax - ymin), f"Mann–Whitney P={p_value:.2e}\nCliff's δ={effect_size:.2f}", ha="center", va="top", fontsize=9)

    ax3 = fig.add_subplot(grid[0, 2])
    if len(study_effects) > 0:
        y = np.arange(len(study_effects))
        colors = np.where(study_effects["cliffs_delta"].values >= 0, palette[disease_name], palette[healthy_name])
        ax3.hlines(y, 0, study_effects["cliffs_delta"].values, color=colors, lw=2)
        ax3.scatter(study_effects["cliffs_delta"].values, y, s=40, c=colors, zorder=3)
        ax3.axvline(0, color="black", lw=0.8, ls="--")
        ax3.set_yticks(y)
        ax3.set_yticklabels(study_effects["study_name"].tolist())
        ax3.set_xlabel("Study-wise Cliff's δ")
        ax3.set_ylabel("Study")
        ax3.set_title("Cross-cohort consistency", pad=8)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
    else:
        ax3.axis("off")

    for axis, label in zip([ax1, ax2, ax3], ["a", "b", "c"]):
        axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontsize=13, fontweight="bold", va="bottom")

    fig.savefig(output_dir / "ibd_spectral_slope.png", bbox_inches="tight")
    fig.savefig(output_dir / "ibd_spectral_slope.pdf", bbox_inches="tight")
    plt.close(fig)


def save_group_spectra(group_summaries, plot_frequency, output_dir):
    records = []
    for group, summary in group_summaries.items():
        records.append(pd.DataFrame({
            "group": group,
            "frequency": plot_frequency,
            "median_power": summary["median"].values,
            "q1_power": summary["lower"].values,
            "q3_power": summary["upper"].values,
            "group_spectral_slope": summary["spectral_slope"],
            "group_intercept": summary["intercept"],
        }))
    pd.concat(records, axis=0, ignore_index=True).to_csv(output_dir / "group_spectra_summary.csv", index=False)


def save_parameters(args, abundance, fmin, p_value, effect_size, output_dir):
    parameters = pd.DataFrame({
        "parameter": [
            "sample_col",
            "group_col",
            "study_col",
            "pseudocount",
            "fmin",
            "fmax",
            "n_samples_after_filter",
            "n_taxa_after_filter",
            "min_samples_per_group",
            "taxonomy_level",
            "mannwhitney_p",
            "overall_cliffs_delta",
        ],
        "value": [
            args.sample_col,
            args.group_col,
            args.study_col,
            args.pseudocount,
            fmin,
            args.fmax,
            abundance.shape[0],
            abundance.shape[1],
            args.min_samples_per_group,
            "genus",
            p_value,
            effect_size,
        ],
    })
    parameters.to_csv(output_dir / "analysis_parameters.csv", index=False)


def main():
    args = parse_args()
    configure_plotting()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    abundance = load_abundance_table(args.abundance, args.sample_col)
    metadata = load_metadata(args.metadata, args.sample_col, args.group_col, args.healthy_label, args.disease_label)
    phylogeny_order = load_phylogeny_order(args.phylogeny)

    abundance = align_taxa_to_phylogeny(abundance, phylogeny_order)
    abundance, metadata = align_samples(abundance, metadata, args.group_col, args.min_samples_per_group)
    abundance = relative_abundance(abundance)
    metadata = metadata.loc[abundance.index]
    abundance = align_taxa_to_phylogeny(abundance, phylogeny_order)
    save_taxon_order(abundance, output_dir)

    n_taxa = abundance.shape[1]
    if n_taxa < 4:
        raise ValueError(f"Too few taxa after filtering: {n_taxa}")

    clr_abundance = centered_log_ratio(abundance, args.pseudocount)
    frequency = np.fft.rfftfreq(n_taxa, d=1.0)
    fmin = 2.0 / n_taxa
    plot_mask = (frequency >= fmin) & (frequency <= args.fmax)
    plot_frequency = frequency[plot_mask]
    if len(plot_frequency) < 3:
        raise ValueError("Too few frequency bins in the selected fitting range")

    window = np.hanning(n_taxa)
    spectral_table, power_table = compute_sample_spectra(clr_abundance, metadata, args.group_col, args.study_col, frequency, window, fmin, args.fmax)
    healthy_values, disease_values, p_value, effect_size = compare_groups(spectral_table, args.healthy_label, args.disease_label)
    study_effects = compute_study_effects(spectral_table, args.healthy_label, args.disease_label)
    group_summaries = summarize_group_spectra(power_table, spectral_table, [args.healthy_label, args.disease_label], plot_frequency, fmin, args.fmax)

    plot_results(spectral_table, group_summaries, study_effects, plot_frequency, args.healthy_label, args.disease_label, args.healthy_name, args.disease_name, p_value, effect_size, output_dir, args.max_points, args.random_seed)
    spectral_table.to_csv(output_dir / "spectral_slope_per_sample.csv")
    study_effects.to_csv(output_dir / "study_level_effects.csv", index=False)
    save_group_spectra(group_summaries, plot_frequency, output_dir)
    save_parameters(args, abundance, fmin, p_value, effect_size, output_dir)

    print(f"Healthy median spectral slope: {np.median(healthy_values):.6g}")
    print(f"Disease median spectral slope: {np.median(disease_values):.6g}")
    print(f"Mann-Whitney P value: {p_value:.6g}")
    print(f"Cliff's delta: {effect_size:.6g}")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
