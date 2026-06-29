from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, silhouette_samples

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Apply Fourier-domain batch harmonization to multi-cohort CRC microbiome profiles."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/crc_7_cohorts"))
    parser.add_argument("--taxonomy", type=Path, default=Path("data/crc_7_cohorts/classification.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/05_harmonization/crc_fourier_harmonization"))
    parser.add_argument("--n-cohorts", type=int, default=7)
    parser.add_argument("--pseudocount", type=float, default=1e-9)
    parser.add_argument("--cutoff-frequency", type=float, default=0.05)
    parser.add_argument("--correction-strength", type=float, default=1.0)
    parser.add_argument("--per-condition", action="store_true", default=True)
    parser.add_argument("--no-per-condition", action="store_false", dest="per_condition")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.labelcolor": "#333333",
            "text.color": "#333333",
        }
    )


def build_taxonomy_table(path):
    taxonomy = pd.read_csv(path, sep="\t")
    required = ["kindom", "phylum", "class", "order", "family", "genus"]
    missing = [column for column in required if column not in taxonomy.columns]
    if missing:
        raise ValueError(f"Missing taxonomy columns: {missing}")

    taxonomy["phylogeny"] = (
        taxonomy["kindom"].astype(str)
        + ";"
        + taxonomy["phylum"].astype(str)
        + ";"
        + taxonomy["class"].astype(str)
        + ";"
        + taxonomy["order"].astype(str)
        + ";"
        + taxonomy["family"].astype(str)
        + ";"
        + taxonomy["genus"].astype(str)
    )
    taxonomy["phylogeny"] = taxonomy["phylogeny"].str.replace("d__", "k__", regex=False)
    taxonomy = taxonomy.drop_duplicates(subset=["genus"]).set_index("genus")
    return taxonomy


def read_cohort_table(path):
    table = pd.read_csv(path, sep="\t", index_col=0).T
    table.index = table.index.astype(str)
    return table.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def load_crc_cohorts(data_dir, taxonomy_path, n_cohorts):
    taxonomy = build_taxonomy_table(taxonomy_path)

    abundance_tables = []
    metadata_records = []

    for cohort in range(1, n_cohorts + 1):
        disease_path = data_dir / f"d_crc{cohort}.txt"
        healthy_path = data_dir / f"h_crc{cohort}.txt"

        if not disease_path.exists():
            raise FileNotFoundError(f"Missing disease abundance table: {disease_path}")
        if not healthy_path.exists():
            raise FileNotFoundError(f"Missing healthy abundance table: {healthy_path}")

        disease = read_cohort_table(disease_path)
        healthy = read_cohort_table(healthy_path)

        abundance_tables.append(disease)
        abundance_tables.append(healthy)

        metadata_records.extend(
            [{"sample_id": sample, "batch": cohort, "disease": 1} for sample in disease.index]
        )
        metadata_records.extend(
            [{"sample_id": sample, "batch": cohort, "disease": 0} for sample in healthy.index]
        )

    abundance = pd.concat(abundance_tables, axis=0).fillna(0.0)
    metadata = pd.DataFrame(metadata_records).set_index("sample_id")
    abundance = abundance.loc[metadata.index]

    overlapping_taxa = abundance.columns.intersection(taxonomy.index)
    if len(overlapping_taxa) == 0:
        raise ValueError("No overlapping genera were found between CRC abundance tables and taxonomy.")

    abundance = abundance.loc[:, overlapping_taxa]
    abundance.columns = taxonomy.loc[overlapping_taxa, "phylogeny"]
    abundance = abundance.T.groupby(level=0).sum().T

    ordered_columns = sorted(abundance.columns.tolist())
    abundance = abundance.loc[:, ordered_columns]
    abundance = abundance.loc[:, abundance.sum(axis=0) > 0]

    return abundance, metadata


def relative_abundance(abundance):
    abundance = abundance.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    row_sums = abundance.sum(axis=1)
    abundance = abundance.loc[row_sums > 0].div(row_sums[row_sums > 0], axis=0)
    return abundance


def centered_log_ratio(abundance, pseudocount):
    abundance = abundance + pseudocount
    log_abundance = np.log(abundance)
    return log_abundance.sub(log_abundance.mean(axis=1), axis=0)


def inverse_centered_log_ratio(clr_table):
    values = np.exp(clr_table)
    return values.div(values.sum(axis=1), axis=0)


def lowpass_signal(signal, cutoff_frequency):
    n = len(signal)
    frequency = np.fft.rfftfreq(n)
    coefficients = np.fft.rfft(signal)
    filtered = np.zeros_like(coefficients)
    filtered[frequency <= cutoff_frequency] = coefficients[frequency <= cutoff_frequency]
    return np.fft.irfft(filtered, n=n)


def fourier_batch_harmonization(
    clr_table,
    metadata,
    batch_column,
    condition_column,
    cutoff_frequency,
    per_condition,
    correction_strength,
):
    corrected = clr_table.copy()
    batches = metadata[batch_column].unique()
    conditions = metadata[condition_column].unique() if per_condition else [None]

    for condition in conditions:
        if condition is None:
            condition_mask = pd.Series(True, index=metadata.index)
        else:
            condition_mask = metadata[condition_column] == condition

        global_mean = corrected.loc[condition_mask].mean(axis=0).to_numpy()

        for batch in batches:
            batch_mask = (metadata[batch_column] == batch) & condition_mask

            if batch_mask.sum() == 0:
                continue

            batch_mean = corrected.loc[batch_mask].mean(axis=0).to_numpy()
            batch_difference = batch_mean - global_mean
            batch_trend = lowpass_signal(batch_difference, cutoff_frequency)
            corrected.loc[batch_mask] = corrected.loc[batch_mask].sub(correction_strength * batch_trend, axis=1)

    corrected = corrected.sub(corrected.mean(axis=1), axis=0)
    return corrected, inverse_centered_log_ratio(corrected)


def batch_r2_from_distance(distance_matrix, labels):
    ids = list(distance_matrix.ids)
    labels = labels.loc[ids].astype(str).to_numpy()
    distances = np.asarray(distance_matrix.data, dtype=float)
    squared_distances = distances ** 2
    n = squared_distances.shape[0]

    if n <= 1:
        return np.nan

    total_sum = squared_distances.sum() / (2.0 * n)
    within_sum = 0.0

    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        if len(indices) <= 1:
            continue
        within_sum += squared_distances[np.ix_(indices, indices)].sum() / (2.0 * len(indices))

    if total_sum <= 0:
        return np.nan

    return float(max(0.0, (total_sum - within_sum) / total_sum))


def batch_silhouette_values(distance_matrix, labels):
    ids = list(distance_matrix.ids)
    labels = labels.loc[ids].astype(str).to_numpy()
    distances = np.asarray(distance_matrix.data, dtype=float)

    if len(np.unique(labels)) < 2:
        return np.full(len(labels), np.nan)

    return silhouette_samples(distances, labels, metric="precomputed")


def leave_one_batch_auc(features, labels, batches, n_estimators, random_state):
    aucs = {}

    for batch in batches.unique():
        test_mask = batches == batch
        train_mask = ~test_mask

        y_test = labels.loc[test_mask]
        if y_test.nunique() < 2:
            continue

        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        classifier.fit(features.loc[train_mask], labels.loc[train_mask])
        probabilities = classifier.predict_proba(features.loc[test_mask])[:, 1]
        aucs[str(float(batch))] = roc_auc_score(y_test, probabilities)

    return pd.Series(aucs, name="auc")


def p_value_text(value):
    if pd.isna(value):
        return "n.s."
    if value < 1e-4:
        return "P < 1e-4"
    if value < 0.001:
        return "P < 0.001"
    if value < 0.01:
        return f"P = {value:.3f}"
    return f"P = {value:.2f}"


def paired_or_unpaired_p(before, after, paired=True):
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    before = before[np.isfinite(before)]
    after = after[np.isfinite(after)]

    if len(before) == 0 or len(after) == 0:
        return np.nan

    try:
        if paired and len(before) == len(after):
            if np.allclose(before, after):
                return 1.0
            return float(wilcoxon(before, after, zero_method="wilcox").pvalue)
        return float(mannwhitneyu(before, after, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def plot_before_after_box(axis, before, after, ylabel, title, lower_is_better=False, paired=True, show_points=False):
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    before = before[np.isfinite(before)]
    after = after[np.isfinite(after)]
    table = pd.DataFrame(
        {
            "group": ["Before"] * len(before) + ["After"] * len(after),
            "value": np.concatenate([before, after]),
        }
    )

    sns.boxplot(
        data=table,
        x="group",
        y="value",
        order=["Before", "After"],
        palette={"Before": "#B7B5A0", "After": "#44757A"},
        width=0.52,
        linewidth=0.9,
        showfliers=False,
        ax=axis,
    )

    if show_points:
        n_pairs = min(len(before), len(after))
        for index in range(n_pairs):
            axis.plot([0, 1], [before[index], after[index]], color="#777777", linewidth=0.6, alpha=0.42, zorder=1)
        sns.stripplot(data=table, x="group", y="value", order=["Before", "After"], color="#666666", size=3.2, jitter=0.07, ax=axis, zorder=2)

    p_value = paired_or_unpaired_p(before, after, paired=paired)
    direction = "↓" if lower_is_better else "↑"
    axis.set_title(f"{title} {direction}", fontsize=11, pad=8)
    axis.set_xlabel("")
    axis.set_ylabel(ylabel, fontsize=10)
    axis.tick_params(axis="both", labelsize=9)
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    y_min, y_max = axis.get_ylim()
    axis.text(0.5, y_max - 0.04 * (y_max - y_min), p_value_text(p_value), ha="center", va="top", fontsize=8.5)


def plot_pcoa_comparison(before_distance, after_distance, metadata, output_dir, dpi):
    disease_labels = {0: "Healthy", 1: "CRC"}
    batch_order = [str(float(value)) for value in sorted(metadata["batch"].unique())]
    batch_palette = dict(
        zip(
            batch_order,
            ["#B3AEBF", "#B59478", "#D44C3C", "#452A3D", "#826AA2", "#9CAD8B", "#44757A"],
        )
    )

    before = pcoa(before_distance)
    after = pcoa(after_distance)

    before_table = before.samples[["PC1", "PC2"]].join(metadata)
    after_table = after.samples[["PC1", "PC2"]].join(metadata)

    before_table["batch"] = before_table["batch"].astype(float).astype(str)
    after_table["batch"] = after_table["batch"].astype(float).astype(str)
    before_table["disease_label"] = before_table["disease"].map(disease_labels)
    after_table["disease_label"] = after_table["disease"].map(disease_labels)
    before_table["PC1_plot"] = -before_table["PC1"]
    before_table["PC2_plot"] = before_table["PC2"]
    after_table["PC1_plot"] = after_table["PC1"]
    after_table["PC2_plot"] = after_table["PC2"]

    before_r2 = batch_r2_from_distance(before_distance, metadata["batch"])
    after_r2 = batch_r2_from_distance(after_distance, metadata["batch"])

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)

    sns.scatterplot(
        data=before_table,
        x="PC1_plot",
        y="PC2_plot",
        hue="batch",
        hue_order=batch_order,
        palette=batch_palette,
        style="disease_label",
        style_order=["Healthy", "CRC"],
        markers={"Healthy": "o", "CRC": "X"},
        s=42,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.95,
        ax=axes[0],
    )
    axes[0].set_title("Before batch correction", fontsize=12)
    axes[0].set_xlabel(f"PCoA1 ({before.proportion_explained.iloc[0] * 100:.1f}%)")
    axes[0].set_ylabel(f"PCoA2 ({before.proportion_explained.iloc[1] * 100:.1f}%)")
    axes[0].text(
        0.98,
        0.98,
        f"Batch R² = {before_r2:.3f}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "white", "edgecolor": "#B7B5A0", "linewidth": 0.8, "alpha": 0.94},
    )

    sns.scatterplot(
        data=after_table,
        x="PC1_plot",
        y="PC2_plot",
        hue="batch",
        hue_order=batch_order,
        palette=batch_palette,
        style="disease_label",
        style_order=["Healthy", "CRC"],
        markers={"Healthy": "o", "CRC": "X"},
        s=42,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.95,
        ax=axes[1],
    )
    axes[1].set_title("After Fourier batch correction", fontsize=12)
    axes[1].set_xlabel(f"PCoA1 ({after.proportion_explained.iloc[0] * 100:.1f}%)")
    axes[1].set_ylabel(f"PCoA2 ({after.proportion_explained.iloc[1] * 100:.1f}%)")
    axes[1].text(
        0.98,
        0.98,
        f"Batch R² = {after_r2:.3f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "white", "edgecolor": "#B7B5A0", "linewidth": 0.8, "alpha": 0.94},
    )

    for axis in axes:
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles, labels = axes[1].get_legend_handles_labels()

    for axis in axes:
        if axis.legend_ is not None:
            axis.legend_.remove()

    figure.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title=None)
    figure.savefig(output_dir / "crc_pcoa_harmonization.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / "crc_pcoa_harmonization.pdf", bbox_inches="tight")
    plt.close(figure)

    return before_r2, after_r2


def evaluate_harmonization(relative_before, clr_before, relative_after, clr_after, metadata, args):
    before_distance = beta_diversity("braycurtis", relative_before.to_numpy(), ids=relative_before.index)
    after_distance = beta_diversity("braycurtis", relative_after.to_numpy(), ids=relative_after.index)

    before_batch_r2 = batch_r2_from_distance(before_distance, metadata["batch"])
    after_batch_r2 = batch_r2_from_distance(after_distance, metadata["batch"])
    before_silhouette = batch_silhouette_values(before_distance, metadata["batch"])
    after_silhouette = batch_silhouette_values(after_distance, metadata["batch"])
    before_auc = leave_one_batch_auc(clr_before, metadata["disease"], metadata["batch"], args.n_estimators, args.random_state)
    after_auc = leave_one_batch_auc(clr_after, metadata["disease"], metadata["batch"], args.n_estimators, args.random_state)

    return {
        "before_distance": before_distance,
        "after_distance": after_distance,
        "before_batch_r2": before_batch_r2,
        "after_batch_r2": after_batch_r2,
        "before_silhouette": before_silhouette,
        "after_silhouette": after_silhouette,
        "before_auc": before_auc,
        "after_auc": after_auc,
    }


def plot_cutoff_sensitivity(clr_table, metadata, args):
    cutoffs = np.linspace(0, 0.1, 11)
    records = []

    for cutoff in cutoffs:
        corrected_clr, _ = fourier_batch_harmonization(
            clr_table,
            metadata,
            batch_column="batch",
            condition_column="disease",
            cutoff_frequency=cutoff,
            per_condition=args.per_condition,
            correction_strength=args.correction_strength,
        )
        aucs = leave_one_batch_auc(corrected_clr, metadata["disease"], metadata["batch"], args.n_estimators, args.random_state)
        for batch, value in aucs.items():
            records.append({"cutoff_frequency": cutoff, "batch": batch, "auc": value})

    table = pd.DataFrame(records)

    figure, axis = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    palette = ["#452A3D", "#44757A", "#9CAD8B", "#B7B5A0", "#A8BCCC", "#EEDFB7", "#E5855D", "#DD6C4C", "#D44C3C", "#B66065", "#EBCEC0"]
    sns.boxplot(
        data=table,
        x="cutoff_frequency",
        y="auc",
        palette=palette,
        width=0.55,
        linewidth=1.0,
        fliersize=2.5,
        ax=axis,
    )
    axis.set_xticklabels([f"{value:.2f}" for value in cutoffs])
    axis.set_xlabel("Cutoff frequency")
    axis.set_ylabel("Leave-one-batch-out disease AUC")
    axis.set_title("Cross-cohort AUC across Fourier cutoff frequencies", fontsize=12)
    axis.set_ylim(0.25, 1.02)
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.savefig(args.output_dir / "crc_cutoff_sensitivity_auc.png", dpi=args.dpi, bbox_inches="tight")
    figure.savefig(args.output_dir / "crc_cutoff_sensitivity_auc.pdf", bbox_inches="tight")
    plt.close(figure)

    table.to_csv(args.output_dir / "cutoff_sensitivity_auc.csv", index=False)


def write_outputs(relative_before, clr_before, relative_after, clr_after, metadata, evaluation, args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "metric": ["batch_r2", "median_batch_silhouette", "median_lobo_auc"],
            "before": [
                evaluation["before_batch_r2"],
                np.nanmedian(evaluation["before_silhouette"]),
                evaluation["before_auc"].median(),
            ],
            "after": [
                evaluation["after_batch_r2"],
                np.nanmedian(evaluation["after_silhouette"]),
                evaluation["after_auc"].median(),
            ],
        }
    ).to_csv(args.output_dir / "harmonization_summary.csv", index=False)

    metadata.to_csv(args.output_dir / "crc_metadata.csv")
    relative_after.to_csv(args.output_dir / "corrected_relative_abundance.csv")
    clr_after.to_csv(args.output_dir / "corrected_clr_abundance.csv")

    pd.DataFrame({"batch": metadata["batch"].astype(str).to_numpy(), "silhouette_before": evaluation["before_silhouette"], "silhouette_after": evaluation["after_silhouette"]}).to_csv(args.output_dir / "batch_silhouette_values.csv", index=False)
    pd.DataFrame({"batch": evaluation["before_auc"].index, "auc_before": evaluation["before_auc"].values, "auc_after": evaluation["after_auc"].reindex(evaluation["before_auc"].index).values}).to_csv(args.output_dir / "lobo_auc_values.csv", index=False)

    parameter_table = pd.DataFrame(
        {
            "parameter": [
                "n_cohorts",
                "pseudocount",
                "cutoff_frequency",
                "correction_strength",
                "per_condition",
                "n_estimators",
                "random_state",
            ],
            "value": [
                args.n_cohorts,
                args.pseudocount,
                args.cutoff_frequency,
                args.correction_strength,
                args.per_condition,
                args.n_estimators,
                args.random_state,
            ],
        }
    )
    parameter_table.to_csv(args.output_dir / "analysis_parameters.csv", index=False)


def main():
    args = parse_arguments()
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    abundance, metadata = load_crc_cohorts(args.data_dir, args.taxonomy, args.n_cohorts)
    relative_before = relative_abundance(abundance)
    metadata = metadata.loc[relative_before.index]
    clr_before = centered_log_ratio(relative_before, args.pseudocount)

    corrected_clr, corrected_relative = fourier_batch_harmonization(
        clr_before,
        metadata,
        batch_column="batch",
        condition_column="disease",
        cutoff_frequency=args.cutoff_frequency,
        per_condition=args.per_condition,
        correction_strength=args.correction_strength,
    )

    evaluation = evaluate_harmonization(
        relative_before,
        clr_before,
        corrected_relative,
        corrected_clr,
        metadata,
        args,
    )

    plot_pcoa_comparison(evaluation["before_distance"], evaluation["after_distance"], metadata, args.output_dir, args.dpi)

    figure, axis = plt.subplots(figsize=(4.2, 4.5), constrained_layout=True)
    plot_before_after_box(
        axis,
        evaluation["before_silhouette"],
        evaluation["after_silhouette"],
        ylabel="Batch silhouette",
        title="Residual batch structure",
        lower_is_better=True,
        paired=True,
        show_points=False,
    )
    figure.savefig(args.output_dir / "crc_batch_silhouette_comparison.png", dpi=args.dpi, bbox_inches="tight")
    figure.savefig(args.output_dir / "crc_batch_silhouette_comparison.pdf", bbox_inches="tight")
    plt.close(figure)

    common_batches = evaluation["before_auc"].index.intersection(evaluation["after_auc"].index)
    figure, axis = plt.subplots(figsize=(4.2, 4.5), constrained_layout=True)
    plot_before_after_box(
        axis,
        evaluation["before_auc"].loc[common_batches].to_numpy(),
        evaluation["after_auc"].loc[common_batches].to_numpy(),
        ylabel="Leave-one-batch-out disease AUC",
        title="Cross-cohort CRC prediction",
        lower_is_better=False,
        paired=True,
        show_points=True,
    )
    axis.set_ylim(0.0, 1.02)
    figure.savefig(args.output_dir / "crc_lobo_auc_comparison.png", dpi=args.dpi, bbox_inches="tight")
    figure.savefig(args.output_dir / "crc_lobo_auc_comparison.pdf", bbox_inches="tight")
    plt.close(figure)

    plot_cutoff_sensitivity(clr_before, metadata, args)
    write_outputs(relative_before, clr_before, corrected_relative, corrected_clr, metadata, evaluation, args)

    print(f"Samples retained: {relative_before.shape[0]}")
    print(f"Taxa retained: {relative_before.shape[1]}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
