from pathlib import Path
import argparse
import gc
import numpy as np
import pandas as pd
from scipy.stats import kruskal


def repository_root():
    return Path(__file__).resolve().parents[2]


def parse_arguments():
    root = repository_root()
    parser = argparse.ArgumentParser(description="Test whether biome spectral separation depends on phylogenetic taxon order.")
    parser.add_argument("--abundance", type=Path, default=root / "data" / "mgnify" / "abu.h5")
    parser.add_argument("--metadata", type=Path, default=root / "data" / "mgnify" / "metadata.csv")
    parser.add_argument("--phylogeny", type=Path, default=root / "data" / "phylogeny.csv")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "01_mgnify_spectral_organization" / "random_order_permutation")
    parser.add_argument("--h5-key", default="genus")
    parser.add_argument("--sample-column", default="SampleID")
    parser.add_argument("--environment-column", default="Env")
    parser.add_argument("--biome-column", default="level_3")
    parser.add_argument("--min-samples-per-biome", type=int, default=100)
    parser.add_argument("--top-n-biomes", type=int, default=8)
    parser.add_argument("--pseudocount", type=float, default=1e-9)
    parser.add_argument("--max-frequency", type=float, default=0.20)
    parser.add_argument("--genus-level-index", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=202504)
    return parser.parse_args()


def normalize_taxon_name(value):
    taxon = str(value).strip()
    if taxon.startswith("sk__"):
        taxon = "k__" + taxon[4:]
    return taxon


def prepare_metadata(metadata, sample_column, environment_column):
    metadata = metadata.rename(columns={sample_column: "sample", environment_column: "biome"})
    metadata["sample"] = metadata["sample"].astype(str)
    metadata = metadata.set_index("sample")
    levels = metadata["biome"].astype(str).str.split(":", expand=True)
    for index in range(levels.shape[1]):
        metadata[f"level_{index + 1}"] = levels[index]
    return metadata


def build_phylogenetic_order(phylogeny, genus_level_index):
    taxonomy = phylogeny.iloc[:, 0].astype(str).str.split(";", expand=True)
    taxonomy.index = taxonomy[genus_level_index]
    taxonomy = taxonomy[~taxonomy.index.duplicated(keep="first")]
    full_names = taxonomy.loc[:, 0:genus_level_index].agg(";".join, axis=1)
    return full_names


def load_and_prepare_data(args):
    abundance = pd.read_hdf(args.abundance, args.h5_key)
    metadata = pd.read_csv(args.metadata)
    phylogeny = pd.read_csv(args.phylogeny)

    metadata = prepare_metadata(metadata, args.sample_column, args.environment_column)
    abundance.index = abundance.index.astype(str)
    abundance.columns = [normalize_taxon_name(column) for column in abundance.columns]

    phylogenetic_order = build_phylogenetic_order(phylogeny, args.genus_level_index)
    shared_taxa = abundance.columns.intersection(phylogenetic_order.index)
    abundance = abundance.loc[:, shared_taxa]
    abundance.columns = phylogenetic_order.loc[abundance.columns].values

    shared_samples = abundance.index.intersection(metadata.index)
    abundance = abundance.loc[shared_samples]
    metadata = metadata.loc[shared_samples]

    metadata = metadata[metadata[args.biome_column].notna()]
    abundance = abundance.loc[metadata.index]

    biome_counts = metadata[args.biome_column].value_counts()
    valid_biomes = biome_counts[biome_counts >= args.min_samples_per_biome].index.tolist()
    metadata = metadata[metadata[args.biome_column].isin(valid_biomes)]
    abundance = abundance.loc[metadata.index]

    ordered_columns = [taxon for taxon in phylogenetic_order.values if taxon in abundance.columns]
    abundance = abundance[ordered_columns]
    abundance = abundance.loc[:, abundance.sum(axis=0) > 0]

    abundance = abundance.div(abundance.sum(axis=1), axis=0)
    abundance = abundance.loc[abundance.sum(axis=1) > 0]
    metadata = metadata.loc[abundance.index]

    abundance = abundance + args.pseudocount
    clr_abundance = np.log(abundance).sub(np.log(abundance).mean(axis=1), axis=0).astype(np.float32)
    return clr_abundance, metadata


def fit_spectral_slope(power, frequencies, minimum_frequency, maximum_frequency):
    mask = (frequencies >= minimum_frequency) & (frequencies <= maximum_frequency) & np.isfinite(power) & (power > 0)
    x_values = np.log10(frequencies[mask])
    y_values = np.log10(power[mask])

    if len(x_values) < 3:
        return np.nan

    slope, _ = np.polyfit(x_values, y_values, 1)
    return float(-slope)


def calculate_beta_vector(clr_values, order_index, window, frequencies, minimum_frequency, maximum_frequency):
    betas = np.empty(clr_values.shape[0], dtype=np.float32)
    n_taxa = len(order_index)

    for row_index in range(clr_values.shape[0]):
        signal = clr_values[row_index, order_index] * window
        power = (np.abs(np.fft.rfft(signal)) ** 2) / n_taxa
        betas[row_index] = fit_spectral_slope(power, frequencies, minimum_frequency, maximum_frequency)

    return betas


def summarize_group_separation(beta_values, group_indices, biome_order):
    groups = []
    for biome in biome_order:
        values = beta_values[group_indices[biome]]
        values = values[np.isfinite(values)]
        if len(values) > 0:
            groups.append(values)

    kruskal_statistic, kruskal_p_value = kruskal(*groups)

    medians = []
    for biome in biome_order:
        values = beta_values[group_indices[biome]]
        values = values[np.isfinite(values)]
        medians.append(np.median(values) if len(values) > 0 else np.nan)
    medians = np.asarray(medians, dtype=float)

    pairwise_differences = []
    for first_index in range(len(medians)):
        for second_index in range(first_index + 1, len(medians)):
            if np.isfinite(medians[first_index]) and np.isfinite(medians[second_index]):
                pairwise_differences.append(abs(medians[first_index] - medians[second_index]))

    return {
        "kruskal_statistic": float(kruskal_statistic),
        "kruskal_p_value": float(kruskal_p_value),
        "median_range": float(np.nanmax(medians) - np.nanmin(medians)),
        "mean_absolute_pairwise_median_difference": float(np.mean(pairwise_differences)),
        "maximum_absolute_pairwise_median_difference": float(np.max(pairwise_differences)),
    }


def empirical_p_value(null_values, observed_value):
    null_values = np.asarray(null_values, dtype=float)
    return float((1.0 + np.sum(null_values >= observed_value)) / (len(null_values) + 1.0))


def create_summary(observed_statistics, null_distribution):
    records = []
    statistic_names = [
        "kruskal_statistic",
        "median_range",
        "mean_absolute_pairwise_median_difference",
        "maximum_absolute_pairwise_median_difference",
    ]

    for statistic in statistic_names:
        observed_value = observed_statistics[statistic]
        null_values = null_distribution[statistic].to_numpy(dtype=float)
        null_mean = float(np.nanmean(null_values))
        null_sd = float(np.nanstd(null_values, ddof=1))
        z_score = float((observed_value - null_mean) / null_sd) if null_sd > 0 else np.nan
        records.append({
            "statistic": statistic,
            "observed_value": observed_value,
            "null_mean": null_mean,
            "null_standard_deviation": null_sd,
            "z_score": z_score,
            "empirical_p_greater_equal": empirical_p_value(null_values, observed_value),
        })

    return pd.DataFrame(records)


def main():
    args = parse_arguments()
    args.output.mkdir(parents=True, exist_ok=True)

    clr_abundance, metadata = load_and_prepare_data(args)
    n_taxa = clr_abundance.shape[1]
    window = np.hanning(n_taxa).astype(np.float32)
    frequencies = np.fft.rfftfreq(n_taxa, d=1.0)
    minimum_frequency = 2.0 / n_taxa
    true_order = np.arange(n_taxa, dtype=int)

    all_sample_ids = clr_abundance.index.to_numpy()
    all_biomes = metadata.loc[all_sample_ids, args.biome_column].astype(str).to_numpy()
    all_values = clr_abundance.to_numpy(copy=True).astype(np.float32)

    observed_beta = calculate_beta_vector(all_values, true_order, window, frequencies, minimum_frequency, args.max_frequency)
    observed_samples = pd.DataFrame({
        "sample": all_sample_ids,
        "biome": all_biomes,
        "beta": observed_beta,
    })

    biome_order = observed_samples["biome"].value_counts().index[:args.top_n_biomes].tolist()
    observed_samples = observed_samples[observed_samples["biome"].isin(biome_order)].copy()
    selected_sample_ids = observed_samples["sample"].to_numpy()
    selected_biomes = observed_samples["biome"].to_numpy()
    selected_values = clr_abundance.loc[selected_sample_ids].to_numpy(copy=True).astype(np.float32)

    del clr_abundance, metadata, all_values
    gc.collect()

    group_indices = {biome: np.where(selected_biomes == biome)[0] for biome in biome_order}
    observed_statistics = summarize_group_separation(observed_samples["beta"].to_numpy(dtype=np.float32), group_indices, biome_order)

    observed_medians = (
        observed_samples.groupby("biome", observed=False)["beta"]
        .median()
        .reindex(biome_order)
        .reset_index()
        .rename(columns={"beta": "observed_beta_median"})
    )

    rng = np.random.default_rng(args.random_seed)
    null_records = []
    for permutation_index in range(1, args.n_permutations + 1):
        permutation_order = rng.permutation(n_taxa)
        permuted_beta = calculate_beta_vector(selected_values, permutation_order, window, frequencies, minimum_frequency, args.max_frequency)
        statistics = summarize_group_separation(permuted_beta, group_indices, biome_order)
        null_records.append({"permutation": permutation_index, **statistics})
        if permutation_index % 20 == 0 or permutation_index == args.n_permutations:
            print(f"Finished {permutation_index}/{args.n_permutations} permutations")

    null_distribution = pd.DataFrame(null_records)
    summary = create_summary(observed_statistics, null_distribution)

    observed_samples.to_csv(args.output / "observed_beta_per_sample.csv", index=False)
    observed_medians.to_csv(args.output / "observed_biome_beta_medians.csv", index=False)
    null_distribution.to_csv(args.output / "null_distribution_beta.csv", index=False)
    summary.to_csv(args.output / "null_summary_beta.csv", index=False)

    parameters = pd.DataFrame({
        "parameter": [
            "min_samples_per_biome",
            "top_n_biomes",
            "biome_column",
            "pseudocount",
            "minimum_frequency",
            "maximum_frequency",
            "n_taxa_after_filtering",
            "n_permutations",
            "random_seed",
        ],
        "value": [
            args.min_samples_per_biome,
            args.top_n_biomes,
            args.biome_column,
            args.pseudocount,
            minimum_frequency,
            args.max_frequency,
            n_taxa,
            args.n_permutations,
            args.random_seed,
        ],
    })
    parameters.to_csv(args.output / "permutation_parameters.csv", index=False)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
