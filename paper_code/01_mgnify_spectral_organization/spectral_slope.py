from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def repository_root():
    return Path(__file__).resolve().parents[2]


def parse_arguments():
    root = repository_root()
    parser = argparse.ArgumentParser(description="Calculate phylogenetic spectral slopes for MGnify biome profiles.")
    parser.add_argument("--abundance", type=Path, default=root / "data" / "mgnify" / "abu.h5")
    parser.add_argument("--metadata", type=Path, default=root / "data" / "mgnify" / "metadata.csv")
    parser.add_argument("--phylogeny", type=Path, default=root / "data" / "phylogeny.csv")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "01_mgnify_spectral_organization" / "spectral_slope")
    parser.add_argument("--h5-key", default="genus")
    parser.add_argument("--sample-column", default="SampleID")
    parser.add_argument("--environment-column", default="Env")
    parser.add_argument("--biome-column", default="level_3")
    parser.add_argument("--min-samples-per-biome", type=int, default=100)
    parser.add_argument("--top-n-biomes", type=int, default=8)
    parser.add_argument("--pseudocount", type=float, default=1e-9)
    parser.add_argument("--max-frequency", type=float, default=0.20)
    parser.add_argument("--genus-level-index", type=int, default=5)
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
    clr_abundance = np.log(abundance).sub(np.log(abundance).mean(axis=1), axis=0)
    return clr_abundance, metadata


def fit_spectral_slope(power, frequencies, minimum_frequency, maximum_frequency):
    mask = (frequencies >= minimum_frequency) & (frequencies <= maximum_frequency) & np.isfinite(power) & (power > 0)
    x_values = np.log10(frequencies[mask])
    y_values = np.log10(power[mask])

    if len(x_values) < 3:
        return np.nan, np.nan, np.nan

    slope, intercept = np.polyfit(x_values, y_values, 1)
    fitted = slope * x_values + intercept
    residual_sum = np.sum((y_values - fitted) ** 2)
    total_sum = np.sum((y_values - y_values.mean()) ** 2)
    r_squared = 1 - residual_sum / total_sum if total_sum > 0 else np.nan
    beta = -slope
    return float(beta), float(intercept), float(r_squared)


def calculate_sample_spectra(clr_abundance, metadata, biome_column, maximum_frequency):
    sample_records = []
    spectra_records = []
    n_taxa = clr_abundance.shape[1]
    window = np.hanning(n_taxa).astype(float)
    frequencies = np.fft.rfftfreq(n_taxa, d=1.0)
    minimum_frequency = 2.0 / n_taxa
    frequency_mask = (frequencies >= minimum_frequency) & (frequencies <= maximum_frequency)

    for sample_id in clr_abundance.index:
        signal = clr_abundance.loc[sample_id].to_numpy(dtype=float) * window
        power = (np.abs(np.fft.rfft(signal)) ** 2) / n_taxa
        beta, intercept, r_squared = fit_spectral_slope(power, frequencies, minimum_frequency, maximum_frequency)
        biome = metadata.loc[sample_id, biome_column]

        sample_records.append({
            "sample": sample_id,
            "biome": biome,
            "beta": beta,
            "intercept": intercept,
            "r_squared": r_squared,
            "n_taxa": int(n_taxa),
            "minimum_frequency": float(minimum_frequency),
            "maximum_frequency": float(maximum_frequency),
        })

        spectra_records.append(pd.DataFrame({
            "sample": sample_id,
            "biome": biome,
            "frequency": frequencies[frequency_mask],
            "power": power[frequency_mask],
        }))

    return pd.DataFrame(sample_records), pd.concat(spectra_records, axis=0, ignore_index=True)


def summarize_spectra(sample_spectra, spectral_slopes, top_n_biomes, maximum_frequency):
    biome_order = spectral_slopes["biome"].value_counts().index[:top_n_biomes].tolist()
    spectral_slopes = spectral_slopes[spectral_slopes["biome"].isin(biome_order)].copy()
    sample_spectra = sample_spectra[sample_spectra["biome"].isin(biome_order)].copy()

    spectral_slopes["biome"] = pd.Categorical(spectral_slopes["biome"], categories=biome_order, ordered=True)
    sample_spectra["biome"] = pd.Categorical(sample_spectra["biome"], categories=biome_order, ordered=True)

    spectra_summary = (
        sample_spectra
        .groupby(["biome", "frequency"], observed=False)["power"]
        .agg(
            median="median",
            q1=lambda values: np.quantile(values, 0.25),
            q3=lambda values: np.quantile(values, 0.75),
            mean="mean",
            standard_deviation="std",
            count="count",
        )
        .reset_index()
    )

    n_taxa = int(spectral_slopes["n_taxa"].iloc[0])
    minimum_frequency = 2.0 / n_taxa
    biome_records = []
    for biome in biome_order:
        subset = spectra_summary[spectra_summary["biome"] == biome].sort_values("frequency")
        beta, intercept, r_squared = fit_spectral_slope(
            subset["median"].to_numpy(dtype=float),
            subset["frequency"].to_numpy(dtype=float),
            minimum_frequency,
            maximum_frequency,
        )
        biome_records.append({
            "biome": biome,
            "group_beta": beta,
            "group_intercept": intercept,
            "group_r_squared": r_squared,
        })

    biome_slope_summary = pd.DataFrame(biome_records)
    panel_spectra = spectra_summary.merge(biome_slope_summary, on="biome", how="left")
    panel_spectra = panel_spectra.sort_values(["biome", "frequency"]).reset_index(drop=True)

    panel_sample_slopes = spectral_slopes.sort_values(["biome", "beta"]).reset_index(drop=True)

    panel_biome_ranking = (
        spectral_slopes.groupby("biome", observed=False)[["beta", "intercept", "r_squared"]]
        .agg(["median", "mean", "std", "count"])
    )
    panel_biome_ranking.columns = ["_".join(column).strip("_") for column in panel_biome_ranking.columns]
    panel_biome_ranking = panel_biome_ranking.reset_index()
    panel_biome_ranking = panel_biome_ranking.sort_values("beta_median", ascending=False).reset_index(drop=True)
    panel_biome_ranking["rank_by_beta_median"] = np.arange(1, len(panel_biome_ranking) + 1)

    return panel_spectra, panel_sample_slopes, panel_biome_ranking, sample_spectra, spectral_slopes


def save_outputs(args, panel_spectra, panel_sample_slopes, panel_biome_ranking, sample_spectra, spectral_slopes, clr_abundance):
    args.output.mkdir(parents=True, exist_ok=True)
    taxa_order = pd.DataFrame({
        "taxon_order": np.arange(1, len(clr_abundance.columns) + 1),
        "taxon_fullname": clr_abundance.columns,
    })
    taxa_order.to_csv(args.output / "taxa_phylogenetic_order.csv", index=False)
    panel_spectra.to_csv(args.output / "panel_spectra.csv", index=False)
    panel_sample_slopes.to_csv(args.output / "panel_sample_spectral_slopes.csv", index=False)
    panel_biome_ranking.to_csv(args.output / "panel_biome_spectral_slope_ranking.csv", index=False)
    sample_spectra.to_csv(args.output / "sample_spectra_long.csv", index=False)
    spectral_slopes.to_csv(args.output / "spectral_slopes_per_sample.csv", index=False)

    parameters = pd.DataFrame({
        "parameter": [
            "min_samples_per_biome",
            "top_n_biomes",
            "biome_column",
            "pseudocount",
            "minimum_frequency",
            "maximum_frequency",
            "n_taxa_after_filtering",
        ],
        "value": [
            args.min_samples_per_biome,
            args.top_n_biomes,
            args.biome_column,
            args.pseudocount,
            2.0 / clr_abundance.shape[1],
            args.max_frequency,
            clr_abundance.shape[1],
        ],
    })
    parameters.to_csv(args.output / "analysis_parameters.csv", index=False)


def main():
    args = parse_arguments()
    clr_abundance, metadata = load_and_prepare_data(args)
    spectral_slopes, sample_spectra = calculate_sample_spectra(clr_abundance, metadata, args.biome_column, args.max_frequency)
    outputs = summarize_spectra(sample_spectra, spectral_slopes, args.top_n_biomes, args.max_frequency)
    save_outputs(args, *outputs, clr_abundance)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
