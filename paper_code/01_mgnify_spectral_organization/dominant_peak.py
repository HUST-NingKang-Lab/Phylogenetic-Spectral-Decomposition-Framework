from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def repository_root():
    return Path(__file__).resolve().parents[2]


def parse_arguments():
    root = repository_root()
    parser = argparse.ArgumentParser(description="Identify dominant spectral peak frequencies in MGnify biome profiles.")
    parser.add_argument("--abundance", type=Path, default=root / "data" / "mgnify" / "abu.h5")
    parser.add_argument("--metadata", type=Path, default=root / "data" / "mgnify" / "metadata.csv")
    parser.add_argument("--phylogeny", type=Path, default=root / "data" / "phylogeny.csv")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "01_mgnify_spectral_organization" / "dominant_peak")
    parser.add_argument("--h5-key", default="genus")
    parser.add_argument("--sample-column", default="SampleID")
    parser.add_argument("--environment-column", default="Env")
    parser.add_argument("--biome-column", default="level_3")
    parser.add_argument("--min-samples-per-biome", type=int, default=100)
    parser.add_argument("--top-n-biomes", type=int, default=8)
    parser.add_argument("--pseudocount", type=float, default=1e-9)
    parser.add_argument("--max-frequency", type=float, default=0.20)
    parser.add_argument("--genus-level-index", type=int, default=5)
    parser.add_argument("--prominence-quantile", type=float, default=0.60)
    parser.add_argument("--prominence-scale", type=float, default=0.15)
    parser.add_argument("--minimum-distance-divisor", type=int, default=12)
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
    valid_biomes = biome_counts[biome_counts >= args.min_samples_per_biome].index[:args.top_n_biomes].tolist()
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


def identify_dominant_peak(power, frequencies, args):
    if len(power) >= 5:
        prominence = np.quantile(power, args.prominence_quantile) * args.prominence_scale
        peaks, properties = find_peaks(
            power,
            prominence=prominence,
            distance=max(1, len(power) // args.minimum_distance_divisor),
        )
    else:
        peaks = np.array([], dtype=int)
        properties = {"prominences": np.array([])}

    if len(peaks) == 0:
        peak_index = int(np.argmax(power))
        return peak_index, 0.0, 0

    best_peak = int(np.argmax(power[peaks]))
    peak_index = int(peaks[best_peak])
    prominence = float(properties["prominences"][best_peak]) if len(properties["prominences"]) > best_peak else 0.0
    return peak_index, prominence, 1


def calculate_dominant_peaks(clr_abundance, metadata, args):
    peak_records = []
    spectra_records = []
    n_taxa = clr_abundance.shape[1]
    window = np.hanning(n_taxa).astype(np.float32)
    frequencies = np.fft.rfftfreq(n_taxa, d=1.0)
    minimum_frequency = 2.0 / n_taxa
    frequency_mask = (frequencies >= minimum_frequency) & (frequencies <= args.max_frequency)
    selected_frequencies = frequencies[frequency_mask]

    for sample_id in clr_abundance.index:
        signal = clr_abundance.loc[sample_id].to_numpy(dtype=np.float32) * window
        power = (np.abs(np.fft.rfft(signal)) ** 2) / n_taxa
        selected_power = power[frequency_mask].astype(np.float32)
        biome = metadata.loc[sample_id, args.biome_column]
        peak_index, prominence, peak_found = identify_dominant_peak(selected_power, selected_frequencies, args)
        dominant_frequency = float(selected_frequencies[peak_index])
        dominant_power = float(selected_power[peak_index])

        spectra_records.append(pd.DataFrame({
            "sample": sample_id,
            "biome": biome,
            "frequency": selected_frequencies,
            "power": selected_power,
        }))

        peak_records.append({
            "sample": sample_id,
            "biome": biome,
            "dominant_peak_frequency": dominant_frequency,
            "dominant_peak_scale": float(1.0 / dominant_frequency),
            "dominant_peak_power": dominant_power,
            "dominant_peak_prominence": prominence,
            "peak_found_by_local_detection": int(peak_found),
            "n_taxa": int(n_taxa),
            "minimum_frequency": float(minimum_frequency),
            "maximum_frequency": float(args.max_frequency),
        })

    return pd.DataFrame(peak_records), pd.concat(spectra_records, axis=0, ignore_index=True)


def summarize_outputs(peak_table, spectra_table):
    biome_order = peak_table["biome"].value_counts().index.tolist()
    peak_table["biome"] = pd.Categorical(peak_table["biome"], categories=biome_order, ordered=True)
    spectra_table["biome"] = pd.Categorical(spectra_table["biome"], categories=biome_order, ordered=True)

    spectra_summary = (
        spectra_table
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

    peak_medians = (
        peak_table.groupby("biome", observed=False)["dominant_peak_frequency"]
        .median()
        .reset_index()
        .rename(columns={"dominant_peak_frequency": "dominant_peak_frequency_median"})
    )

    panel_spectra = spectra_summary.merge(peak_medians, on="biome", how="left")
    panel_spectra = panel_spectra.sort_values(["biome", "frequency"]).reset_index(drop=True)
    panel_sample_peaks = peak_table.sort_values(["biome", "dominant_peak_frequency"]).reset_index(drop=True)

    panel_biome_ranking = (
        peak_table.groupby("biome", observed=False)[
            ["dominant_peak_frequency", "dominant_peak_scale", "dominant_peak_power", "dominant_peak_prominence"]
        ]
        .agg(["median", "mean", "std", "count"])
    )
    panel_biome_ranking.columns = ["_".join(column).strip("_") for column in panel_biome_ranking.columns]
    panel_biome_ranking = panel_biome_ranking.reset_index()
    panel_biome_ranking = panel_biome_ranking.sort_values("dominant_peak_frequency_median", ascending=False).reset_index(drop=True)
    panel_biome_ranking["rank_by_peak_frequency_median"] = np.arange(1, len(panel_biome_ranking) + 1)

    return panel_spectra, panel_sample_peaks, panel_biome_ranking


def save_outputs(args, clr_abundance, peak_table, spectra_table, panel_spectra, panel_sample_peaks, panel_biome_ranking):
    args.output.mkdir(parents=True, exist_ok=True)
    taxa_order = pd.DataFrame({
        "taxon_order": np.arange(1, len(clr_abundance.columns) + 1),
        "taxon_fullname": clr_abundance.columns,
    })
    taxa_order.to_csv(args.output / "taxa_phylogenetic_order.csv", index=False)
    panel_spectra.to_csv(args.output / "panel_spectra.csv", index=False)
    panel_sample_peaks.to_csv(args.output / "panel_sample_dominant_peaks.csv", index=False)
    panel_biome_ranking.to_csv(args.output / "panel_biome_peak_ranking.csv", index=False)
    spectra_table.to_csv(args.output / "sample_spectra_long.csv", index=False)
    peak_table.to_csv(args.output / "dominant_peaks_per_sample.csv", index=False)

    parameters = pd.DataFrame({
        "parameter": [
            "min_samples_per_biome",
            "top_n_biomes",
            "biome_column",
            "pseudocount",
            "minimum_frequency",
            "maximum_frequency",
            "prominence_quantile",
            "prominence_scale",
            "minimum_distance_divisor",
            "n_taxa_after_filtering",
        ],
        "value": [
            args.min_samples_per_biome,
            args.top_n_biomes,
            args.biome_column,
            args.pseudocount,
            2.0 / clr_abundance.shape[1],
            args.max_frequency,
            args.prominence_quantile,
            args.prominence_scale,
            args.minimum_distance_divisor,
            clr_abundance.shape[1],
        ],
    })
    parameters.to_csv(args.output / "analysis_parameters.csv", index=False)


def main():
    args = parse_arguments()
    clr_abundance, metadata = load_and_prepare_data(args)
    peak_table, spectra_table = calculate_dominant_peaks(clr_abundance, metadata, args)
    panel_spectra, panel_sample_peaks, panel_biome_ranking = summarize_outputs(peak_table, spectra_table)
    save_outputs(args, clr_abundance, peak_table, spectra_table, panel_spectra, panel_sample_peaks, panel_biome_ranking)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
