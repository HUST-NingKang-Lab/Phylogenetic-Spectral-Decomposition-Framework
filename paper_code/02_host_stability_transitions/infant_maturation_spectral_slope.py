from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute phylogenetic spectral slopes for infant gut maturation analysis."
    )
    parser.add_argument("--abundance", type=Path, default=Path("data/infant/abundance.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/infant/metadata.csv"))
    parser.add_argument("--phylogeny", type=Path, default=Path("data/phylogeny.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/02_host_stability_transitions/infant_maturation_spectral_slope"),
    )
    parser.add_argument("--sample-col", default="sample_id")
    parser.add_argument("--group-col", default="group")
    parser.add_argument("--group-order", nargs="+", default=["immature", "mature"])
    parser.add_argument("--pseudocount", type=float, default=1e-9)
    parser.add_argument("--fmax", type=float, default=0.20)
    return parser.parse_args()


def normalize_taxon_name(value):
    value = str(value).strip().replace("|", ";")
    parts = [part.strip() for part in value.split(";") if part.strip()]
    normalized = []
    for part in parts:
        if part.startswith("sk__"):
            part = "k__" + part[4:]
        normalized.append(part)
    return ";".join(normalized)


def collapse_to_genus(value):
    value = normalize_taxon_name(value)
    parts = [part for part in value.split(";") if part]
    retained = []
    for part in parts:
        if part.startswith(("s__", "t__")):
            break
        retained.append(part)
        if part.startswith("g__"):
            break
    return ";".join(retained)


def load_phylogeny_order(path):
    phylogeny = pd.read_csv(path, low_memory=False)
    taxon_column = phylogeny.columns[0]
    order = (
        phylogeny[taxon_column]
        .astype(str)
        .map(collapse_to_genus)
        .dropna()
        .loc[lambda x: x != ""]
        .drop_duplicates()
        .tolist()
    )
    return order


def load_abundance_table(path, sample_column):
    abundance = pd.read_csv(path, low_memory=False)
    if sample_column not in abundance.columns:
        raise ValueError(f"Sample column '{sample_column}' was not found in the abundance table.")
    abundance[sample_column] = abundance[sample_column].astype(str)
    abundance = abundance.set_index(sample_column)
    abundance.columns = [collapse_to_genus(column) for column in abundance.columns]
    abundance = abundance.loc[:, [column != "" for column in abundance.columns]]
    abundance = abundance.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    abundance = abundance.T.groupby(level=0).sum().T
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


def align_data(abundance, metadata, phylogeny_order):
    ordered_columns = [taxon for taxon in phylogeny_order if taxon in abundance.columns]
    if len(ordered_columns) == 0:
        raise ValueError("No overlapping genus-level taxa were found between abundance and phylogeny tables.")

    abundance = abundance.loc[:, ordered_columns]
    abundance = abundance.loc[:, abundance.sum(axis=0) > 0]

    shared_samples = abundance.index.intersection(metadata.index)
    if len(shared_samples) == 0:
        raise ValueError("No overlapping samples were found between abundance and metadata tables.")

    abundance = abundance.loc[shared_samples].copy()
    metadata = metadata.loc[shared_samples].copy()

    row_sums = abundance.sum(axis=1)
    abundance = abundance.loc[row_sums > 0].div(row_sums[row_sums > 0], axis=0)
    metadata = metadata.loc[abundance.index].copy()
    abundance = abundance.loc[:, abundance.sum(axis=0) > 0]

    if abundance.shape[1] < 4:
        raise ValueError(f"Too few taxa remained after filtering: {abundance.shape[1]}.")

    return abundance, metadata


def centered_log_ratio(abundance, pseudocount):
    abundance = abundance + pseudocount
    log_abundance = np.log(abundance)
    return log_abundance.sub(log_abundance.mean(axis=1), axis=0)


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


def compute_sample_spectra(clr_abundance, metadata, group_column, fmax):
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
    records = []

    indexed_slopes = slope_table.set_index("sample_id")

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

        group_table = pd.DataFrame(
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
        records.append(group_table)

    if len(records) == 0:
        return pd.DataFrame()

    return pd.concat(records, axis=0, ignore_index=True)


def write_outputs(
    output_dir,
    abundance,
    slope_table,
    group_spectra,
    sample_column,
    group_column,
    pseudocount,
    fmin,
    fmax,
    phylogeny_order,
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

    if not group_spectra.empty:
        group_spectra.to_csv(output_dir / "group_spectra_summary.csv", index=False)

    parameter_table = pd.DataFrame(
        {
            "parameter": [
                "sample_column",
                "group_column",
                "pseudocount",
                "fmin",
                "fmax",
                "n_samples",
                "n_taxa",
                "n_phylogeny_taxa",
                "taxonomy_level",
            ],
            "value": [
                sample_column,
                group_column,
                pseudocount,
                fmin,
                fmax,
                abundance.shape[0],
                abundance.shape[1],
                len(phylogeny_order),
                "genus",
            ],
        }
    )
    parameter_table.to_csv(output_dir / "analysis_parameters.csv", index=False)


def main():
    args = parse_arguments()

    phylogeny_order = load_phylogeny_order(args.phylogeny)
    abundance = load_abundance_table(args.abundance, args.sample_col)
    metadata = load_metadata(args.metadata, args.sample_col, args.group_col, args.group_order)
    abundance, metadata = align_data(abundance, metadata, phylogeny_order)
    clr_abundance = centered_log_ratio(abundance, args.pseudocount)

    slope_table, power_table, frequency, frequency_mask, fmin = compute_sample_spectra(
        clr_abundance=clr_abundance,
        metadata=metadata,
        group_column=args.group_col,
        fmax=args.fmax,
    )

    group_spectra = summarize_group_spectra(
        power_table=power_table,
        slope_table=slope_table,
        frequency=frequency,
        frequency_mask=frequency_mask,
        group_order=args.group_order,
    )

    write_outputs(
        output_dir=args.output_dir,
        abundance=abundance,
        slope_table=slope_table,
        group_spectra=group_spectra,
        sample_column=args.sample_col,
        group_column=args.group_col,
        pseudocount=args.pseudocount,
        fmin=fmin,
        fmax=args.fmax,
        phylogeny_order=phylogeny_order,
    )

    print(f"Samples retained: {abundance.shape[0]}")
    print(f"Taxa retained: {abundance.shape[1]}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
