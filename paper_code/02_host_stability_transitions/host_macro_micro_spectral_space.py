from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Construct a macro-micro spectral space for host-associated microbiome transitions."
    )
    parser.add_argument("--palleja-abundance", type=Path, default=Path("data/palleja/abundance_phylogeny_ordered.csv"))
    parser.add_argument("--palleja-metadata", type=Path, default=Path("data/palleja/metadata.csv"))
    parser.add_argument("--infant-abundance", type=Path, default=Path("data/infant/abundance.csv"))
    parser.add_argument("--infant-metadata", type=Path, default=Path("data/infant/metadata.csv"))
    parser.add_argument("--ibd-abundance", type=Path, default=Path("data/ibd/abundance.csv"))
    parser.add_argument("--ibd-metadata", type=Path, default=Path("data/ibd/metadata.csv"))
    parser.add_argument("--phylogeny", type=Path, default=Path("data/phylogeny.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/02_host_stability_transitions/host_macro_micro_spectral_space"),
    )
    parser.add_argument("--pseudocount", type=float, default=1e-9)
    parser.add_argument("--fmax", type=float, default=0.20)
    parser.add_argument("--low-fraction", type=float, default=0.25)
    parser.add_argument("--high-fraction", type=float, default=0.35)
    parser.add_argument("--max-points-per-group", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--contrast-stretch", action="store_true")
    return parser.parse_args()


def configure_matplotlib(dpi):
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.6,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": dpi,
            "figure.dpi": 160,
        }
    )


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
    return (
        phylogeny[taxon_column]
        .astype(str)
        .map(collapse_to_genus)
        .dropna()
        .loc[lambda x: x != ""]
        .drop_duplicates()
        .tolist()
    )


def robust_scale(values, lower_quantile=0.02, upper_quantile=0.98):
    values = pd.Series(values, dtype=float)
    lower = values.quantile(lower_quantile)
    upper = values.quantile(upper_quantile)

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = values.min()
        upper = values.max()

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return pd.Series(np.full(len(values), 0.5), index=values.index)

    return ((values - lower) / (upper - lower)).clip(0, 1)


def stretch_unit_interval(values, strength):
    values = pd.Series(values, dtype=float).clip(0, 1)
    transformed = 1.0 / (1.0 + np.exp(-strength * (values - 0.5)))
    lower = 1.0 / (1.0 + np.exp(-strength * (0.0 - 0.5)))
    upper = 1.0 / (1.0 + np.exp(-strength * (1.0 - 0.5)))
    return ((transformed - lower) / (upper - lower)).clip(0, 1)


def load_palleja_dataset(abundance_path, metadata_path):
    sample_column = "sample_id"
    group_column = "label"
    group_map = {
        "pre_stable": "Pre-stable",
        "perturbation_unstable": "Perturbation",
        "recovery_stable": "Recovery",
    }

    abundance = pd.read_csv(abundance_path, low_memory=False)
    metadata = pd.read_csv(metadata_path, low_memory=False)

    if sample_column not in abundance.columns:
        raise ValueError(f"Sample column '{sample_column}' was not found in Palleja abundance table.")
    if sample_column not in metadata.columns:
        raise ValueError(f"Sample column '{sample_column}' was not found in Palleja metadata table.")
    if group_column not in metadata.columns:
        raise ValueError(f"Group column '{group_column}' was not found in Palleja metadata table.")

    abundance[sample_column] = abundance[sample_column].astype(str)
    metadata[sample_column] = metadata[sample_column].astype(str)

    abundance = abundance.set_index(sample_column)
    metadata = metadata.set_index(sample_column)
    metadata = metadata[metadata[group_column].isin(group_map.keys())].copy()

    shared_samples = abundance.index.intersection(metadata.index)
    abundance = abundance.loc[shared_samples].copy()
    metadata = metadata.loc[shared_samples].copy()

    abundance = abundance.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    metadata["dataset"] = "Palleja"
    metadata["group"] = metadata[group_column].map(group_map)

    return abundance, metadata[["dataset", "group"]]


def load_infant_dataset(abundance_path, metadata_path, phylogeny_order):
    sample_column = "sample_id"
    group_column = "group"
    group_map = {"immature": "Immature", "mature": "Mature"}

    abundance = pd.read_csv(abundance_path, low_memory=False)
    metadata = pd.read_csv(metadata_path, low_memory=False)

    if sample_column not in abundance.columns:
        raise ValueError(f"Sample column '{sample_column}' was not found in infant abundance table.")
    if sample_column not in metadata.columns:
        raise ValueError(f"Sample column '{sample_column}' was not found in infant metadata table.")
    if group_column not in metadata.columns:
        raise ValueError(f"Group column '{group_column}' was not found in infant metadata table.")

    abundance[sample_column] = abundance[sample_column].astype(str)
    metadata[sample_column] = metadata[sample_column].astype(str)

    abundance = abundance.set_index(sample_column)
    metadata = metadata.set_index(sample_column)

    abundance.columns = [collapse_to_genus(column) for column in abundance.columns]
    abundance = abundance.loc[:, [column != "" for column in abundance.columns]]
    abundance = abundance.T.groupby(level=0).sum().T

    ordered_columns = [taxon for taxon in phylogeny_order if taxon in abundance.columns]
    if len(ordered_columns) == 0:
        raise ValueError("No overlapping infant taxa were found in the genus-level phylogeny.")
    abundance = abundance.loc[:, ordered_columns]

    metadata = metadata[metadata[group_column].isin(group_map.keys())].copy()

    shared_samples = abundance.index.intersection(metadata.index)
    abundance = abundance.loc[shared_samples].copy()
    metadata = metadata.loc[shared_samples].copy()

    abundance = abundance.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    metadata["dataset"] = "Infant"
    metadata["group"] = metadata[group_column].map(group_map)

    return abundance, metadata[["dataset", "group"]]


def load_ibd_dataset(abundance_path, metadata_path, phylogeny_order):
    sample_column = "sample_id"
    group_column = "label"
    group_map = {"healthy": "Healthy", "disease": "IBD"}

    abundance = pd.read_csv(abundance_path, low_memory=False)
    metadata = pd.read_csv(metadata_path, low_memory=False)

    if sample_column not in abundance.columns:
        raise ValueError(f"Sample column '{sample_column}' was not found in IBD abundance table.")
    if sample_column not in metadata.columns:
        raise ValueError(f"Sample column '{sample_column}' was not found in IBD metadata table.")
    if group_column not in metadata.columns:
        raise ValueError(f"Group column '{group_column}' was not found in IBD metadata table.")

    abundance[sample_column] = abundance[sample_column].astype(str)
    metadata[sample_column] = metadata[sample_column].astype(str)

    abundance = abundance.set_index(sample_column)
    metadata = metadata.set_index(sample_column)

    abundance.columns = [collapse_to_genus(column) for column in abundance.columns]
    abundance = abundance.loc[:, [column != "" for column in abundance.columns]]
    abundance = abundance.T.groupby(level=0).sum().T

    ordered_columns = [taxon for taxon in phylogeny_order if taxon in abundance.columns]
    if len(ordered_columns) == 0:
        raise ValueError("No overlapping IBD taxa were found in the genus-level phylogeny.")
    abundance = abundance.loc[:, ordered_columns]

    metadata = metadata[metadata[group_column].isin(group_map.keys())].copy()

    shared_samples = abundance.index.intersection(metadata.index)
    abundance = abundance.loc[shared_samples].copy()
    metadata = metadata.loc[shared_samples].copy()

    abundance = abundance.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    metadata["dataset"] = "IBD"
    metadata["group"] = metadata[group_column].map(group_map)

    return abundance, metadata[["dataset", "group"]]


def compute_macro_micro_scores(
    abundance,
    metadata,
    pseudocount,
    fmax,
    low_fraction,
    high_fraction,
):
    abundance = abundance.loc[:, abundance.sum(axis=0) > 0].copy()
    row_sums = abundance.sum(axis=1)
    abundance = abundance.loc[row_sums > 0].div(row_sums[row_sums > 0], axis=0)
    metadata = metadata.loc[abundance.index].copy()

    n_taxa = abundance.shape[1]
    if n_taxa < 8:
        raise ValueError(f"Too few taxa remained for {metadata['dataset'].iloc[0]}: {n_taxa}.")

    log_abundance = np.log(abundance + pseudocount)
    clr_abundance = log_abundance.sub(log_abundance.mean(axis=1), axis=0)

    window = np.hanning(n_taxa).astype(float)
    frequency = np.fft.rfftfreq(n_taxa, d=1.0)
    fmin = 2.0 / n_taxa
    mode_mask = (frequency >= fmin) & (frequency <= fmax)
    modes = np.where(mode_mask)[0]

    if len(modes) < 6:
        raise ValueError(f"Too few frequency modes remained for {metadata['dataset'].iloc[0]}: {len(modes)}.")

    n_low = max(2, int(np.ceil(len(modes) * low_fraction)))
    n_high = max(2, int(np.ceil(len(modes) * high_fraction)))

    low_modes = modes[:n_low]
    high_modes = modes[-n_high:]
    middle_modes = np.array([mode for mode in modes if mode not in set(low_modes) and mode not in set(high_modes)])

    records = []

    for sample_id in clr_abundance.index:
        transformed = clr_abundance.loc[sample_id].to_numpy(dtype=float) * window
        power = (np.abs(np.fft.rfft(transformed)) ** 2) / n_taxa

        low_power = float(np.nansum(power[low_modes]))
        middle_power = float(np.nansum(power[middle_modes])) if len(middle_modes) else 0.0
        high_power = float(np.nansum(power[high_modes]))

        records.append(
            {
                "sample_id": sample_id,
                "dataset": metadata.loc[sample_id, "dataset"],
                "group": metadata.loc[sample_id, "group"],
                "low_power": low_power,
                "middle_power": middle_power,
                "high_power": high_power,
                "low_frequency_macro_organization_raw": np.log10(
                    (low_power + 1e-30) / (middle_power + high_power + 1e-30)
                ),
                "high_frequency_micro_fragmentation_raw": np.log10(
                    (high_power + 1e-30) / (low_power + 1e-30)
                ),
            }
        )

    return pd.DataFrame(records)


def load_all_scores(args):
    phylogeny_order = load_phylogeny_order(args.phylogeny)

    datasets = [
        load_palleja_dataset(args.palleja_abundance, args.palleja_metadata),
        load_infant_dataset(args.infant_abundance, args.infant_metadata, phylogeny_order),
        load_ibd_dataset(args.ibd_abundance, args.ibd_metadata, phylogeny_order),
    ]

    score_tables = []

    for abundance, metadata in datasets:
        scores = compute_macro_micro_scores(
            abundance=abundance,
            metadata=metadata,
            pseudocount=args.pseudocount,
            fmax=args.fmax,
            low_fraction=args.low_fraction,
            high_fraction=args.high_fraction,
        )
        scores["low_frequency_macro_organization"] = robust_scale(
            scores["low_frequency_macro_organization_raw"]
        ).to_numpy()
        scores["high_frequency_micro_fragmentation"] = robust_scale(
            scores["high_frequency_micro_fragmentation_raw"]
        ).to_numpy()
        score_tables.append(scores)

    scores = pd.concat(score_tables, axis=0, ignore_index=True)

    if args.contrast_stretch:
        scores["low_frequency_macro_organization"] = stretch_unit_interval(
            scores["low_frequency_macro_organization"], strength=7.0
        ).to_numpy()
        scores["high_frequency_micro_fragmentation"] = stretch_unit_interval(
            scores["high_frequency_micro_fragmentation"], strength=6.0
        ).to_numpy()

    return scores


def add_confidence_ellipse(axis, x_values, y_values, color, alpha=0.09, linewidth=0.9, n_std=1.22, zorder=1):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[mask]
    y_values = y_values[mask]

    if len(x_values) < 5:
        return

    covariance = np.cov(x_values, y_values)
    if not np.all(np.isfinite(covariance)):
        return

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 1e-12)
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    ellipse = Ellipse(
        xy=(np.nanmedian(x_values), np.nanmedian(y_values)),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    axis.add_patch(ellipse)


def add_arrow(axis, start, end, color, linewidth=2.25, curvature=0.0, zorder=26):
    arrow = FancyArrowPatch(
        posA=start,
        posB=end,
        arrowstyle="-|>",
        connectionstyle=f"arc3,rad={curvature}",
        mutation_scale=11,
        linewidth=linewidth,
        color=color,
        alpha=0.92,
        shrinkA=9,
        shrinkB=9,
        zorder=zorder,
    )
    axis.add_patch(arrow)


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#2F2330")
    axis.spines["bottom"].set_color("#2F2330")
    axis.tick_params(colors="#2F2330")
    axis.grid(True, color="#E8E2D8", linewidth=0.7, alpha=0.85)
    axis.set_axisbelow(True)


def compute_group_centers(scores):
    return (
        scores.groupby(["dataset", "group"], as_index=False)
        .agg(
            low_frequency_macro_organization=("low_frequency_macro_organization", "median"),
            high_frequency_micro_fragmentation=("high_frequency_micro_fragmentation", "median"),
            n_samples=("sample_id", "size"),
        )
    )


def plot_macro_micro_space(scores, output_dir, dpi, max_points_per_group, random_state):
    dataset_colors = {
        "Palleja": "#7E9FBE",
        "Infant": "#9B7A4A",
        "IBD": "#BE8A86",
    }
    dataset_markers = {
        "Palleja": "o",
        "Infant": "^",
        "IBD": "D",
    }
    group_colors = {
        "Pre-stable": "#8BB8E8",
        "Perturbation": "#4C78A8",
        "Recovery": "#1F5C99",
        "Immature": "#E3A137",
        "Mature": "#B97A18",
        "Healthy": "#E79A92",
        "IBD": "#C9453B",
    }
    arrow_colors = {
        "Palleja": "#3972B3",
        "Infant": "#D99128",
        "IBD": "#C9453B",
    }
    label_offsets = {
        "Pre-stable": (0.010, 0.018, "left"),
        "Perturbation": (0.012, 0.020, "left"),
        "Recovery": (0.010, -0.022, "left"),
        "Immature": (0.010, 0.014, "left"),
        "Mature": (0.010, 0.018, "left"),
        "Healthy": (0.010, -0.020, "left"),
        "IBD": (0.010, 0.020, "left"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    random_generator = np.random.default_rng(random_state)

    figure, axis = plt.subplots(figsize=(6.25, 4.95))
    axis.set_facecolor("#FBF8F3")

    axis.fill_between([0.0, 0.38], [0.66, 0.66], [1.06, 1.06], color="#F2B37F", alpha=0.115, linewidth=0, zorder=0)
    axis.fill_between([0.58, 1.08], [-0.04, -0.04], [0.34, 0.34], color="#82B6D9", alpha=0.12, linewidth=0, zorder=0)
    axis.text(0.050, 0.985, "micro-fragmented\n/ unstable", color="#B25C36", fontsize=7.4, ha="left", va="top", alpha=0.88)
    axis.text(0.945, 0.075, "macro-structured\nstability basin", color="#3576A7", fontsize=7.4, ha="right", va="bottom", alpha=0.92)

    for dataset in ["Palleja", "Infant", "IBD"]:
        dataset_scores = scores[scores["dataset"] == dataset]
        color = dataset_colors[dataset]
        marker = dataset_markers[dataset]

        add_confidence_ellipse(
            axis,
            dataset_scores["low_frequency_macro_organization"],
            dataset_scores["high_frequency_micro_fragmentation"],
            color=color,
        )

        for _, group_scores in dataset_scores.groupby("group"):
            if len(group_scores) > max_points_per_group:
                group_scores = group_scores.sample(max_points_per_group, random_state=random_state)

            x_values = (
                group_scores["low_frequency_macro_organization"].to_numpy()
                + random_generator.normal(0, 0.004, len(group_scores))
            ).clip(0, 1)
            y_values = (
                group_scores["high_frequency_micro_fragmentation"].to_numpy()
                + random_generator.normal(0, 0.004, len(group_scores))
            ).clip(0, 1)

            axis.scatter(
                x_values,
                y_values,
                s=20,
                marker=marker,
                c=color,
                alpha=0.16,
                edgecolor="none",
                zorder=2,
            )

    centers = compute_group_centers(scores)
    center_lookup = {
        dataset: table.set_index("group")
        for dataset, table in centers.groupby("dataset")
    }

    if "Palleja" in center_lookup:
        table = center_lookup["Palleja"]
        start = (
            table.loc["Pre-stable", "low_frequency_macro_organization"],
            table.loc["Pre-stable", "high_frequency_micro_fragmentation"],
        )
        middle = (
            table.loc["Perturbation", "low_frequency_macro_organization"],
            table.loc["Perturbation", "high_frequency_micro_fragmentation"],
        )
        end = (
            table.loc["Recovery", "low_frequency_macro_organization"],
            table.loc["Recovery", "high_frequency_micro_fragmentation"],
        )
        add_arrow(axis, start, middle, color=arrow_colors["Palleja"], curvature=0.24, zorder=27)
        add_arrow(axis, middle, end, color=arrow_colors["Palleja"], curvature=-0.22, zorder=27)

    if "Infant" in center_lookup:
        table = center_lookup["Infant"]
        start = (
            table.loc["Immature", "low_frequency_macro_organization"],
            table.loc["Immature", "high_frequency_micro_fragmentation"],
        )
        end = (
            table.loc["Mature", "low_frequency_macro_organization"],
            table.loc["Mature", "high_frequency_micro_fragmentation"],
        )
        add_arrow(axis, start, end, color=arrow_colors["Infant"], curvature=0.0, zorder=28)

    if "IBD" in center_lookup:
        table = center_lookup["IBD"]
        start = (
            table.loc["Healthy", "low_frequency_macro_organization"],
            table.loc["Healthy", "high_frequency_micro_fragmentation"],
        )
        end = (
            table.loc["IBD", "low_frequency_macro_organization"],
            table.loc["IBD", "high_frequency_micro_fragmentation"],
        )
        add_arrow(axis, start, end, color=arrow_colors["IBD"], curvature=0.0, zorder=29)

    for _, row in centers.iterrows():
        dataset = row["dataset"]
        group = row["group"]
        x_value = row["low_frequency_macro_organization"]
        y_value = row["high_frequency_micro_fragmentation"]
        marker = dataset_markers[dataset]
        color = group_colors.get(group, dataset_colors[dataset])

        axis.scatter(
            x_value,
            y_value,
            s=118,
            marker=marker,
            c=color,
            edgecolor="white",
            linewidth=1.1,
            zorder=32,
        )

        dx, dy, alignment = label_offsets.get(group, (0.012, 0.012, "left"))
        axis.text(
            x_value + dx,
            y_value + dy,
            group,
            fontsize=7.2,
            color=color,
            ha=alignment,
            va="center",
            zorder=35,
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.78},
        )

    axis.set_xlim(-0.035, 1.095)
    axis.set_ylim(-0.035, 1.085)
    axis.set_xlabel("Low-frequency macro-organization")
    axis.set_ylabel("High-frequency micro-fragmentation")
    axis.set_title("Macro-micro spectral space of host microbiomes", pad=10)

    style_axis(axis)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=dataset_markers[dataset],
            color="none",
            markerfacecolor=dataset_colors[dataset],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=7.3,
            label=dataset,
        )
        for dataset in ["Palleja", "Infant", "IBD"]
    ]

    axis.legend(
        handles=legend_handles,
        title="Dataset",
        frameon=True,
        fancybox=False,
        edgecolor="#E3D9CC",
        facecolor="white",
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        borderpad=0.45,
        handletextpad=0.45,
        labelspacing=0.35,
    )

    figure.savefig(output_dir / "host_macro_micro_spectral_space.png", bbox_inches="tight", dpi=dpi)
    figure.savefig(output_dir / "host_macro_micro_spectral_space.pdf", bbox_inches="tight")
    plt.close(figure)


def write_outputs(scores, output_dir, args):
    output_dir.mkdir(parents=True, exist_ok=True)

    scores.to_csv(output_dir / "host_macro_micro_scores.csv", index=False)
    compute_group_centers(scores).to_csv(output_dir / "host_macro_micro_group_centers.csv", index=False)

    parameter_table = pd.DataFrame(
        {
            "parameter": [
                "pseudocount",
                "fmax",
                "low_fraction",
                "high_fraction",
                "contrast_stretch",
                "max_points_per_group",
                "random_state",
            ],
            "value": [
                args.pseudocount,
                args.fmax,
                args.low_fraction,
                args.high_fraction,
                args.contrast_stretch,
                args.max_points_per_group,
                args.random_state,
            ],
        }
    )
    parameter_table.to_csv(output_dir / "analysis_parameters.csv", index=False)


def main():
    args = parse_arguments()
    configure_matplotlib(args.dpi)

    scores = load_all_scores(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_outputs(scores, args.output_dir, args)

    plot_macro_micro_space(
        scores=scores,
        output_dir=args.output_dir,
        dpi=args.dpi,
        max_points_per_group=args.max_points_per_group,
        random_state=args.random_state,
    )

    print(f"Samples retained: {scores.shape[0]}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
