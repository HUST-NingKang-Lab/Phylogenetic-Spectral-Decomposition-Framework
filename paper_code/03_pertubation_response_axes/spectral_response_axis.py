from pathlib import Path
import argparse
import re
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Construct dataset-specific phylogenetic spectral response axes for ETEC and O'Keefe perturbation analyses."
    )
    parser.add_argument("--etec-abundance", type=Path, default=Path("data/etec/etec_abundance_genus.csv"))
    parser.add_argument("--etec-metadata", type=Path, default=Path("data/etec/etec_metadata.csv"))
    parser.add_argument("--okeefe-abundance", type=Path, default=Path("data/okeefe/okeefe_dietswap_abundance.csv"))
    parser.add_argument("--okeefe-metadata", type=Path, default=Path("data/okeefe/okeefe_dietswap_metadata.csv"))
    parser.add_argument("--phylogeny", type=Path, default=Path("data/phylogeny.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/03_spectral_seesaw_response/spectral_response_axis"))
    parser.add_argument("--pseudocount", type=float, default=1e-6)
    parser.add_argument("--fmax", type=float, default=0.45)
    parser.add_argument("--window", choices=["hann", "none"], default="hann")
    parser.add_argument("--min-prevalence", type=float, default=0.02)
    parser.add_argument("--min-total-count", type=float, default=10.0)
    parser.add_argument("--min-taxa", type=int, default=20)
    parser.add_argument("--pole-quantile", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=20260428)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--save-tables", action="store_true")
    return parser.parse_args()


def configure_plotting(dpi=600):
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": dpi,
            "figure.dpi": 160,
        }
    )


def normalize_taxon_label(value):
    corrections = {
        "Allistipes": "Alistipes",
        "Klebisiella": "Klebsiella",
        "Fusobacteria": "Fusobacterium",
        "Escherichia/Shigella": "Escherichia",
        "Clostridiales": "Clostridium",
    }

    value = str(value).strip().strip('"').strip("'").replace("|", ";")

    if value.lower() in {"nan", "none", "uncultured", "outgrouping", "incertae", ""}:
        return ""

    if ";" in value:
        value = value.split(";")[-1]

    for prefix in ["sk__", "k__", "p__", "c__", "o__", "f__", "g__", "s__"]:
        if value.startswith(prefix):
            value = value[len(prefix):]

    value = value.replace(" et rel.", "").replace(" et rel", "")
    value = re.sub(r"\s+cluster.*$", "", value, flags=re.I)
    value = re.sub(r"\s+group.*$", "", value, flags=re.I)
    value = re.sub(r"\s+sensu.*$", "", value, flags=re.I)
    value = value.strip()

    if " " in value:
        value = value.split()[0]

    return corrections.get(value, value)


def read_phylogenetic_order(path):
    phylogeny = pd.read_csv(path, low_memory=False)
    order = []
    seen = set()

    for raw_value in phylogeny.iloc[:, 0].astype(str):
        taxon = normalize_taxon_label(raw_value)
        if taxon and taxon not in seen:
            order.append(taxon)
            seen.add(taxon)

    return order


def read_abundance_table(path):
    table = pd.read_csv(path, index_col=0, low_memory=False)
    table.index = [normalize_taxon_label(taxon) for taxon in table.index]
    table = table.loc[[taxon != "" for taxon in table.index]]
    table = table.groupby(table.index).sum()
    abundance = table.T
    abundance.index = abundance.index.astype(str)
    abundance = abundance.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return abundance


def read_etec_metadata(path):
    metadata = pd.read_csv(path, low_memory=False)
    sample_column = metadata.columns[0]
    metadata = metadata.rename(columns={sample_column: "sample"})
    metadata["sample"] = metadata["sample"].astype(str)
    metadata["SubjectID"] = metadata["SubjectID"].astype(str)
    metadata["Day"] = pd.to_numeric(metadata["Day"], errors="coerce")

    def assign_phase(day):
        if day == -1:
            return "Baseline"
        if day == 0:
            return "Challenge"
        if day in [1, 2, 3, 4, 5, 6, 7, 9]:
            return "Acute"
        if day in [28, 84]:
            return "Recovery"
        return "Other"

    metadata["phase"] = metadata["Day"].apply(assign_phase)
    return metadata.set_index("sample")


def read_okeefe_metadata(path):
    metadata = pd.read_csv(path, low_memory=False)
    required_columns = ["sample", "subject", "nationality", "group"]

    missing = [column for column in required_columns if column not in metadata.columns]
    if missing:
        raise ValueError(f"Missing columns in O'Keefe metadata table: {missing}")

    metadata["sample"] = metadata["sample"].astype(str)
    metadata["subject"] = metadata["subject"].astype(str)
    metadata["nationality"] = metadata["nationality"].astype(str)
    metadata["group"] = metadata["group"].astype(str)

    if "timepoint" in metadata.columns:
        metadata["timepoint"] = pd.to_numeric(metadata["timepoint"], errors="coerce")

    return metadata.set_index("sample")


def align_samples(abundance, metadata):
    shared_samples = abundance.index.intersection(metadata.index)
    if len(shared_samples) == 0:
        raise ValueError("No overlapping samples were found between abundance and metadata tables.")
    return abundance.loc[shared_samples].copy(), metadata.loc[shared_samples].copy()


def filter_taxa(abundance, min_prevalence, min_total_count):
    prevalence = (abundance > 0).mean(axis=0)
    total_count = abundance.sum(axis=0)
    keep = (prevalence >= min_prevalence) & (total_count >= min_total_count)
    return abundance.loc[:, keep].copy()


def order_taxa(abundance, phylogenetic_order, dataset_name, min_taxa):
    taxa = [taxon for taxon in phylogenetic_order if taxon in abundance.columns]
    if len(taxa) < min_taxa:
        raise ValueError(f"{dataset_name}: only {len(taxa)} taxa matched the phylogenetic order.")
    return taxa


def centered_log_ratio(abundance, pseudocount):
    abundance = abundance.loc[abundance.sum(axis=1) > 0].copy()
    relative_abundance = abundance.div(abundance.sum(axis=1), axis=0).fillna(0.0) + pseudocount
    log_abundance = np.log(relative_abundance)
    return log_abundance.sub(log_abundance.mean(axis=1), axis=0)


def spectral_features(clr_abundance, fmax, window_type):
    n_taxa = clr_abundance.shape[1]

    if window_type == "hann":
        window = np.hanning(n_taxa)
    elif window_type == "none":
        window = np.ones(n_taxa)
    else:
        raise ValueError("window_type must be 'hann' or 'none'.")

    frequency = np.fft.rfftfreq(n_taxa, d=1.0)
    coefficients = np.fft.rfft(clr_abundance.values.astype(float) * window[None, :], axis=1)

    frequency_table = pd.DataFrame(
        {
            "frequency_id": [f"f{i:03d}" for i in range(len(frequency))],
            "frequency_index": np.arange(len(frequency)),
            "frequency": frequency,
        }
    )
    retained_frequencies = frequency_table[
        (frequency_table["frequency"] > 0) & (frequency_table["frequency"] <= fmax)
    ].copy()

    retained_indices = retained_frequencies["frequency_index"].to_numpy()
    retained_coefficients = coefficients[:, retained_indices]
    features = np.concatenate([retained_coefficients.real, retained_coefficients.imag], axis=1)

    columns = (
        [f"real_{frequency_id}" for frequency_id in retained_frequencies["frequency_id"]]
        + [f"imag_{frequency_id}" for frequency_id in retained_frequencies["frequency_id"]]
    )

    return (
        pd.DataFrame(features, index=clr_abundance.index, columns=columns),
        frequency_table,
        retained_frequencies,
    )


def feature_scale(matrix):
    median = matrix.median(axis=0)
    mad = (matrix - median).abs().median(axis=0)
    standard_deviation = matrix.std(axis=0)
    scale = mad.copy()
    invalid = (~np.isfinite(scale)) | (scale < 1e-12)
    scale.loc[invalid] = standard_deviation.loc[invalid]
    scale[(~np.isfinite(scale)) | (scale < 1e-12)] = 1.0
    return scale


def normalize_vector(values):
    values = np.asarray(values, dtype=float)
    norm = np.linalg.norm(values)
    if norm <= 1e-12:
        return values * np.nan
    return values / norm


def cosine_similarity(first_values, second_values):
    first_values = np.asarray(first_values, dtype=float)
    second_values = np.asarray(second_values, dtype=float)
    denominator = np.linalg.norm(first_values) * np.linalg.norm(second_values)
    if denominator <= 1e-12:
        return np.nan
    return float(first_values.dot(second_values) / denominator)


def project_on_axis(matrix, axis):
    return pd.Series(matrix[axis.index].values @ axis.values, index=matrix.index)


def baseline_deltas_etec(features, metadata):
    baseline = metadata[metadata["Day"] == -1]
    baseline_subjects = set(baseline["SubjectID"])
    rows = []
    sample_ids = []

    for sample_id in features.index:
        subject = metadata.loc[sample_id, "SubjectID"]
        if subject not in baseline_subjects:
            continue
        baseline_ids = baseline[baseline["SubjectID"] == subject].index
        baseline_vector = features.loc[baseline_ids].mean(axis=0).values
        rows.append(features.loc[sample_id].values - baseline_vector)
        sample_ids.append(sample_id)

    return pd.DataFrame(rows, index=sample_ids, columns=features.columns), metadata.loc[sample_ids].copy()


def baseline_deltas_okeefe(features, metadata):
    baseline = metadata[metadata["group"] == "HE"]
    baseline_subjects = set(baseline["subject"])
    rows = []
    sample_ids = []

    for sample_id in features.index:
        subject = metadata.loc[sample_id, "subject"]
        if subject not in baseline_subjects:
            continue
        baseline_ids = baseline[baseline["subject"] == subject].index
        baseline_vector = features.loc[baseline_ids].mean(axis=0).values
        rows.append(features.loc[sample_id].values - baseline_vector)
        sample_ids.append(sample_id)

    return pd.DataFrame(rows, index=sample_ids, columns=features.columns), metadata.loc[sample_ids].copy()


def mean_by_subject(matrix, metadata, subject_column, mask):
    sample_ids = metadata.index[mask]
    subset = matrix.loc[sample_ids].copy()
    subset["subject"] = metadata.loc[sample_ids, subject_column].values
    return subset.groupby("subject").mean()


def inverse_project_axis(axis, scale, retained_frequencies, frequency_table, n_taxa, taxa_order):
    raw_axis = axis / scale.loc[axis.index]
    n_retained = len(retained_frequencies)

    real_values = raw_axis.iloc[:n_retained].values
    imaginary_values = raw_axis.iloc[n_retained:2 * n_retained].values

    coefficients = np.zeros(len(frequency_table), dtype=complex)
    frequency_to_index = dict(zip(frequency_table["frequency_id"], frequency_table["frequency_index"]))

    for frequency_id, real_value, imaginary_value in zip(
        retained_frequencies["frequency_id"], real_values, imaginary_values
    ):
        coefficients[frequency_to_index[frequency_id]] = real_value + 1j * imaginary_value

    weights = np.fft.irfft(coefficients, n=n_taxa)
    weights = (weights - np.mean(weights)) / (np.std(weights) + 1e-12)

    return pd.DataFrame(
        {
            "taxon_order": np.arange(1, n_taxa + 1),
            "taxon": taxa_order,
            "axis_weight": weights,
        }
    )


def extract_axis_poles(axis_weights, quantile):
    upper = axis_weights["axis_weight"].quantile(1 - quantile)
    lower = axis_weights["axis_weight"].quantile(quantile)
    positive = axis_weights.loc[axis_weights["axis_weight"] >= upper, "taxon"].tolist()
    negative = axis_weights.loc[axis_weights["axis_weight"] <= lower, "taxon"].tolist()
    return positive, negative


def compute_pole_balance(clr_abundance, positive_taxa, negative_taxa):
    positive_taxa = [taxon for taxon in positive_taxa if taxon in clr_abundance.columns]
    negative_taxa = [taxon for taxon in negative_taxa if taxon in clr_abundance.columns]

    output = pd.DataFrame(index=clr_abundance.index)
    output["positive_pole_mean_clr"] = clr_abundance[positive_taxa].mean(axis=1)
    output["negative_pole_mean_clr"] = clr_abundance[negative_taxa].mean(axis=1)
    output["pole_balance"] = output["positive_pole_mean_clr"] - output["negative_pole_mean_clr"]
    return output


def validate_etec_poles(clr_abundance, metadata, positive_taxa, negative_taxa):
    rows = []

    for subject, subject_metadata in metadata.groupby("SubjectID"):
        baseline_ids = subject_metadata[subject_metadata["Day"] == -1].index
        acute_ids = subject_metadata[subject_metadata["phase"] == "Acute"].index
        recovery_ids = subject_metadata[subject_metadata["phase"] == "Recovery"].index

        if len(baseline_ids) and len(acute_ids):
            delta_positive = (
                clr_abundance.loc[acute_ids, positive_taxa].mean(axis=0).mean()
                - clr_abundance.loc[baseline_ids, positive_taxa].mean(axis=0).mean()
            )
            delta_negative = (
                clr_abundance.loc[acute_ids, negative_taxa].mean(axis=0).mean()
                - clr_abundance.loc[baseline_ids, negative_taxa].mean(axis=0).mean()
            )
            rows.append(
                {
                    "dataset": "ETEC",
                    "response": "acute_minus_baseline",
                    "subject": subject,
                    "delta_positive": delta_positive,
                    "delta_negative": delta_negative,
                    "delta_balance": delta_positive - delta_negative,
                    "oriented_balance": delta_positive - delta_negative,
                }
            )

        if len(acute_ids) and len(recovery_ids):
            delta_positive = (
                clr_abundance.loc[recovery_ids, positive_taxa].mean(axis=0).mean()
                - clr_abundance.loc[acute_ids, positive_taxa].mean(axis=0).mean()
            )
            delta_negative = (
                clr_abundance.loc[recovery_ids, negative_taxa].mean(axis=0).mean()
                - clr_abundance.loc[acute_ids, negative_taxa].mean(axis=0).mean()
            )
            rows.append(
                {
                    "dataset": "ETEC",
                    "response": "recovery_minus_acute",
                    "subject": subject,
                    "delta_positive": delta_positive,
                    "delta_negative": delta_negative,
                    "delta_balance": delta_positive - delta_negative,
                    "oriented_balance": -(delta_positive - delta_negative),
                }
            )

    return pd.DataFrame(rows)


def validate_okeefe_poles(clr_abundance, metadata, positive_taxa, negative_taxa):
    rows = []

    for subject, subject_metadata in metadata.groupby("subject"):
        baseline_ids = subject_metadata[subject_metadata["group"] == "HE"].index
        intervention_ids = subject_metadata[subject_metadata["group"] == "DI"].index

        if not (len(baseline_ids) and len(intervention_ids)):
            continue

        nationality = str(subject_metadata["nationality"].iloc[0])
        delta_positive = (
            clr_abundance.loc[intervention_ids, positive_taxa].mean(axis=0).mean()
            - clr_abundance.loc[baseline_ids, positive_taxa].mean(axis=0).mean()
        )
        delta_negative = (
            clr_abundance.loc[intervention_ids, negative_taxa].mean(axis=0).mean()
            - clr_abundance.loc[baseline_ids, negative_taxa].mean(axis=0).mean()
        )
        direction = 1 if nationality == "AFR" else -1

        rows.append(
            {
                "dataset": "OKeefe",
                "response": f"{nationality}_DI_minus_HE",
                "subject": subject,
                "nationality": nationality,
                "delta_positive": delta_positive,
                "delta_negative": delta_negative,
                "delta_balance": delta_positive - delta_negative,
                "oriented_balance": direction * (delta_positive - delta_negative),
            }
        )

    return pd.DataFrame(rows)


def summarize_pole_validation(validation_table):
    rows = []

    for response, subset in validation_table.groupby("response"):
        if len(subset) >= 3:
            rho, p_value = spearmanr(subset["delta_positive"], subset["delta_negative"])
        else:
            rho, p_value = np.nan, np.nan

        try:
            _, wilcoxon_p = wilcoxon(subset["oriented_balance"], alternative="greater", zero_method="wilcox")
        except Exception:
            wilcoxon_p = np.nan

        rows.append(
            {
                "response": response,
                "n": len(subset),
                "median_delta_positive": np.nanmedian(subset["delta_positive"]),
                "median_delta_negative": np.nanmedian(subset["delta_negative"]),
                "median_oriented_balance": np.nanmedian(subset["oriented_balance"]),
                "wilcoxon_p_oriented_greater": wilcoxon_p,
                "spearman_positive_negative_rho": rho,
                "spearman_positive_negative_p": p_value,
            }
        )

    return pd.DataFrame(rows)


def axis_frequency_profile(axis, scale, retained_frequencies):
    raw_axis = axis / scale.loc[axis.index]
    n_retained = len(retained_frequencies)
    real_values = raw_axis.iloc[:n_retained].values
    imaginary_values = raw_axis.iloc[n_retained:2 * n_retained].values
    loading = np.sqrt(real_values ** 2 + imaginary_values ** 2)
    loading = loading / (loading.sum() + 1e-12)

    return pd.DataFrame(
        {
            "frequency": retained_frequencies["frequency"].values,
            "loading": loading,
        }
    )


def build_etec_analysis(abundance, metadata, taxa_order, args):
    abundance = abundance.loc[:, taxa_order].copy()
    clr_abundance = centered_log_ratio(abundance, args.pseudocount)
    metadata = metadata.loc[clr_abundance.index].copy()

    features, frequency_table, retained_frequencies = spectral_features(
        clr_abundance,
        fmax=args.fmax,
        window_type=args.window,
    )

    raw_delta, aligned_metadata = baseline_deltas_etec(features, metadata)
    scale = feature_scale(raw_delta.loc[~aligned_metadata["phase"].eq("Baseline")])
    delta = raw_delta / scale.loc[raw_delta.columns]

    acute = mean_by_subject(delta, aligned_metadata, "SubjectID", aligned_metadata["phase"].eq("Acute"))
    recovery = mean_by_subject(delta, aligned_metadata, "SubjectID", aligned_metadata["phase"].eq("Recovery"))
    shared_subjects = acute.index.intersection(recovery.index)
    recovery_minus_acute = recovery.loc[shared_subjects] - acute.loc[shared_subjects]

    acute_direction = normalize_vector(acute.mean(axis=0).values)
    recovery_direction = normalize_vector(recovery_minus_acute.mean(axis=0).values)
    axis_values = normalize_vector(acute_direction - recovery_direction)
    axis = pd.Series(axis_values, index=delta.columns, name="etec_perturbation_recovery_axis")

    axis_scores = aligned_metadata.copy()
    axis_scores["axis_score"] = project_on_axis(delta, axis)

    response_points = []
    for response, matrix in {
        "acute_minus_baseline": acute,
        "recovery_minus_acute": recovery_minus_acute,
    }.items():
        table = matrix.copy()
        table["response"] = response
        table["subject"] = table.index
        response_points.append(table.reset_index(drop=True))

    response_points = pd.concat(response_points, ignore_index=True)
    pca = PCA(n_components=2, random_state=args.random_state)
    coordinates = pca.fit_transform(response_points[delta.columns].values)
    response_points["PC1"] = coordinates[:, 0]
    response_points["PC2"] = coordinates[:, 1]

    weights = inverse_project_axis(axis, scale, retained_frequencies, frequency_table, len(taxa_order), taxa_order)
    positive_taxa, negative_taxa = extract_axis_poles(weights, args.pole_quantile)
    pole_balance = compute_pole_balance(clr_abundance, positive_taxa, negative_taxa).join(metadata)
    subject_validation = validate_etec_poles(clr_abundance, metadata, positive_taxa, negative_taxa)

    return {
        "name": "ETEC",
        "taxa_order": taxa_order,
        "clr_abundance": clr_abundance,
        "metadata": metadata,
        "delta": delta,
        "scale": scale,
        "frequency_table": frequency_table,
        "retained_frequencies": retained_frequencies,
        "axis": axis,
        "axis_scores": axis_scores,
        "response_points": response_points,
        "pca_explained_variance": pca.explained_variance_ratio_,
        "acute_direction": pd.Series(acute_direction, index=delta.columns),
        "recovery_direction": pd.Series(recovery_direction, index=delta.columns),
        "response_cosine": cosine_similarity(acute_direction, recovery_direction),
        "weights": weights,
        "positive_taxa": positive_taxa,
        "negative_taxa": negative_taxa,
        "pole_balance": pole_balance,
        "subject_validation": subject_validation,
        "frequency_profile": axis_frequency_profile(axis, scale, retained_frequencies),
    }


def build_okeefe_analysis(abundance, metadata, taxa_order, args):
    metadata = metadata[metadata["group"].isin(["HE", "DI"])].copy()
    abundance = abundance.loc[metadata.index, taxa_order].copy()

    clr_abundance = centered_log_ratio(abundance, args.pseudocount)
    metadata = metadata.loc[clr_abundance.index].copy()

    features, frequency_table, retained_frequencies = spectral_features(
        clr_abundance,
        fmax=args.fmax,
        window_type=args.window,
    )

    raw_delta, aligned_metadata = baseline_deltas_okeefe(features, metadata)
    scale = feature_scale(raw_delta.loc[~aligned_metadata["group"].eq("HE")])
    delta = raw_delta / scale.loc[raw_delta.columns]

    response_matrices = {}

    for nationality in ["AFR", "AAM"]:
        response_matrices[nationality] = mean_by_subject(
            delta,
            aligned_metadata,
            "subject",
            aligned_metadata["nationality"].eq(nationality) & aligned_metadata["group"].eq("DI"),
        )

    afr_direction = normalize_vector(response_matrices["AFR"].mean(axis=0).values)
    aam_direction = normalize_vector(response_matrices["AAM"].mean(axis=0).values)
    axis_values = normalize_vector(afr_direction - aam_direction)
    axis = pd.Series(axis_values, index=delta.columns, name="okeefe_reciprocal_diet_axis")

    axis_scores = aligned_metadata.copy()
    axis_scores["axis_score"] = project_on_axis(delta, axis)

    response_points = []

    for nationality, matrix in response_matrices.items():
        table = matrix.copy()
        table["response"] = f"{nationality}_DI_minus_HE"
        table["subject"] = table.index
        response_points.append(table.reset_index(drop=True))

    response_points = pd.concat(response_points, ignore_index=True)
    pca = PCA(n_components=2, random_state=args.random_state)
    coordinates = pca.fit_transform(response_points[delta.columns].values)
    response_points["PC1"] = coordinates[:, 0]
    response_points["PC2"] = coordinates[:, 1]

    weights = inverse_project_axis(axis, scale, retained_frequencies, frequency_table, len(taxa_order), taxa_order)
    positive_taxa, negative_taxa = extract_axis_poles(weights, args.pole_quantile)
    pole_balance = compute_pole_balance(clr_abundance, positive_taxa, negative_taxa).join(metadata)
    subject_validation = validate_okeefe_poles(clr_abundance, metadata, positive_taxa, negative_taxa)

    return {
        "name": "OKeefe",
        "taxa_order": taxa_order,
        "clr_abundance": clr_abundance,
        "metadata": metadata,
        "delta": delta,
        "scale": scale,
        "frequency_table": frequency_table,
        "retained_frequencies": retained_frequencies,
        "axis": axis,
        "axis_scores": axis_scores,
        "response_points": response_points,
        "pca_explained_variance": pca.explained_variance_ratio_,
        "afr_direction": pd.Series(afr_direction, index=delta.columns),
        "aam_direction": pd.Series(aam_direction, index=delta.columns),
        "response_cosine": cosine_similarity(afr_direction, aam_direction),
        "weights": weights,
        "positive_taxa": positive_taxa,
        "negative_taxa": negative_taxa,
        "pole_balance": pole_balance,
        "subject_validation": subject_validation,
        "frequency_profile": axis_frequency_profile(axis, scale, retained_frequencies),
    }


def build_analyses(args):
    phylogenetic_order = read_phylogenetic_order(args.phylogeny)

    etec_abundance = read_abundance_table(args.etec_abundance)
    etec_metadata = read_etec_metadata(args.etec_metadata)
    etec_abundance, etec_metadata = align_samples(etec_abundance, etec_metadata)
    etec_abundance = filter_taxa(etec_abundance, args.min_prevalence, args.min_total_count)
    etec_taxa = order_taxa(etec_abundance, phylogenetic_order, "ETEC", args.min_taxa)
    etec = build_etec_analysis(etec_abundance, etec_metadata, etec_taxa, args)

    okeefe_abundance = read_abundance_table(args.okeefe_abundance)
    okeefe_metadata = read_okeefe_metadata(args.okeefe_metadata)
    okeefe_abundance, okeefe_metadata = align_samples(okeefe_abundance, okeefe_metadata)
    okeefe_abundance = filter_taxa(okeefe_abundance, args.min_prevalence, args.min_total_count)
    okeefe_taxa = order_taxa(okeefe_abundance, phylogenetic_order, "OKeefe", args.min_taxa)
    okeefe = build_okeefe_analysis(okeefe_abundance, okeefe_metadata, okeefe_taxa, args)

    return etec, okeefe


def format_p_value(value):
    if not np.isfinite(value):
        return "p=NA"
    if value < 1e-4:
        return f"p={value:.1e}"
    if value < 1e-3:
        return f"p={value:.1e}"
    return f"p={value:.3g}"


def style_axis(axis, grid=True):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", width=0.65, length=3)
    if grid:
        axis.grid(axis="y", color="#EEECE5", linewidth=0.55, zorder=0)
    axis.set_axisbelow(True)


def add_panel_label(axis, label):
    axis.text(-0.12, 1.08, label, transform=axis.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")


def plot_etec_axis(axis, analysis):
    scores = analysis["axis_scores"].copy()

    axis.axvspan(1, 9, color="#EEDFB7", alpha=0.35, linewidth=0, zorder=0)
    axis.axvspan(28, 84, color="#A8BCCC", alpha=0.12, linewidth=0, zorder=0)
    axis.axvline(0, color="#452A3D", linewidth=0.75, linestyle=":", alpha=0.65, zorder=1)

    for _, subset in scores.sort_values("Day").groupby("SubjectID"):
        axis.plot(subset["Day"], subset["axis_score"], color="#B7B5A0", linewidth=0.75, alpha=0.45, zorder=1)

    summary = (
        scores.groupby("Day")["axis_score"]
        .agg(
            median="median",
            q1=lambda values: np.quantile(values, 0.25),
            q3=lambda values: np.quantile(values, 0.75),
        )
        .reset_index()
        .sort_values("Day")
    )

    axis.fill_between(summary["Day"], summary["q1"], summary["q3"], color="#E5855D", alpha=0.30, linewidth=0, zorder=2)
    axis.plot(summary["Day"], summary["median"], color="#D44C3C", linewidth=2.6, zorder=4)
    axis.scatter(summary["Day"], summary["median"], s=12, color="#D44C3C", edgecolor="white", linewidth=0.35, zorder=5)
    axis.axhline(0, color="#452A3D", linewidth=0.75, linestyle="--", alpha=0.65)
    axis.set_xlabel("Day relative to ETEC challenge")
    axis.set_ylabel("ETEC spectral-axis score")
    axis.set_title("ETEC perturbation-recovery axis", pad=4)
    style_axis(axis)


def plot_okeefe_axis(axis, analysis):
    scores = analysis["axis_scores"].copy()
    groups = [("AFR", "#B66065", -0.08), ("AAM", "#44757A", 0.08)]

    for nationality, color, offset in groups:
        subset = scores[scores["nationality"] == nationality]
        summary = subset.groupby(["subject", "group"])["axis_score"].mean().reset_index()
        wide = summary.pivot(index="subject", columns="group", values="axis_score")
        positions = np.array([0, 1]) + offset

        for _, row in wide.iterrows():
            if "HE" in row.index and "DI" in row.index:
                axis.plot(positions, [row["HE"], row["DI"]], color=color, alpha=0.25, linewidth=0.8, zorder=1)
                axis.scatter(positions, [row["HE"], row["DI"]], color=color, alpha=0.18, s=8, edgecolor="none", zorder=1)

        medians = np.array([wide["HE"].median(), wide["DI"].median()])
        q1 = np.array([wide["HE"].quantile(0.25), wide["DI"].quantile(0.25)])
        q3 = np.array([wide["HE"].quantile(0.75), wide["DI"].quantile(0.75)])
        axis.fill_between(positions, q1, q3, color=color, alpha=0.12, linewidth=0, zorder=2)
        axis.plot(positions, medians, color=color, linewidth=2.7, label=nationality, zorder=4)
        axis.scatter(positions, medians, color=color, s=34, edgecolor="white", linewidth=0.5, zorder=5)

    axis.axhline(0, color="#452A3D", linewidth=0.75, linestyle="--", alpha=0.65)
    axis.set_xlim(-0.25, 1.25)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["HE", "DI"])
    axis.set_ylabel("O'Keefe spectral-axis score")
    axis.set_title("Reciprocal diet-response axis", pad=4)
    axis.legend(frameon=False, loc="upper left", handlelength=1.7, borderaxespad=0.2)
    style_axis(axis)


def plot_axis_weights(axis, analysis, title):
    weights = analysis["weights"].copy()
    positive_cutoff = weights["axis_weight"].quantile(0.85)
    negative_cutoff = weights["axis_weight"].quantile(0.15)

    axis.axhspan(positive_cutoff, weights["axis_weight"].max(), color="#E6D8E2", alpha=0.65, linewidth=0)
    axis.axhspan(weights["axis_weight"].min(), negative_cutoff, color="#D9E7E3", alpha=0.65, linewidth=0)
    axis.plot(weights["taxon_order"], weights["axis_weight"], color="#4C3A48", linewidth=1.0)
    axis.fill_between(
        weights["taxon_order"],
        0,
        weights["axis_weight"],
        where=weights["axis_weight"] >= positive_cutoff,
        color="#B987A5",
        alpha=0.85,
    )
    axis.fill_between(
        weights["taxon_order"],
        0,
        weights["axis_weight"],
        where=weights["axis_weight"] <= negative_cutoff,
        color="#86A9A6",
        alpha=0.85,
    )
    axis.axhline(0, color="#452A3D", linewidth=0.65, linestyle="--", alpha=0.65)
    axis.set_xlabel("Phylogenetic genus order")
    axis.set_ylabel("Axis weight")
    axis.set_title(title, pad=4)
    style_axis(axis)


def plot_oriented_balance(axis, etec, okeefe):
    validation = pd.concat(
        [
            etec["subject_validation"].assign(system="ETEC"),
            okeefe["subject_validation"].assign(system="OKeefe"),
        ],
        axis=0,
        ignore_index=True,
    )

    order = ["acute_minus_baseline", "recovery_minus_acute", "AFR_DI_minus_HE", "AAM_DI_minus_HE"]
    labels = ["ETEC acute", "ETEC recovery", "AFR diet", "AAM diet"]
    colors = ["#D77A61", "#6F95A0", "#C989AB", "#5E9F8B"]

    data = [validation.loc[validation["response"] == response, "oriented_balance"].dropna().values for response in order]
    positions = np.arange(1, len(order) + 1)

    box = axis.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#452A3D", "linewidth": 1.0},
        boxprops={"linewidth": 0.75},
        whiskerprops={"linewidth": 0.75},
        capprops={"linewidth": 0.75},
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)

    random_generator = np.random.default_rng(12)

    for position, values, color in zip(positions, data, colors):
        x_values = random_generator.normal(position, 0.055, len(values))
        axis.scatter(x_values, values, color=color, s=13, alpha=0.75, edgecolor="white", linewidth=0.2, zorder=3)

    axis.axhline(0, color="#452A3D", linewidth=0.75, linestyle="--", alpha=0.65)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.set_ylabel("Oriented pole-balance change")
    axis.set_title("Pole-balance response", pad=4)
    style_axis(axis)


def plot_positive_negative_response(axis, validation, title, colors):
    for response, subset in validation.groupby("response"):
        color = colors.get(response, "#777777")
        axis.scatter(
            subset["delta_positive"],
            subset["delta_negative"],
            s=22,
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.35,
            label=response.replace("_", " "),
            zorder=3,
        )

    if len(validation) >= 3:
        rho, p_value = spearmanr(validation["delta_positive"], validation["delta_negative"])
    else:
        rho, p_value = np.nan, np.nan

    axis.axhline(0, color="#E1DED3", linewidth=0.85, zorder=1)
    axis.axvline(0, color="#E1DED3", linewidth=0.85, zorder=1)
    axis.set_xlabel("Delta positive pole")
    axis.set_ylabel("Delta negative pole")
    axis.set_title(f"{title}\nSpearman rho={rho:.2f}, {format_p_value(p_value)}", pad=4)
    axis.legend(frameon=False, loc="best", handletextpad=0.25, labelspacing=0.25)
    style_axis(axis, grid=False)


def create_integrated_figure(etec, okeefe, output_dir, dpi):
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(9.4, 9.8))
    grid = GridSpec(3, 6, figure=figure, hspace=0.70, wspace=0.62, height_ratios=[1.03, 1.03, 1.08])

    axis = figure.add_subplot(grid[0, 0:3])
    add_panel_label(axis, "a")
    plot_etec_axis(axis, etec)

    axis = figure.add_subplot(grid[0, 3:6])
    add_panel_label(axis, "b")
    plot_okeefe_axis(axis, okeefe)

    axis = figure.add_subplot(grid[1, 0:3])
    add_panel_label(axis, "c")
    plot_axis_weights(axis, etec, "ETEC spectral axis projected to taxa")

    axis = figure.add_subplot(grid[1, 3:6])
    add_panel_label(axis, "d")
    plot_axis_weights(axis, okeefe, "O'Keefe spectral axis projected to taxa")

    axis = figure.add_subplot(grid[2, 0:2])
    add_panel_label(axis, "e")
    plot_oriented_balance(axis, etec, okeefe)

    axis = figure.add_subplot(grid[2, 2:4])
    add_panel_label(axis, "f")
    plot_positive_negative_response(
        axis,
        etec["subject_validation"],
        "ETEC pole responses",
        {"acute_minus_baseline": "#D77A61", "recovery_minus_acute": "#6F95A0"},
    )

    axis = figure.add_subplot(grid[2, 4:6])
    add_panel_label(axis, "g")
    plot_positive_negative_response(
        axis,
        okeefe["subject_validation"],
        "O'Keefe pole responses",
        {"AFR_DI_minus_HE": "#C989AB", "AAM_DI_minus_HE": "#5E9F8B"},
    )

    figure.savefig(output_dir / "spectral_response_axis_integrated.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "spectral_response_axis_integrated.png", bbox_inches="tight", dpi=dpi)
    plt.close(figure)


def write_analysis_tables(etec, okeefe, output_dir, save_tables):
    output_dir.mkdir(parents=True, exist_ok=True)

    summarize_pole_validation(etec["subject_validation"]).to_csv(output_dir / "etec_pole_validation_summary.csv", index=False)
    summarize_pole_validation(okeefe["subject_validation"]).to_csv(output_dir / "okeefe_pole_validation_summary.csv", index=False)
    etec["frequency_profile"].assign(dataset="ETEC").to_csv(output_dir / "etec_axis_frequency_profile.csv", index=False)
    okeefe["frequency_profile"].assign(dataset="OKeefe").to_csv(output_dir / "okeefe_axis_frequency_profile.csv", index=False)

    if not save_tables:
        return

    etec["axis_scores"].reset_index().rename(columns={"index": "sample_id"}).to_csv(output_dir / "etec_axis_scores.csv", index=False)
    okeefe["axis_scores"].reset_index().rename(columns={"index": "sample_id"}).to_csv(output_dir / "okeefe_axis_scores.csv", index=False)
    etec["weights"].to_csv(output_dir / "etec_axis_taxon_weights.csv", index=False)
    okeefe["weights"].to_csv(output_dir / "okeefe_axis_taxon_weights.csv", index=False)
    etec["subject_validation"].to_csv(output_dir / "etec_pole_validation_subject_level.csv", index=False)
    okeefe["subject_validation"].to_csv(output_dir / "okeefe_pole_validation_subject_level.csv", index=False)


def main():
    args = parse_arguments()
    configure_plotting(args.dpi)

    etec, okeefe = build_analyses(args)

    write_analysis_tables(etec, okeefe, args.output_dir, args.save_tables)
    create_integrated_figure(etec, okeefe, args.output_dir, args.dpi)

    print(f"ETEC samples retained: {etec['clr_abundance'].shape[0]}")
    print(f"ETEC taxa retained: {etec['clr_abundance'].shape[1]}")
    print(f"O'Keefe samples retained: {okeefe['clr_abundance'].shape[0]}")
    print(f"O'Keefe taxa retained: {okeefe['clr_abundance'].shape[1]}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
