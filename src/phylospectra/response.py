import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .spectral import fourier_coefficients


def normalize_vector(values):
    values = np.asarray(values, dtype=float)
    norm = np.linalg.norm(values)
    if norm <= 1e-12:
        return values * np.nan
    return values / norm


def cosine_similarity(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-12:
        return np.nan
    return float(first.dot(second) / denominator)


def feature_scale(matrix):
    median = matrix.median(axis=0)
    mad = (matrix - median).abs().median(axis=0)
    standard_deviation = matrix.std(axis=0)
    scale = mad.copy()
    invalid = (~np.isfinite(scale)) | (scale < 1e-12)
    scale.loc[invalid] = standard_deviation.loc[invalid]
    scale[(~np.isfinite(scale)) | (scale < 1e-12)] = 1.0
    return scale


def baseline_deltas(features, metadata, subject_column, baseline_mask):
    baseline = metadata.loc[baseline_mask]
    baseline_subjects = set(baseline[subject_column])
    rows = []
    sample_ids = []

    for sample_id in features.index:
        subject = metadata.loc[sample_id, subject_column]

        if subject not in baseline_subjects:
            continue

        baseline_ids = baseline.index[baseline[subject_column] == subject]
        baseline_vector = features.loc[baseline_ids].mean(axis=0).to_numpy()
        rows.append(features.loc[sample_id].to_numpy() - baseline_vector)
        sample_ids.append(sample_id)

    return pd.DataFrame(rows, index=sample_ids, columns=features.columns), metadata.loc[sample_ids].copy()


def mean_by_subject(matrix, metadata, subject_column, mask):
    sample_ids = metadata.index[mask]
    subset = matrix.loc[sample_ids].copy()
    subset["subject"] = metadata.loc[sample_ids, subject_column].to_numpy()
    return subset.groupby("subject").mean()


def response_axis(first_response, second_response=None, orientation="difference"):
    first_direction = normalize_vector(first_response.mean(axis=0).to_numpy())

    if second_response is None:
        axis_values = first_direction
        second_direction = np.full_like(first_direction, np.nan)
    else:
        second_direction = normalize_vector(second_response.mean(axis=0).to_numpy())
        if orientation == "difference":
            axis_values = normalize_vector(first_direction - second_direction)
        elif orientation == "sum":
            axis_values = normalize_vector(first_direction + second_direction)
        else:
            raise ValueError("orientation must be 'difference' or 'sum'.")

    axis = pd.Series(axis_values, index=first_response.columns)
    return axis, pd.Series(first_direction, index=first_response.columns), pd.Series(second_direction, index=first_response.columns)


def project_on_axis(matrix, axis):
    return pd.Series(matrix[axis.index].to_numpy() @ axis.to_numpy(), index=matrix.index)


def spectral_feature_table(clr_table, fmax=0.45, use_hann_window=True):
    coefficients, frequency = fourier_coefficients(clr_table.to_numpy(), use_hann_window=use_hann_window, real_output=False, drop_dc=False)
    retained = pd.DataFrame(
        {
            "frequency_id": [f"f{i:03d}" for i in range(len(frequency))],
            "frequency_index": np.arange(len(frequency)),
            "frequency": frequency,
        }
    )
    retained = retained[(retained["frequency"] > 0) & (retained["frequency"] <= fmax)].copy()
    indices = retained["frequency_index"].to_numpy()
    selected = coefficients[:, indices]
    features = np.concatenate([selected.real, selected.imag], axis=1)
    columns = [f"real_{item}" for item in retained["frequency_id"]] + [f"imag_{item}" for item in retained["frequency_id"]]
    return pd.DataFrame(features, index=clr_table.index, columns=columns), retained, frequency


def inverse_project_axis(axis, scale, retained_frequencies, n_taxa, taxa_order):
    raw_axis = axis / scale.loc[axis.index]
    n_retained = len(retained_frequencies)
    real_values = raw_axis.iloc[:n_retained].to_numpy()
    imaginary_values = raw_axis.iloc[n_retained:2 * n_retained].to_numpy()
    coefficients = np.zeros(retained_frequencies["frequency_index"].max() + 1, dtype=complex)

    for frequency_index, real_value, imaginary_value in zip(retained_frequencies["frequency_index"], real_values, imaginary_values):
        coefficients[int(frequency_index)] = real_value + 1j * imaginary_value

    weights = np.fft.irfft(coefficients, n=n_taxa)
    weights = (weights - np.mean(weights)) / (np.std(weights) + 1e-12)

    return pd.DataFrame(
        {
            "taxon_order": np.arange(1, n_taxa + 1),
            "taxon": taxa_order,
            "axis_weight": weights,
        }
    )


def extract_axis_poles(axis_weights, quantile=0.15):
    upper = axis_weights["axis_weight"].quantile(1 - quantile)
    lower = axis_weights["axis_weight"].quantile(quantile)
    positive = axis_weights.loc[axis_weights["axis_weight"] >= upper, "taxon"].tolist()
    negative = axis_weights.loc[axis_weights["axis_weight"] <= lower, "taxon"].tolist()
    return positive, negative


def pole_balance(clr_table, positive_taxa, negative_taxa):
    positive_taxa = [taxon for taxon in positive_taxa if taxon in clr_table.columns]
    negative_taxa = [taxon for taxon in negative_taxa if taxon in clr_table.columns]

    output = pd.DataFrame(index=clr_table.index)
    output["positive_pole_mean_clr"] = clr_table[positive_taxa].mean(axis=1)
    output["negative_pole_mean_clr"] = clr_table[negative_taxa].mean(axis=1)
    output["pole_balance"] = output["positive_pole_mean_clr"] - output["negative_pole_mean_clr"]
    return output


def residual_axes(matrix, target_axis, n_axes=5, random_state=0):
    target = normalize_vector(target_axis.to_numpy())
    projection = matrix[target_axis.index].to_numpy() @ target
    residual = matrix[target_axis.index].to_numpy() - np.outer(projection, target)
    residual = pd.DataFrame(residual, index=matrix.index, columns=target_axis.index)
    n_components = min(n_axes, residual.shape[0], residual.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(residual.to_numpy())
    axes = []

    for index, component in enumerate(pca.components_, start=1):
        component = component - component.dot(target) * target
        component = normalize_vector(component)
        axes.append(pd.Series(component, index=target_axis.index, name=f"residual_pc{index}"))

    return axes, residual, pca.explained_variance_ratio_
