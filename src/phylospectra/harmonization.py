import numpy as np
import pandas as pd

from .io import inverse_centered_log_ratio


def lowpass_signal(signal, cutoff_frequency):
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    frequency = np.fft.rfftfreq(n)
    coefficients = np.fft.rfft(signal)
    filtered = np.zeros_like(coefficients)
    filtered[frequency <= cutoff_frequency] = coefficients[frequency <= cutoff_frequency]
    return np.fft.irfft(filtered, n=n)


def fourier_batch_harmonization(
    clr_table,
    metadata,
    batch_column="batch",
    condition_column=None,
    cutoff_frequency=0.05,
    per_condition=True,
    correction_strength=1.0,
):
    corrected = clr_table.copy()
    batches = metadata[batch_column].unique()

    if condition_column is None or not per_condition:
        conditions = [None]
    else:
        conditions = metadata[condition_column].unique()

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
