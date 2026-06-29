import numpy as np
import pandas as pd


def fourier_coefficients(values, use_hann_window=True, real_output=False, drop_dc=True):
    array = np.asarray(values, dtype=float)

    if array.ndim == 1:
        array = array.reshape(1, -1)

    n = array.shape[1]

    if use_hann_window:
        array = array * np.hanning(n)[None, :]

    coefficients = np.fft.rfft(array, axis=1)
    frequency = np.fft.rfftfreq(n, d=1.0)

    if drop_dc:
        coefficients = coefficients[:, 1:]
        frequency = frequency[1:]

    if real_output:
        coefficients = np.concatenate([coefficients.real, coefficients.imag], axis=1)

    return coefficients, frequency


def spectral_power(values, use_hann_window=True):
    array = np.asarray(values, dtype=float)

    if array.ndim == 1:
        array = array.reshape(1, -1)

    n = array.shape[1]

    if use_hann_window:
        array = array * np.hanning(n)[None, :]

    coefficients = np.fft.rfft(array, axis=1)
    power = (np.abs(coefficients) ** 2) / n
    frequency = np.fft.rfftfreq(n, d=1.0)
    return power, frequency


def spectral_slope(values, fmax=0.20, use_hann_window=True):
    power, frequency = spectral_power(values, use_hann_window=use_hann_window)
    n_taxa = np.asarray(values).shape[-1]
    fmin = 2.0 / n_taxa
    records = []

    for row in power:
        mask = (frequency >= fmin) & (frequency <= fmax) & np.isfinite(row) & (row > 0)

        if mask.sum() < 3:
            records.append({"beta": np.nan, "intercept": np.nan, "r2": np.nan})
            continue

        x = np.log10(frequency[mask])
        y = np.log10(row[mask])
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        residual_sum = np.sum((y - fitted) ** 2)
        total_sum = np.sum((y - y.mean()) ** 2)
        r2 = 1 - residual_sum / total_sum if total_sum > 0 else np.nan
        records.append({"beta": float(-slope), "intercept": float(intercept), "r2": float(r2)})

    return pd.DataFrame(records)


def cumulative_energy(values, fmax=0.20, use_hann_window=True):
    power, frequency = spectral_power(values, use_hann_window=use_hann_window)
    n_taxa = np.asarray(values).shape[-1]
    fmin = 2.0 / n_taxa
    mask = (frequency >= fmin) & (frequency <= fmax)
    active_power = power[:, mask]
    totals = np.nansum(active_power, axis=1, keepdims=True)
    probability = np.divide(active_power, totals, out=np.zeros_like(active_power), where=totals > 0)
    cumulative = np.cumsum(probability, axis=1)
    mode_fraction = np.arange(1, cumulative.shape[1] + 1) / cumulative.shape[1]
    return cumulative, mode_fraction, frequency[mask]


def fraction_reaching_threshold(cumulative, threshold):
    indices = np.where(cumulative >= threshold)[0]
    if len(indices) == 0:
        return 1.0
    return (indices[0] + 1) / len(cumulative)


def compressibility_metrics(values, richness=None, fmax=0.20, use_hann_window=True):
    cumulative, mode_fraction, frequency = cumulative_energy(values, fmax=fmax, use_hann_window=use_hann_window)
    records = []

    for index, curve in enumerate(cumulative):
        probability = np.diff(np.r_[0.0, curve])
        entropy = -float(np.nansum(probability * np.log(probability + 1e-30)))
        n_modes = len(probability)
        effective_mode_number = float(np.exp(entropy))
        record = {
            "C50": fraction_reaching_threshold(curve, 0.50),
            "C80": fraction_reaching_threshold(curve, 0.80),
            "C90": fraction_reaching_threshold(curve, 0.90),
            "E10_low_order_energy": float(curve[max(0, int(np.ceil(0.10 * n_modes)) - 1)]),
            "E20_low_order_energy": float(curve[max(0, int(np.ceil(0.20 * n_modes)) - 1)]),
            "spectral_entropy_norm": entropy / np.log(n_modes),
            "effective_mode_number": effective_mode_number,
            "effective_spectral_dimension": effective_mode_number / n_modes,
            "spectral_centroid_norm": float(np.nansum(frequency * probability) / (frequency.max() + 1e-30)),
            "n_active_modes": n_modes,
        }

        if richness is not None:
            record["richness"] = int(richness[index])

        records.append(record)

    return pd.DataFrame(records)


def macro_micro_scores(values, fmax=0.20, low_fraction=0.25, high_fraction=0.35, pseudocount=1e-30, use_hann_window=True):
    power, frequency = spectral_power(values, use_hann_window=use_hann_window)
    n_taxa = np.asarray(values).shape[-1]
    fmin = 2.0 / n_taxa
    modes = np.where((frequency >= fmin) & (frequency <= fmax))[0]

    if len(modes) < 6:
        raise ValueError("Too few frequency modes were retained.")

    n_low = max(2, int(np.ceil(len(modes) * low_fraction)))
    n_high = max(2, int(np.ceil(len(modes) * high_fraction)))
    low_modes = modes[:n_low]
    high_modes = modes[-n_high:]
    middle_modes = np.array([mode for mode in modes if mode not in set(low_modes) and mode not in set(high_modes)])

    records = []

    for row in power:
        low_power = float(np.nansum(row[low_modes]))
        middle_power = float(np.nansum(row[middle_modes])) if len(middle_modes) else 0.0
        high_power = float(np.nansum(row[high_modes]))
        records.append(
            {
                "low_power": low_power,
                "middle_power": middle_power,
                "high_power": high_power,
                "low_frequency_macro_organization": np.log10((low_power + pseudocount) / (middle_power + high_power + pseudocount)),
                "high_frequency_micro_fragmentation": np.log10((high_power + pseudocount) / (low_power + pseudocount)),
            }
        )

    return pd.DataFrame(records)
