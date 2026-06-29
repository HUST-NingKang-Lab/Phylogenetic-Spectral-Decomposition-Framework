import numpy as np
import pandas as pd


def enforce_simplex(values):
    values = np.maximum(values, 0.0)
    sums = values.sum(axis=-1, keepdims=True)
    return np.where(sums > 0, values / sums, np.full_like(values, 1.0 / values.shape[-1]))


def positive_normal(mean, standard_deviation, random_generator):
    standard_deviation = max(standard_deviation, 1e-8)

    for _ in range(16):
        value = random_generator.normal(loc=mean, scale=standard_deviation)
        if value > 0:
            return value

    return max(mean, 1e-6)


def generate_from_class(
    values,
    n_samples,
    keep_modes=None,
    amplitude_jitter=0.1,
    modify_proportion=0.15,
    random_generator=None,
):
    if random_generator is None:
        random_generator = np.random.default_rng()

    values = np.asarray(values, dtype=float)
    n_reference, n_features = values.shape
    n_frequency = n_features // 2 + 1
    coefficients_all = np.fft.rfft(values, axis=1)
    norms = np.linalg.norm(values, ord=2, axis=1, keepdims=True) + 1e-12
    amplitude = np.abs(coefficients_all) / norms
    amplitude_sd = amplitude.std(axis=0)

    max_mode = n_frequency if keep_modes is None else int(min(max(1, keep_modes), n_frequency))
    candidate_modes = np.arange(1, max_mode, dtype=int)

    if candidate_modes.size == 0:
        candidate_modes = np.array([0], dtype=int)

    n_modified = max(1, int(round(modify_proportion * len(candidate_modes))))
    generated = np.zeros((n_samples, n_features), dtype=np.float64)

    for index in range(n_samples):
        template_index = int(random_generator.integers(0, n_reference))
        coefficients = np.fft.rfft(values[template_index], axis=0)
        selected_modes = random_generator.choice(candidate_modes, size=min(n_modified, len(candidate_modes)), replace=False)

        for mode in selected_modes:
            sigma = (amplitude_sd[mode] if n_reference > 1 else 0.1) * max(amplitude_jitter, 1e-6)
            factor = positive_normal(1.0, sigma, random_generator)
            coefficients[mode] *= factor

        generated[index] = enforce_simplex(np.fft.irfft(coefficients, n=n_features))

    return generated


def generate_by_group(
    abundance,
    labels,
    multipliers=(1,),
    keep_modes=None,
    amplitude_jitter=0.1,
    modify_proportion=0.15,
    random_state=42,
):
    random_generator = np.random.default_rng(random_state)
    labels = pd.Series(labels, index=abundance.index)
    generated_tables = {}

    for multiplier in multipliers:
        records = []

        for group in labels.value_counts().index:
            sample_ids = labels.index[labels == group]
            values = abundance.loc[sample_ids].to_numpy(dtype=float)
            n_samples = int(len(sample_ids) * multiplier)
            generated = generate_from_class(
                values=values,
                n_samples=n_samples,
                keep_modes=keep_modes,
                amplitude_jitter=amplitude_jitter,
                modify_proportion=modify_proportion,
                random_generator=random_generator,
            )
            table = pd.DataFrame(generated, columns=abundance.columns)
            table["group"] = group
            records.append(table)

        generated_tables[multiplier] = pd.concat(records, axis=0, ignore_index=True)

    return generated_tables
