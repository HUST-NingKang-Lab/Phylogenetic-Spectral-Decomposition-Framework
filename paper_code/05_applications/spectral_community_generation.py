from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate synthetic microbiome profiles by perturbing Fourier-domain components within biome classes."
    )
    parser.add_argument("--abundance", type=Path, default=Path("data/generation/mgnify_biome_subset/abundance.h5"))
    parser.add_argument("--abundance-key", default="genus")
    parser.add_argument("--metadata", type=Path, default=Path("data/generation/mgnify_biome_subset/metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/06_generation/spectral_community_generation"))
    parser.add_argument("--sample-id-col", default=None)
    parser.add_argument("--class-col", default="class")
    parser.add_argument("--max-real-samples-per-class", type=int, default=1000)
    parser.add_argument("--generation-multipliers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--target-tail-classes", type=int, default=4)
    parser.add_argument("--keep-modes", type=int, default=None)
    parser.add_argument("--amplitude-jitter", type=float, default=0.1)
    parser.add_argument("--modify-proportion", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
        }
    )


def load_generation_data(args):
    metadata = pd.read_csv(args.metadata, index_col=0 if args.sample_id_col is None else None)
    abundance = pd.read_hdf(args.abundance, args.abundance_key)

    if args.sample_id_col is not None:
        if args.sample_id_col not in metadata.columns:
            raise ValueError(f"Sample column '{args.sample_id_col}' was not found in metadata.")
        metadata[args.sample_id_col] = metadata[args.sample_id_col].astype(str)
        metadata = metadata.set_index(args.sample_id_col)

    if args.class_col not in metadata.columns:
        raise ValueError(f"Class column '{args.class_col}' was not found in metadata.")

    metadata.index = metadata.index.astype(str)
    abundance.index = abundance.index.astype(str)

    shared_samples = abundance.index.intersection(metadata.index)
    if len(shared_samples) == 0:
        raise ValueError("No overlapping samples were found between abundance and metadata.")

    abundance = abundance.loc[shared_samples].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    metadata = metadata.loc[shared_samples].copy()
    abundance = abundance.loc[:, abundance.sum(axis=0) > 0]

    return abundance, metadata


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
    keep_modes,
    amplitude_jitter,
    modify_proportion,
    random_generator,
):
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


def sample_real_profiles(abundance, metadata, class_column, max_samples_per_class, random_state):
    random_generator = np.random.default_rng(random_state)
    records = []

    for class_name in metadata[class_column].value_counts().index:
        class_sample_ids = metadata.index[metadata[class_column] == class_name].to_numpy()

        if len(class_sample_ids) > max_samples_per_class:
            class_sample_ids = random_generator.choice(class_sample_ids, size=max_samples_per_class, replace=False)

        table = abundance.loc[class_sample_ids].copy()
        table["class"] = class_name
        table["type"] = "real"
        records.append(table)

    real_profiles = pd.concat(records, axis=0)
    real_profiles = real_profiles.loc[:, (real_profiles.drop(columns=["class", "type"]) != 0).any(axis=0).tolist() + [True, True]]
    return real_profiles


def generate_profiles_by_multiplier(real_profiles, multipliers, keep_modes, amplitude_jitter, modify_proportion, random_state):
    random_generator = np.random.default_rng(random_state)
    class_counts = real_profiles["class"].value_counts()
    feature_columns = [column for column in real_profiles.columns if column not in {"class", "type"}]
    generated_tables = {}

    for multiplier in multipliers:
        generated_records = []

        for class_name in class_counts.index:
            class_values = real_profiles.loc[real_profiles["class"] == class_name, feature_columns].to_numpy(dtype=float)
            n_samples = int(class_counts[class_name] * multiplier)
            generated = generate_from_class(
                values=class_values,
                n_samples=n_samples,
                keep_modes=keep_modes,
                amplitude_jitter=amplitude_jitter,
                modify_proportion=modify_proportion,
                random_generator=random_generator,
            )
            table = pd.DataFrame(generated, columns=feature_columns)
            table["class"] = class_name
            table["type"] = "generated"
            generated_records.append(table)

        generated_tables[multiplier] = pd.concat(generated_records, axis=0, ignore_index=True)

    return generated_tables


def embed_profiles(table, random_state):
    feature_columns = [column for column in table.columns if column not in {"class", "type"}]

    try:
        import umap
        reducer = umap.UMAP(random_state=random_state)
        coordinates = reducer.fit_transform(table[feature_columns].to_numpy(dtype=float))
        method = "UMAP"
        columns = ["UMAP1", "UMAP2"]
    except Exception:
        reducer = PCA(n_components=2, random_state=random_state)
        coordinates = reducer.fit_transform(table[feature_columns].to_numpy(dtype=float))
        method = "PCA"
        columns = ["Axis1", "Axis2"]

    embedding = pd.DataFrame(coordinates, columns=columns)
    embedding["class"] = table["class"].to_numpy()
    embedding["type"] = table["type"].to_numpy()
    embedding["embedding_method"] = method
    return embedding


def classification_transfer_accuracy(real_profiles, generated_profiles):
    feature_columns = [column for column in real_profiles.columns if column not in {"class", "type"}]
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    classifier.fit(real_profiles[feature_columns].to_numpy(dtype=float), real_profiles["class"].to_numpy())
    predicted = classifier.predict(generated_profiles[feature_columns].to_numpy(dtype=float))
    return accuracy_score(generated_profiles["class"].to_numpy(), predicted)


def plot_embedding_grid(embeddings, output_dir, dpi):
    multipliers = sorted(embeddings.keys())
    figure, axes = plt.subplots(2, len(multipliers), figsize=(4.0 * len(multipliers), 7.2), sharex=False, sharey=False)

    if len(multipliers) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for column_index, multiplier in enumerate(multipliers):
        embedding = embeddings[multiplier]
        coordinate_columns = [column for column in embedding.columns if column in {"UMAP1", "UMAP2", "Axis1", "Axis2"}]
        x_column, y_column = coordinate_columns[:2]

        for type_name, marker, color in [("real", "o", "#44757A"), ("generated", "x", "#D44C3C")]:
            subset = embedding[embedding["type"] == type_name]
            axes[0, column_index].scatter(subset[x_column], subset[y_column], s=9, alpha=0.55, marker=marker, color=color, label=type_name)

        axes[0, column_index].set_title(f"{multiplier}x generated")
        axes[0, column_index].set_xlabel(x_column)
        axes[0, column_index].set_ylabel(y_column)

        for class_name in embedding["class"].unique():
            subset = embedding[embedding["class"] == class_name]
            axes[1, column_index].scatter(subset[x_column], subset[y_column], s=9, alpha=0.55, label=str(class_name))

        axes[1, column_index].set_title(f"{multiplier}x generated")
        axes[1, column_index].set_xlabel(x_column)
        axes[1, column_index].set_ylabel(y_column)

    axes[0, 0].legend(frameon=False)
    axes[1, 0].legend(frameon=False, fontsize=6)
    figure.tight_layout()
    figure.savefig(output_dir / "synthetic_profiles_embedding_grid.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / "synthetic_profiles_embedding_grid.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_target_class_embeddings(real_profiles, generated_profiles, target_classes, output_dir, dpi, random_state):
    for class_name in target_classes:
        subset_generated = generated_profiles[generated_profiles["class"] == class_name]
        combined = pd.concat([real_profiles, subset_generated], axis=0, ignore_index=True)
        embedding = embed_profiles(combined, random_state)
        coordinate_columns = [column for column in embedding.columns if column in {"UMAP1", "UMAP2", "Axis1", "Axis2"}]
        x_column, y_column = coordinate_columns[:2]

        colors = np.where(
            embedding["class"] == class_name,
            np.where(embedding["type"] == "generated", "#D44C3C", "#44757A"),
            "#D9D9D9",
        )

        figure, axis = plt.subplots(figsize=(6, 5))
        axis.scatter(embedding[x_column], embedding[y_column], s=12, alpha=0.65, c=colors, edgecolors="none")
        axis.set_title(f"Synthetic profiles for {class_name}")
        axis.set_xlabel(x_column)
        axis.set_ylabel(y_column)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        safe_name = str(class_name).replace("/", "_").replace(" ", "_")
        figure.savefig(output_dir / f"synthetic_profiles_{safe_name}.png", dpi=dpi, bbox_inches="tight")
        figure.savefig(output_dir / f"synthetic_profiles_{safe_name}.pdf", bbox_inches="tight")
        plt.close(figure)


def write_outputs(real_profiles, generated_tables, embeddings, accuracies, output_dir, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    real_profiles.to_csv(output_dir / "real_profiles_subset.csv", index=True)

    summary_records = []

    for multiplier, generated in generated_tables.items():
        generated.to_csv(output_dir / f"synthetic_profiles_{multiplier}x.csv", index=False)
        embeddings[multiplier].to_csv(output_dir / f"synthetic_profiles_embedding_{multiplier}x.csv", index=False)
        summary_records.append(
            {
                "generation_multiplier": multiplier,
                "n_generated": len(generated),
                "classification_transfer_accuracy": accuracies[multiplier],
            }
        )

    pd.DataFrame(summary_records).to_csv(output_dir / "generation_summary.csv", index=False)

    parameter_table = pd.DataFrame(
        {
            "parameter": [
                "abundance_key",
                "class_col",
                "max_real_samples_per_class",
                "generation_multipliers",
                "target_tail_classes",
                "keep_modes",
                "amplitude_jitter",
                "modify_proportion",
                "random_state",
            ],
            "value": [
                args.abundance_key,
                args.class_col,
                args.max_real_samples_per_class,
                ",".join(map(str, args.generation_multipliers)),
                args.target_tail_classes,
                args.keep_modes,
                args.amplitude_jitter,
                args.modify_proportion,
                args.random_state,
            ],
        }
    )
    parameter_table.to_csv(output_dir / "analysis_parameters.csv", index=False)


def main():
    args = parse_arguments()
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    abundance, metadata = load_generation_data(args)
    real_profiles = sample_real_profiles(
        abundance,
        metadata,
        class_column=args.class_col,
        max_samples_per_class=args.max_real_samples_per_class,
        random_state=args.random_state,
    )

    generated_tables = generate_profiles_by_multiplier(
        real_profiles,
        multipliers=args.generation_multipliers,
        keep_modes=args.keep_modes,
        amplitude_jitter=args.amplitude_jitter,
        modify_proportion=args.modify_proportion,
        random_state=args.random_state,
    )

    embeddings = {}
    accuracies = {}
    combined_tables = {}

    for multiplier, generated in generated_tables.items():
        combined = pd.concat([real_profiles, generated], axis=0, ignore_index=True)
        combined_tables[multiplier] = combined
        embeddings[multiplier] = embed_profiles(combined, args.random_state)
        accuracies[multiplier] = classification_transfer_accuracy(real_profiles, generated)

    plot_embedding_grid(embeddings, args.output_dir, args.dpi)

    largest_multiplier = max(generated_tables)
    class_counts = real_profiles["class"].value_counts()
    target_classes = class_counts.index[-args.target_tail_classes:].tolist()
    plot_target_class_embeddings(
        real_profiles,
        generated_tables[largest_multiplier],
        target_classes,
        args.output_dir,
        args.dpi,
        args.random_state,
    )

    write_outputs(real_profiles, generated_tables, embeddings, accuracies, args.output_dir, args)

    print(f"Real profiles retained: {len(real_profiles)}")
    print(f"Generated multipliers: {args.generation_multipliers}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
