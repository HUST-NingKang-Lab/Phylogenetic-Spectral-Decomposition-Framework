from pathlib import Path
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullLocator
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate low-order Fourier mode representations for MGnify biome classification."
    )
    parser.add_argument("--abundance", type=Path, default=Path("data/mgnify/abu.h5"))
    parser.add_argument("--abundance-key", default="genus")
    parser.add_argument("--metadata", type=Path, default=Path("data/mgnify/metadata.csv"))
    parser.add_argument("--phylogeny", type=Path, default=Path("data/phylogeny.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/04_spectral_compressibility/low_order_biome_classification"),
    )
    parser.add_argument("--biome-col", default="level_3")
    parser.add_argument("--top-n-biomes", type=int, default=8)
    parser.add_argument("--min-samples-per-biome", type=int, default=1000)
    parser.add_argument("--samples-per-biome", type=int, default=1000)
    parser.add_argument("--pseudocount", type=float, default=1e-9)
    parser.add_argument("--low-order-fractions", type=float, nargs="+", default=[0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--focal-low-order-fraction", type=float, default=0.005)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--use-hann-window", action="store_true", default=True)
    parser.add_argument("--no-hann-window", action="store_false", dest="use_hann_window")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def configure_matplotlib(dpi):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.3,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.1,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": dpi,
            "figure.dpi": 160,
        }
    )


def style_axis(axis, grid=True, grid_axis="y"):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#3A2634")
    axis.spines["bottom"].set_color("#3A2634")
    axis.tick_params(colors="#3A2634", length=3, width=0.75)

    if grid:
        if grid_axis == "both":
            axis.grid(color="#E8E3DB", linewidth=0.65, alpha=0.85)
        else:
            axis.grid(axis=grid_axis, color="#E8E3DB", linewidth=0.65, alpha=0.85)
        axis.set_axisbelow(True)


def add_panel_label(axis, label):
    axis.text(
        -0.115,
        1.075,
        label,
        transform=axis.transAxes,
        fontsize=10.0,
        fontweight="bold",
        va="top",
        ha="left",
        color="#3A2634",
    )


def normalize_taxon_name(value):
    value = str(value).strip()
    if value.startswith("sk__"):
        value = "k__" + value[4:]
    return value


def short_taxon_name(value):
    last = str(value).split(";")[-1]
    if "__" in last:
        last = last.split("__")[-1]
    return last if last else str(value)


def format_fraction_label(fraction):
    percent = fraction * 100
    if percent < 1:
        return f"{percent:.2g}%"
    if percent < 10:
        return f"{percent:.1f}%"
    return f"{percent:.0f}%"


def format_runtime(seconds):
    if not np.isfinite(seconds):
        return "NA"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min"


def build_phylogenetic_name_map(phylogeny):
    phylogeny = phylogeny.iloc[:, 0].astype(str).str.split(";", expand=True)

    if phylogeny.shape[1] < 6:
        raise ValueError("The phylogeny table must contain semicolon-delimited taxonomic paths to genus level.")

    phylogeny.index = phylogeny[5]
    phylogeny = phylogeny[~phylogeny.index.duplicated(keep="first")]
    fullnames = phylogeny[0] + ";" + phylogeny[1] + ";" + phylogeny[2] + ";" + phylogeny[3] + ";" + phylogeny[4] + ";" + phylogeny[5]
    return fullnames


def load_balanced_mgnify_data(args):
    abundance = pd.read_hdf(args.abundance, args.abundance_key)
    metadata = pd.read_csv(args.metadata, low_memory=False)
    phylogeny = pd.read_csv(args.phylogeny, low_memory=False)

    metadata = metadata.rename(columns={"SampleID": "sample", "Env": "biome"})
    if "sample" not in metadata.columns:
        raise ValueError("Metadata must contain either 'SampleID' or 'sample'.")
    if "biome" not in metadata.columns:
        raise ValueError("Metadata must contain either 'Env' or 'biome'.")

    metadata["sample"] = metadata["sample"].astype(str)
    metadata = metadata.set_index("sample")

    environment_parts = metadata["biome"].astype(str).str.split(":", expand=True)
    for index in range(environment_parts.shape[1]):
        metadata[f"level_{index + 1}"] = environment_parts[index]

    abundance.index = abundance.index.astype(str)
    abundance.columns = [normalize_taxon_name(column) for column in abundance.columns]

    fullnames = build_phylogenetic_name_map(phylogeny)
    overlap = abundance.columns.intersection(fullnames.index)

    if len(overlap) == 0:
        raise ValueError("No overlapping taxa were found between the abundance table and phylogeny.")

    abundance = abundance.loc[:, overlap]
    abundance.columns = fullnames.loc[abundance.columns].values

    shared_samples = abundance.index.intersection(metadata.index)
    if len(shared_samples) == 0:
        raise ValueError("No overlapping samples were found between abundance and metadata.")

    abundance = abundance.loc[shared_samples]
    metadata = metadata.loc[shared_samples]

    if args.biome_col not in metadata.columns:
        raise ValueError(f"Biome column '{args.biome_col}' was not found in metadata.")

    metadata = metadata[metadata[args.biome_col].notna()]
    abundance = abundance.loc[metadata.index]

    counts = metadata[args.biome_col].value_counts()
    biome_order = counts[counts >= args.min_samples_per_biome].index[:args.top_n_biomes].tolist()

    if len(biome_order) < args.top_n_biomes:
        raise ValueError(
            f"Only {len(biome_order)} biomes have at least {args.min_samples_per_biome} samples."
        )

    metadata = metadata[metadata[args.biome_col].isin(biome_order)].copy()
    abundance = abundance.loc[metadata.index].copy()
    metadata[args.biome_col] = pd.Categorical(metadata[args.biome_col], categories=biome_order, ordered=True)

    random_generator = np.random.default_rng(args.random_state)
    sampled_ids = []

    for biome in biome_order:
        ids = metadata.index[metadata[args.biome_col].astype(str) == str(biome)].to_numpy()
        if len(ids) < args.samples_per_biome:
            raise ValueError(f"Biome '{biome}' has only {len(ids)} samples, fewer than {args.samples_per_biome}.")
        ids = random_generator.choice(ids, size=args.samples_per_biome, replace=False)
        sampled_ids.extend(ids.tolist())

    sampled_ids = np.array(sampled_ids, dtype=object)
    random_generator.shuffle(sampled_ids)

    metadata = metadata.loc[sampled_ids].copy()
    abundance = abundance.loc[metadata.index].copy()

    ordered_columns = [column for column in fullnames.values if column in abundance.columns]
    abundance = abundance.loc[:, ordered_columns]
    abundance = abundance.loc[:, abundance.sum(axis=0) > 0]

    taxa_order = pd.DataFrame(
        {
            "taxon_order": np.arange(1, len(abundance.columns) + 1),
            "taxon_fullname": abundance.columns,
            "taxon_short": [short_taxon_name(column) for column in abundance.columns],
        }
    )

    return abundance, metadata, taxa_order, biome_order


def centered_log_ratio(abundance, pseudocount):
    row_sums = abundance.sum(axis=1)
    abundance = abundance.loc[row_sums > 0]
    relative = abundance.div(abundance.sum(axis=1), axis=0).astype(np.float32)
    log_relative = np.log(relative + np.float32(pseudocount)).astype(np.float32)
    return log_relative.sub(log_relative.mean(axis=1), axis=0).astype(np.float32)


def alpha_diversity_features(abundance):
    values = abundance.to_numpy(dtype=float)
    row_sum = values.sum(axis=1)
    relative = np.divide(values, row_sum[:, None], out=np.zeros_like(values), where=row_sum[:, None] > 0)
    log_relative = np.where(relative > 0, np.log(relative), 0.0)

    shannon = -np.sum(relative * log_relative, axis=1)
    richness = np.sum(values > 0, axis=1).astype(float)
    evenness = np.divide(shannon, np.log(richness), out=np.zeros_like(shannon), where=richness > 1)

    rounded = np.rint(values).astype(int)
    f1 = np.sum(rounded == 1, axis=1).astype(float)
    f2 = np.sum(rounded == 2, axis=1).astype(float)
    chao1 = richness + np.where(f2 > 0, (f1 * f1) / (2.0 * f2), (f1 * (f1 - 1.0)) / 2.0)

    return pd.DataFrame(
        {
            "sample": abundance.index.astype(str),
            "Shannon": shannon,
            "Richness": richness,
            "Chao1": chao1,
            "Evenness": evenness,
        }
    ).set_index("sample")


def compute_fourier_coefficients(clr_abundance, use_hann_window):
    values = clr_abundance.to_numpy(dtype=np.float32, copy=False)
    n_taxa = values.shape[1]

    if use_hann_window:
        values = values * np.hanning(n_taxa).astype(np.float32)[None, :]

    coefficients = np.fft.rfft(values, axis=1).astype(np.complex64)
    frequency = np.fft.rfftfreq(n_taxa, d=1.0).astype(np.float32)
    mode_indices = np.arange(1, len(frequency))
    return coefficients[:, mode_indices], frequency[mode_indices], mode_indices


def coefficients_to_features(coefficients, selected_indices=None):
    selected = coefficients if selected_indices is None else coefficients[:, selected_indices]
    return np.concatenate([selected.real, selected.imag], axis=1).astype(np.float32)


def build_classifier():
    return make_pipeline(StandardScaler(), RidgeClassifier())


def cross_validated_predictions(features, labels, n_splits, random_state):
    classes = np.unique(labels)
    predictions = np.empty_like(labels)
    scores = np.empty((len(labels), len(classes)), dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    start_time = time.perf_counter()

    for train_index, test_index in splitter.split(features, labels):
        classifier = build_classifier()
        classifier.fit(features[train_index], labels[train_index])
        predictions[test_index] = classifier.predict(features[test_index])
        decision = classifier.decision_function(features[test_index])
        if decision.ndim == 1:
            decision = np.vstack([-decision, decision]).T
        scores[test_index] = decision

    elapsed = time.perf_counter() - start_time

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "score": scores,
        "cv_time_sec": float(elapsed),
    }


def roc_tables(labels, scores, class_names, representation):
    classes = np.arange(len(class_names))
    binarized = label_binarize(labels, classes=classes)
    curve_records = []
    auc_records = []
    mean_fpr = np.linspace(0.0, 1.0, 201)
    interpolated_tprs = []

    for class_index, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(binarized[:, class_index], scores[:, class_index])
        class_auc = auc(fpr, tpr)

        auc_records.append(
            {
                "representation": representation,
                "class": class_name,
                "auc": float(class_auc),
            }
        )

        curve_records.extend(
            [
                {
                    "representation": representation,
                    "class": class_name,
                    "curve": "class",
                    "fpr": float(x),
                    "tpr": float(y),
                    "auc": float(class_auc),
                }
                for x, y in zip(fpr, tpr)
            ]
        )

        interpolated = np.interp(mean_fpr, fpr, tpr)
        interpolated[0] = 0.0
        interpolated_tprs.append(interpolated)

    macro_tpr = np.mean(np.vstack(interpolated_tprs), axis=0)
    macro_tpr[-1] = 1.0
    macro_auc = auc(mean_fpr, macro_tpr)

    auc_records.append({"representation": representation, "class": "macro", "auc": float(macro_auc)})
    curve_records.extend(
        [
            {
                "representation": representation,
                "class": "macro",
                "curve": "macro",
                "fpr": float(x),
                "tpr": float(y),
                "auc": float(macro_auc),
            }
            for x, y in zip(mean_fpr, macro_tpr)
        ]
    )

    return pd.DataFrame(curve_records), pd.DataFrame(auc_records)


def run_classification_analysis(args):
    raw_abundance, metadata, taxa_order, biome_order = load_balanced_mgnify_data(args)
    clr_abundance = centered_log_ratio(raw_abundance, args.pseudocount)
    raw_abundance = raw_abundance.loc[clr_abundance.index]
    metadata = metadata.loc[clr_abundance.index]
    alpha = alpha_diversity_features(raw_abundance).loc[clr_abundance.index]

    encoder = LabelEncoder()
    labels = encoder.fit_transform(metadata[args.biome_col].astype(str).to_numpy())
    class_names = list(encoder.classes_)

    coefficients, frequency, mode_indices = compute_fourier_coefficients(clr_abundance, args.use_hann_window)
    n_total_modes = coefficients.shape[1]
    n_taxa = raw_abundance.shape[1]

    predictions = {}
    benchmark_records = []

    alpha_methods = {
        "Shannon": ["Shannon"],
        "Richness": ["Richness"],
        "Chao1": ["Chao1"],
        "Evenness": ["Evenness"],
        "Alpha diversity combined": ["Shannon", "Richness", "Chao1", "Evenness"],
    }

    for method_name, columns in alpha_methods.items():
        features = alpha[columns].to_numpy(dtype=np.float32)
        result = cross_validated_predictions(features, labels, args.n_splits, args.random_state)
        predictions[method_name] = result
        _, auc_table = roc_tables(labels, result["score"], class_names, method_name)
        macro_auc = auc_table.loc[auc_table["class"] == "macro", "auc"].iloc[0]
        benchmark_records.append(
            {
                "method": method_name,
                "representation_type": "alpha",
                "mode_fraction": np.nan,
                "n_modes": 0,
                "n_features": features.shape[1],
                "accuracy": result["accuracy"],
                "macro_auc": macro_auc,
                "cv_time_sec": result["cv_time_sec"],
                "feature_reduction_vs_taxa": n_taxa / features.shape[1],
            }
        )

    full_method = "Full taxon abundance"
    full_features = raw_abundance.to_numpy(dtype=np.float32, copy=False)
    result = cross_validated_predictions(full_features, labels, args.n_splits, args.random_state)
    predictions[full_method] = result
    _, auc_table = roc_tables(labels, result["score"], class_names, full_method)
    macro_auc = auc_table.loc[auc_table["class"] == "macro", "auc"].iloc[0]
    benchmark_records.append(
        {
            "method": full_method,
            "representation_type": "full_taxon",
            "mode_fraction": np.nan,
            "n_modes": np.nan,
            "n_features": full_features.shape[1],
            "accuracy": result["accuracy"],
            "macro_auc": macro_auc,
            "cv_time_sec": result["cv_time_sec"],
            "feature_reduction_vs_taxa": 1.0,
        }
    )

    for fraction in args.low_order_fractions:
        n_modes = int(np.ceil(fraction * n_total_modes))
        n_modes = max(1, min(n_modes, n_total_modes))
        selected_modes = np.arange(n_modes)
        features = coefficients_to_features(coefficients, selected_modes)
        method_name = f"Low-order {format_fraction_label(fraction)} modes"

        result = cross_validated_predictions(features, labels, args.n_splits, args.random_state)
        predictions[method_name] = result
        _, auc_table = roc_tables(labels, result["score"], class_names, method_name)
        macro_auc = auc_table.loc[auc_table["class"] == "macro", "auc"].iloc[0]

        benchmark_records.append(
            {
                "method": method_name,
                "representation_type": "low_order_spectral",
                "mode_fraction": fraction,
                "n_modes": n_modes,
                "n_features": features.shape[1],
                "accuracy": result["accuracy"],
                "macro_auc": macro_auc,
                "cv_time_sec": result["cv_time_sec"],
                "feature_reduction_vs_taxa": n_taxa / features.shape[1],
            }
        )

    benchmark = pd.DataFrame(benchmark_records)
    focal_method = f"Low-order {format_fraction_label(args.focal_low_order_fraction)} modes"

    if focal_method not in predictions:
        raise ValueError(
            f"The focal method '{focal_method}' was not evaluated. Add {args.focal_low_order_fraction} to --low-order-fractions."
        )

    roc_methods = ["Shannon", "Richness", "Chao1", "Evenness", "Alpha diversity combined", focal_method, full_method]
    roc_curve_tables = []
    roc_auc_tables = []

    for representation in roc_methods:
        curves, aucs = roc_tables(labels, predictions[representation]["score"], class_names, representation)
        roc_curve_tables.append(curves)
        roc_auc_tables.append(aucs)

    roc_curves = pd.concat(roc_curve_tables, ignore_index=True)
    roc_aucs = pd.concat(roc_auc_tables, ignore_index=True)
    per_biome_curves, per_biome_aucs = roc_tables(labels, predictions[focal_method]["score"], class_names, focal_method)

    return {
        "class_names": class_names,
        "biome_order": biome_order,
        "taxa_order": taxa_order,
        "n_taxa": n_taxa,
        "n_modes_total": n_total_modes,
        "benchmark": benchmark,
        "roc_curves": roc_curves,
        "roc_aucs": roc_aucs,
        "per_biome_curves": per_biome_curves,
        "per_biome_aucs": per_biome_aucs,
        "focal_method": focal_method,
        "full_method": full_method,
    }


def plot_performance_vs_fraction(axis, result):
    colors = {
        "Alpha diversity combined": "#6B5570",
        "Low-order spectral": "#57928B",
        "Full taxon abundance": "#A8BCCC",
        "Focal": "#D75A49",
    }

    benchmark = result["benchmark"]
    low_order = benchmark[benchmark["representation_type"] == "low_order_spectral"].sort_values("mode_fraction")

    x_values = low_order["mode_fraction"].to_numpy() * 100
    y_values = low_order["macro_auc"].to_numpy()

    axis.plot(x_values, y_values, marker="o", markersize=4.0, linewidth=1.8, color=colors["Low-order spectral"], label="Low-order spectral")

    alpha = benchmark[benchmark["method"] == "Alpha diversity combined"].iloc[0]
    full = benchmark[benchmark["method"] == result["full_method"]].iloc[0]
    focal = benchmark[benchmark["method"] == result["focal_method"]].iloc[0]

    axis.axhline(alpha["macro_auc"], color=colors["Alpha diversity combined"], linewidth=1.15, linestyle="--", alpha=0.92, label=f"Alpha combined AUC={alpha['macro_auc']:.3f}")
    axis.axhline(full["macro_auc"], color=colors["Full taxon abundance"], linewidth=1.15, linestyle=":", alpha=0.98, label=f"Full abundance AUC={full['macro_auc']:.3f}")
    axis.axvline(focal["mode_fraction"] * 100, color=colors["Focal"], linewidth=0.9, alpha=0.48)
    axis.scatter(focal["mode_fraction"] * 100, focal["macro_auc"], s=54, marker="o", color=colors["Focal"], edgecolor="white", linewidth=0.8, zorder=5)
    axis.text(
        focal["mode_fraction"] * 100 * 1.07,
        focal["macro_auc"] + 0.008,
        format_fraction_label(focal["mode_fraction"]),
        fontsize=6.6,
        color=colors["Focal"],
        ha="left",
        va="bottom",
    )

    lower = max(0.0, min(low_order["macro_auc"].min(), alpha["macro_auc"], full["macro_auc"]) - 0.035)
    upper = min(1.01, max(low_order["macro_auc"].max(), alpha["macro_auc"], full["macro_auc"]) + 0.035)

    axis.set_ylim(lower, upper)
    axis.set_xscale("log")
    axis.set_xticks([0.1, 0.5, 1, 5, 10, 30])
    axis.set_xticklabels(["0.1", "0.5", "1", "5", "10", "30"])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_xlabel("Lowest-frequency Fourier modes retained (%)")
    axis.set_ylabel("Macro-average ROC AUC")
    axis.set_title("Low-order bandwidth recovers ROC performance")
    axis.legend(frameon=False, loc="lower right", handlelength=1.8)
    style_axis(axis, grid=True, grid_axis="both")


def plot_macro_roc(axis, result):
    colors = {
        "Shannon": "#5D8790",
        "Richness": "#D75A49",
        "Chao1": "#6FAE8D",
        "Evenness": "#D8A24C",
        "Alpha diversity combined": "#6B5570",
        result["focal_method"]: "#57928B",
        result["full_method"]: "#A8BCCC",
    }

    display_names = {
        "Shannon": "Shannon",
        "Richness": "Richness",
        "Chao1": "Chao1",
        "Evenness": "Evenness",
        "Alpha diversity combined": "Alpha diversity combined",
        result["focal_method"]: f"{result['focal_method']} ({int(result['benchmark'].loc[result['benchmark']['method'] == result['focal_method'], 'n_features'].iloc[0])} features)",
        result["full_method"]: "Full taxon abundance",
    }

    curves = result["roc_curves"]
    aucs = result["roc_aucs"]

    for representation, display_name in display_names.items():
        subset = curves[(curves["representation"] == representation) & (curves["class"] == "macro")]
        macro_auc = aucs[(aucs["representation"] == representation) & (aucs["class"] == "macro")]["auc"].iloc[0]
        linewidth = 1.15 if representation in {"Shannon", "Richness", "Chao1", "Evenness"} else 1.55
        alpha = 0.82 if representation in {"Shannon", "Richness", "Chao1", "Evenness"} else 0.98
        axis.plot(
            subset["fpr"],
            subset["tpr"],
            linewidth=linewidth,
            alpha=alpha,
            color=colors[representation],
            label=f"{display_name} (AUC={macro_auc:.3f})",
        )

    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, color="#AFA99B", alpha=0.9)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Macro-average one-vs-rest ROC")
    axis.legend(frameon=False, loc="lower right", handlelength=1.65, borderaxespad=0.25)
    style_axis(axis, grid=False)


def plot_per_biome_roc(axis, result):
    colors = ["#5D8790", "#D75A49", "#6FAE8D", "#D8A24C", "#6B5570", "#B98B55", "#57928B", "#A8BCCC"]
    curves = result["per_biome_curves"]
    aucs = result["per_biome_aucs"]

    for index, biome in enumerate(result["class_names"]):
        subset = curves[(curves["representation"] == result["focal_method"]) & (curves["class"] == biome)]
        class_auc = aucs[(aucs["representation"] == result["focal_method"]) & (aucs["class"] == biome)]["auc"].iloc[0]
        axis.plot(subset["fpr"], subset["tpr"], linewidth=1.25, color=colors[index % len(colors)], label=f"{biome} ({class_auc:.2f})")

    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, color="#AFA99B")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(f"Per-biome ROC for {format_fraction_label(0.005)} low-order modes")
    axis.legend(frameon=False, loc="lower right", ncol=1, handlelength=1.3, borderaxespad=0.2)
    style_axis(axis, grid=False)


def plot_feature_performance_tradeoff(axis, result):
    colors = {
        "Alpha diversity combined": "#6B5570",
        result["focal_method"]: "#57928B",
        result["full_method"]: "#A8BCCC",
    }

    benchmark = result["benchmark"]
    focal = benchmark[benchmark["method"] == result["focal_method"]].iloc[0]

    methods = [
        ("Alpha diversity combined", "Alpha diversity combined"),
        (result["focal_method"], f"Low-order {format_fraction_label(focal['mode_fraction'])}"),
        (result["full_method"], "Full taxon abundance"),
    ]

    x_values = []
    y_values = []
    point_colors = []
    labels = []

    for method, display in methods:
        row = benchmark[benchmark["method"] == method].iloc[0]
        x_values.append(row["n_features"])
        y_values.append(row["macro_auc"])
        point_colors.append(colors[method])
        labels.append(
            (
                display,
                f"{int(row['n_features'])} features\n"
                f"{row['feature_reduction_vs_taxa']:.0f}× reduction\n"
                f"{format_runtime(row['cv_time_sec'])}",
            )
        )

    axis.scatter(x_values, y_values, s=[58, 72, 58], c=point_colors, edgecolor="white", linewidth=0.85, zorder=4)
    axis.plot(x_values, y_values, linewidth=0.75, color="#AFA99B", alpha=0.55, zorder=2)

    for x_value, y_value, (label, note) in zip(x_values, y_values, labels):
        offset = 0.88 if label == "Full taxon abundance" else 1.08
        alignment = "right" if label == "Full taxon abundance" else "left"
        axis.text(x_value * offset, y_value, f"{label}\n{note}", fontsize=6.25, ha=alignment, va="center", color="#3A2634")

    axis.set_xscale("log")
    axis.set_xlabel("Number of input features (log scale)")
    axis.set_ylabel("Macro-average ROC AUC")
    axis.set_title("Spectral compression balances accuracy and cost")
    style_axis(axis, grid=True, grid_axis="both")


def create_figure(result, output_dir, dpi):
    figure = plt.figure(figsize=(8.00, 7.25), constrained_layout=False)
    grid = GridSpec(2, 2, figure=figure, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0], wspace=0.34, hspace=0.50)

    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, 0])
    axis_d = figure.add_subplot(grid[1, 1])

    add_panel_label(axis_a, "a")
    plot_performance_vs_fraction(axis_a, result)

    add_panel_label(axis_b, "b")
    plot_macro_roc(axis_b, result)

    add_panel_label(axis_c, "c")
    plot_per_biome_roc(axis_c, result)

    add_panel_label(axis_d, "d")
    plot_feature_performance_tradeoff(axis_d, result)

    figure.suptitle("Low-order spectral compression preserves biome identity", y=0.985, fontsize=11.2, color="#3A2634")
    figure.subplots_adjust(left=0.065, right=0.975, bottom=0.095, top=0.895)
    figure.savefig(output_dir / "low_order_biome_classification.png", bbox_inches="tight", dpi=dpi)
    figure.savefig(output_dir / "low_order_biome_classification.pdf", bbox_inches="tight")
    plt.close(figure)


def write_outputs(result, output_dir, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    result["benchmark"].to_csv(output_dir / "low_order_classification_benchmark.csv", index=False)
    result["roc_curves"].to_csv(output_dir / "macro_roc_curves.csv", index=False)
    result["roc_aucs"].to_csv(output_dir / "macro_roc_aucs.csv", index=False)
    result["per_biome_curves"].to_csv(output_dir / "per_biome_roc_curves.csv", index=False)
    result["per_biome_aucs"].to_csv(output_dir / "per_biome_roc_aucs.csv", index=False)
    result["taxa_order"].to_csv(output_dir / "taxa_phylogenetic_order.csv", index=False)

    parameter_table = pd.DataFrame(
        {
            "parameter": [
                "abundance_key",
                "biome_col",
                "top_n_biomes",
                "min_samples_per_biome",
                "samples_per_biome",
                "pseudocount",
                "low_order_fractions",
                "focal_low_order_fraction",
                "n_splits",
                "random_state",
                "use_hann_window",
            ],
            "value": [
                args.abundance_key,
                args.biome_col,
                args.top_n_biomes,
                args.min_samples_per_biome,
                args.samples_per_biome,
                args.pseudocount,
                ",".join(map(str, args.low_order_fractions)),
                args.focal_low_order_fraction,
                args.n_splits,
                args.random_state,
                args.use_hann_window,
            ],
        }
    )
    parameter_table.to_csv(output_dir / "analysis_parameters.csv", index=False)


def main():
    args = parse_arguments()
    configure_matplotlib(args.dpi)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = run_classification_analysis(args)
    write_outputs(result, args.output_dir, args)
    create_figure(result, args.output_dir, args.dpi)

    print(f"Samples retained: {args.top_n_biomes * args.samples_per_biome}")
    print(f"Taxa retained: {result['n_taxa']}")
    print(f"Total Fourier modes: {result['n_modes_total']}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
