from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from spectral_response_axis import (
    build_analyses,
    configure_plotting,
    parse_arguments as parse_response_axis_arguments,
    normalize_vector,
    project_on_axis,
    style_axis,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate whether ETEC and O'Keefe response trajectories are specific to the learned spectral response axes."
    )
    parser.add_argument("--etec-abundance", type=Path, default=Path("data/etec/etec_abundance_genus.csv"))
    parser.add_argument("--etec-metadata", type=Path, default=Path("data/etec/etec_metadata.csv"))
    parser.add_argument("--okeefe-abundance", type=Path, default=Path("data/okeefe/okeefe_dietswap_abundance.csv"))
    parser.add_argument("--okeefe-metadata", type=Path, default=Path("data/okeefe/okeefe_dietswap_metadata.csv"))
    parser.add_argument("--phylogeny", type=Path, default=Path("data/phylogeny.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/03_spectral_seesaw_response/control_axis_specificity"))
    parser.add_argument("--pseudocount", type=float, default=1e-6)
    parser.add_argument("--fmax", type=float, default=0.45)
    parser.add_argument("--window", choices=["hann", "none"], default="hann")
    parser.add_argument("--min-prevalence", type=float, default=0.02)
    parser.add_argument("--min-total-count", type=float, default=10.0)
    parser.add_argument("--min-taxa", type=int, default=20)
    parser.add_argument("--pole-quantile", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=20260428)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--n-control-axes", type=int, default=5)
    parser.add_argument("--save-tables", action="store_true")
    return parser.parse_args()


def remove_target_axis(matrix, axis):
    axis_values = normalize_vector(axis.values)
    projection = matrix[axis.index].values @ axis_values
    residual = matrix[axis.index].values - np.outer(projection, axis_values)
    return pd.DataFrame(residual, index=matrix.index, columns=axis.index)


def build_control_axes(analysis, n_axes, random_state):
    residual = remove_target_axis(analysis["delta"], analysis["axis"])
    n_components = min(n_axes, residual.shape[0], residual.shape[1])

    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(residual.values)

    axes = []

    for index, component in enumerate(pca.components_, start=1):
        target = normalize_vector(analysis["axis"].values)
        component = component - component.dot(target) * target
        component = normalize_vector(component)
        axes.append(
            pd.Series(
                component,
                index=analysis["axis"].index,
                name=f"residual_pc{index}",
            )
        )

    return axes, residual, pca.explained_variance_ratio_


def score_axis(analysis, axis, label):
    scores = analysis["axis_scores"].copy()
    scores["axis_score"] = project_on_axis(analysis["delta"], axis)
    scores["axis_name"] = label
    return scores


def etec_trajectory_summary(scores):
    phase_medians = scores.groupby("phase")["axis_score"].median()
    baseline = phase_medians.get("Baseline", 0.0)
    acute = phase_medians.get("Acute", np.nan)
    recovery = phase_medians.get("Recovery", np.nan)
    acute_displacement = acute - baseline
    recovery_shift = recovery - acute
    trajectory_score = acute_displacement - recovery_shift

    return {
        "acute_displacement": acute_displacement,
        "recovery_shift": recovery_shift,
        "trajectory_score": trajectory_score,
    }


def okeefe_reciprocity_summary(scores):
    subject_level = scores.groupby(["nationality", "subject", "group"])["axis_score"].mean().reset_index()
    wide = subject_level.pivot_table(index=["nationality", "subject"], columns="group", values="axis_score").reset_index()

    wide = wide.dropna(subset=["HE", "DI"]).copy()
    wide["diet_shift"] = wide["DI"] - wide["HE"]

    afr_shift = wide.loc[wide["nationality"] == "AFR", "diet_shift"].median()
    aam_shift = wide.loc[wide["nationality"] == "AAM", "diet_shift"].median()
    reciprocity_score = afr_shift - aam_shift

    return {
        "afr_diet_shift": afr_shift,
        "aam_diet_shift": aam_shift,
        "reciprocity_score": reciprocity_score,
    }


def summarize_control_axes(etec, okeefe, n_control_axes, random_state):
    etec_axes, _, etec_variance = build_control_axes(etec, n_control_axes, random_state)
    okeefe_axes, _, okeefe_variance = build_control_axes(okeefe, n_control_axes, random_state)

    rows = []

    etec_target_scores = score_axis(etec, etec["axis"], "target_axis")
    etec_summary = etec_trajectory_summary(etec_target_scores)
    rows.append(
        {
            "dataset": "ETEC",
            "axis_name": "target_axis",
            "explained_variance_ratio": np.nan,
            **etec_summary,
            "afr_diet_shift": np.nan,
            "aam_diet_shift": np.nan,
            "reciprocity_score": np.nan,
        }
    )

    for index, axis in enumerate(etec_axes, start=1):
        scores = score_axis(etec, axis, f"residual_pc{index}")
        summary = etec_trajectory_summary(scores)
        rows.append(
            {
                "dataset": "ETEC",
                "axis_name": f"residual_pc{index}",
                "explained_variance_ratio": etec_variance[index - 1],
                **summary,
                "afr_diet_shift": np.nan,
                "aam_diet_shift": np.nan,
                "reciprocity_score": np.nan,
            }
        )

    okeefe_target_scores = score_axis(okeefe, okeefe["axis"], "target_axis")
    okeefe_summary = okeefe_reciprocity_summary(okeefe_target_scores)
    rows.append(
        {
            "dataset": "OKeefe",
            "axis_name": "target_axis",
            "explained_variance_ratio": np.nan,
            "acute_displacement": np.nan,
            "recovery_shift": np.nan,
            "trajectory_score": np.nan,
            **okeefe_summary,
        }
    )

    for index, axis in enumerate(okeefe_axes, start=1):
        scores = score_axis(okeefe, axis, f"residual_pc{index}")
        summary = okeefe_reciprocity_summary(scores)
        rows.append(
            {
                "dataset": "OKeefe",
                "axis_name": f"residual_pc{index}",
                "explained_variance_ratio": okeefe_variance[index - 1],
                "acute_displacement": np.nan,
                "recovery_shift": np.nan,
                "trajectory_score": np.nan,
                **summary,
            }
        )

    return pd.DataFrame(rows)


def plot_specificity_summary(summary, output_dir, dpi):
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"wspace": 0.42})

    etec = summary[summary["dataset"] == "ETEC"].copy()
    okeefe = summary[summary["dataset"] == "OKeefe"].copy()

    etec_colors = ["#D44C3C" if name == "target_axis" else "#B7B5A0" for name in etec["axis_name"]]
    okeefe_colors = ["#B66065" if name == "target_axis" else "#B7B5A0" for name in okeefe["axis_name"]]

    axes[0].bar(np.arange(len(etec)), etec["trajectory_score"], color=etec_colors, width=0.72)
    axes[0].axhline(0, color="#452A3D", linewidth=0.75, linestyle="--", alpha=0.65)
    axes[0].set_xticks(np.arange(len(etec)))
    axes[0].set_xticklabels(etec["axis_name"], rotation=45, ha="right")
    axes[0].set_ylabel("Acute-recovery trajectory score")
    axes[0].set_title("ETEC axis specificity")
    style_axis(axes[0])

    axes[1].bar(np.arange(len(okeefe)), okeefe["reciprocity_score"], color=okeefe_colors, width=0.72)
    axes[1].axhline(0, color="#452A3D", linewidth=0.75, linestyle="--", alpha=0.65)
    axes[1].set_xticks(np.arange(len(okeefe)))
    axes[1].set_xticklabels(okeefe["axis_name"], rotation=45, ha="right")
    axes[1].set_ylabel("Reciprocal diet-response score")
    axes[1].set_title("O'Keefe axis specificity")
    style_axis(axes[1])

    figure.savefig(output_dir / "control_axis_specificity.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "control_axis_specificity.png", bbox_inches="tight", dpi=dpi)
    plt.close(figure)


def main():
    args = parse_arguments()
    configure_plotting(args.dpi)

    etec, okeefe = build_analyses(args)
    summary = summarize_control_axes(
        etec=etec,
        okeefe=okeefe,
        n_control_axes=args.n_control_axes,
        random_state=args.random_state,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "control_axis_specificity_summary.csv", index=False)
    plot_specificity_summary(summary, args.output_dir, args.dpi)

    print(f"Control axes evaluated: {args.n_control_axes}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
