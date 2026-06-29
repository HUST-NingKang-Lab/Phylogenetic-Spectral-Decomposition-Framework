import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def configure_matplotlib(dpi=300, font_family="DejaVu Sans"):
    plt.rcParams.update(
        {
            "font.family": font_family,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": dpi,
        }
    )


def style_axis(axis, grid=True, grid_color="#E8E3DB"):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    if grid:
        axis.grid(color=grid_color, linewidth=0.65, alpha=0.85)
        axis.set_axisbelow(True)


def embed_profiles(table, feature_columns=None, random_state=42):
    if feature_columns is None:
        feature_columns = [column for column in table.columns if column not in {"class", "group", "type"}]

    try:
        import umap
        reducer = umap.UMAP(random_state=random_state)
        coordinates = reducer.fit_transform(table[feature_columns].to_numpy(dtype=float))
        columns = ["UMAP1", "UMAP2"]
        method = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
        coordinates = reducer.fit_transform(table[feature_columns].to_numpy(dtype=float))
        columns = ["Axis1", "Axis2"]
        method = "PCA"

    embedding = pd.DataFrame(coordinates, columns=columns, index=table.index)
    embedding["embedding_method"] = method
    return embedding


def save_figure(figure, output_path, dpi=300):
    output_path = str(output_path)
    figure.savefig(output_path, bbox_inches="tight", dpi=dpi if output_path.lower().endswith(".png") else None)
