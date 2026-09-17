import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ── Palette ───────────────────────────────────────────────────────────────────
# Every colour below is a documented slot, checked rather than eyeballed. The
# checks were ported from the reference validator (a Node script, unavailable
# here) as Machado-2009 CVD simulation + CIE76 delta-E on adjacent and all pairs,
# OKLCH lightness band 0.43-0.77 light / 0.48-0.67 dark, chroma floor 0.10, and
# WCAG contrast against the surfaces below. Port fidelity was confirmed against
# the reference palette's published numbers (light adjacent 24.2, dark adjacent
# 10.3, aqua/yellow/magenta below 3:1).
#
#   categorical, light, adjacent pairs : worst protan delta-E 24.2  -> pass
#   categorical, light, all pairs      : worst protan delta-E 11.2  -> floor
#   categorical, dark,  adjacent pairs : worst protan delta-E 10.3  -> floor
#   categorical, dark,  all pairs      : worst protan delta-E  2.5  -> fail at 8 slots
#
# Consequences, applied by the interactive tools:
#   * the floor band is legal only with a second channel, so scatter charts pair
#     colour with a marker symbol and label their group centroids directly;
#   * the dark all-pairs failure is why up to 8 groups never rely on hue alone;
#   * slot order is fixed and never cycled - a ninth group folds into "Other";
#   * three light slots sit below 3:1, so every chart ships a table view.
CATEGORICAL_LIGHT = (
    "#2a78d6",  # 1 blue
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 yellow
    "#008300",  # 4 green
    "#4a3aa7",  # 5 violet
    "#e34948",  # 6 red
    "#e87ba4",  # 7 magenta
    "#eb6834",  # 8 orange
)

CATEGORICAL_DARK = (
    "#3987e5",  # 1 blue
    "#199e70",  # 2 aqua
    "#c98500",  # 3 yellow
    "#008300",  # 4 green
    "#9085e9",  # 5 violet
    "#e66767",  # 6 red
    "#d55181",  # 7 magenta
    "#d95926",  # 8 orange
)

OTHER_LIGHT = "#898781"
OTHER_DARK = "#898781"

# Single hue, light to dark, for magnitude. Documented steps 100-700.
SEQUENTIAL_BLUE = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

# Diverging pair for zero-centred values (CLR signals, residuals): the documented
# blue ramp against a red arm generated on the same OKLCH lightness *and* chroma
# ladder at hue 27 degrees, so the arms are perceptually symmetric and neither
# clipped the sRGB gamut. Two hues that read as opposite plus a neutral grey
# midpoint - never a hue at the midpoint.
DIVERGING_RED_BLUE = (
    "#621b18", "#76221e", "#892c27", "#9e342e", "#b13f38", "#c74941", "#d7584f",
    "#dd7167", "#e4857b", "#ea9a91", "#f0aea6", "#f4c3bc", "#fad6d1",
    "#f0efec",  # neutral midpoint
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

# Ordinal ramp for ordered tiers (richness strata). Monotone lightness, adjacent
# delta-L 0.19 / 0.24, light end 2.06:1 against the light surface.
ORDINAL_BLUE = ("#86b6ef", "#2a78d6", "#0d366b")

# The steps an ordinal ramp may start from: nothing lighter than step 250, which is
# the first step clearing 2:1 against the light surface.
ORDINAL_STEPS = SEQUENTIAL_BLUE[3:]

SURFACE_LIGHT = "#fcfcfb"
SURFACE_DARK = "#1a1a19"

INK_LIGHT = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}
INK_DARK = {"primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"}

FONT_STACK = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def palette_for(mode="light"):
    return CATEGORICAL_DARK if mode == "dark" else CATEGORICAL_LIGHT


def surface_for(mode="light"):
    return SURFACE_DARK if mode == "dark" else SURFACE_LIGHT


def ink_for(mode="light"):
    return INK_DARK if mode == "dark" else INK_LIGHT


def group_color_map(groups, mode="light"):
    """Map every group to a fixed slot, so filtering never repaints the survivors.

    Pass the full set of groups, not the currently visible subset: colour follows
    the entity, never its position in a filtered view. Groups beyond the eight
    documented slots share the neutral "Other" bucket rather than a generated hue.
    """
    palette = palette_for(mode)
    ordered = sorted({str(group) for group in groups})
    other = OTHER_DARK if mode == "dark" else OTHER_LIGHT
    return {
        group: (palette[index] if index < len(palette) else other)
        for index, group in enumerate(ordered)
    }


# Subsets of ORDINAL_STEPS chosen by brute force to maximise the smallest adjacent
# lightness gap, keeping both ends of the ramp. Measured gaps: 2 steps 0.43, 3 steps
# 0.19, 4 steps 0.14, 5 steps 0.10. Six or more cannot clear the 0.06 ordinal floor
# from these steps -- the documented ramp simply is not that finely divided -- so
# callers wanting more than five classes must fold the tail rather than subdivide.
_ORDINAL_SUBSETS = {
    1: (-1,),
    2: (0, 9),
    3: (0, 5, 9),
    4: (0, 3, 6, 9),
    5: (0, 3, 5, 7, 9),
}
ORDINAL_RAMP_LIMIT = 5


def ordinal_ramp(count):
    """``count`` ordered steps of one hue, light to dark, capped at five.

    Used where categories are ordered but discrete -- frequency components, for
    instance, where component 5 must read as "further along" than component 2.
    """
    count = max(1, min(int(count), ORDINAL_RAMP_LIMIT))
    return tuple(ORDINAL_STEPS[index] for index in _ORDINAL_SUBSETS[count])


def group_symbol_map(groups):
    """Marker symbols paired with colour, the second identity channel."""
    symbols = ["circle", "square", "triangle-up", "diamond", "cross", "x", "star", "triangle-down"]
    ordered = sorted({str(group) for group in groups})
    return {group: symbols[index % len(symbols)] for index, group in enumerate(ordered)}


def sequential_colormap():
    return LinearSegmentedColormap.from_list("phylospectra_blue", SEQUENTIAL_BLUE)


def diverging_colormap():
    return LinearSegmentedColormap.from_list("phylospectra_red_blue", DIVERGING_RED_BLUE)


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
