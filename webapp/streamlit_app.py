"""Interactive phylogeny-ordered spectral explorer.

Run it locally from the repository root::

    streamlit run webapp/streamlit_app.py

Users can upload their own abundance table -- and optionally their own metadata
and phylogeny -- or start from one of the bundled datasets. Everything is
computed in memory; nothing is written to disk, so the app also runs unchanged on
Streamlit Community Cloud.

The app is organised as pages sharing one sidebar. The Transform page is the
interactive version of panel a of the manuscript overview figure: it shows the
phylogeny-ordered signal on the left, its Fourier decomposition on the right, and
lets the reader keep fewer and fewer frequency components to watch the signal
rebuild from the low ones alone.
"""

from pathlib import Path
import io
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phylospectra.pipeline import (  # noqa: E402
    DEFAULT_PHYLOGENY,
    EXAMPLE_DATASETS,
    available_example_datasets,
    run_spectral_pipeline,
    unmatched_taxa,
)
from phylospectra.io import inverse_centered_log_ratio  # noqa: E402
from phylospectra.visualization import (  # noqa: E402
    DIVERGING_RED_BLUE,
    FONT_STACK,
    ORDINAL_RAMP_LIMIT,
    SEQUENTIAL_BLUE,
    group_color_map,
    group_symbol_map,
    ink_for,
    ordinal_ramp,
    surface_for,
)

REPOSITORY_URL = "https://github.com/HUST-NingKang-Lab/Phylogenetic-Spectral-Decomposition-Framework"
CLR_LABEL = "CLR (what the transform sees)"

st.set_page_config(
    page_title="PhyloSpectra · interactive explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Everything the pages read, filled once per run after the sidebar is built.
STATE = {}


# ── theme ─────────────────────────────────────────────────────────────────────


def active_mode():
    """Charts follow the viewer's Streamlit theme, with validated steps for each."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def style_figure(figure, height=430, legend=True, mode="light"):
    ink = ink_for(mode)
    surface = surface_for(mode)

    figure.update_layout(
        paper_bgcolor=surface,
        plot_bgcolor=surface,
        font=dict(family=FONT_STACK, size=13, color=ink["secondary"]),
        # The empty text is required: a title object carrying only a font renders in
        # the browser as the literal string "undefined". Charts that want a title set
        # it afterwards with title_text, which keeps this font.
        title=dict(text="", font=dict(family=FONT_STACK, size=16, color=ink["primary"])),
        margin=dict(l=8, r=8, t=56, b=8),
        height=height,
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(family=FONT_STACK, color=ink["secondary"]),
            itemsizing="constant",
        ),
        hoverlabel=dict(font=dict(family=FONT_STACK), bgcolor=surface, bordercolor=ink["axis"]),
    )
    figure.update_xaxes(
        showgrid=True, gridcolor=ink["grid"], gridwidth=1, zeroline=False,
        linecolor=ink["axis"], linewidth=1, ticks="outside", tickcolor=ink["axis"],
        tickfont=dict(color=ink["muted"]), title_font=dict(color=ink["secondary"]),
    )
    figure.update_yaxes(
        showgrid=True, gridcolor=ink["grid"], gridwidth=1, zeroline=False,
        linecolor=ink["axis"], linewidth=1, ticks="outside", tickcolor=ink["axis"],
        tickfont=dict(color=ink["muted"]), title_font=dict(color=ink["secondary"]),
    )
    return figure


def colorscale(stops):
    positions = np.linspace(0.0, 1.0, len(stops))
    return [[float(position), color] for position, color in zip(positions, stops)]


def with_alpha(color, alpha):
    """Translucent fill keyed to a palette slot, leaving its outline opaque."""
    red, green, blue = (int(color.lstrip("#")[index:index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def taxon_tick_axis(figure, taxa_order, count=22, angle=45):
    positions = np.arange(len(taxa_order))
    step = max(1, len(taxa_order) // count)
    figure.update_xaxes(
        title_text="Position along the phylogeny →",
        tickmode="array",
        tickvals=positions[::step],
        ticktext=taxa_order["taxon_short"].tolist()[::step],
        tickangle=angle,
        showgrid=False,
    )
    return figure


def show_table(frame, label, filename, key):
    """Every chart has a table view, so no value is reachable only by hovering."""
    with st.expander(label):
        st.dataframe(frame, width="stretch", height=280, hide_index=True)
        st.download_button(
            "Download CSV",
            frame.to_csv(index=False).encode(),
            file_name=filename,
            mime="text/csv",
            key=f"download_{key}",
        )


# ── loading ───────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Decomposing phylogeny-ordered spectra…", max_entries=8)
def compute_bundle(
    abundance_payload,
    metadata_payload,
    phylogeny_payload,
    sample_column,
    taxa_as_rows,
    group_column,
    pseudocount,
    fmax,
    use_hann_window,
    low_fraction,
    high_fraction,
    max_samples,
    min_samples_per_group,
    cache_token,
):
    abundance = io.BytesIO(abundance_payload) if isinstance(abundance_payload, bytes) else abundance_payload
    metadata = io.BytesIO(metadata_payload) if isinstance(metadata_payload, bytes) else metadata_payload
    phylogeny = io.BytesIO(phylogeny_payload) if isinstance(phylogeny_payload, bytes) else phylogeny_payload

    return run_spectral_pipeline(
        abundance,
        metadata=metadata,
        group_column=group_column,
        phylogeny=phylogeny or DEFAULT_PHYLOGENY,
        root=REPOSITORY_ROOT,
        sample_column=sample_column or None,
        taxa_as_rows=taxa_as_rows,
        max_samples=max_samples,
        min_samples_per_group=min_samples_per_group,
        pseudocount=pseudocount,
        fmax=fmax,
        use_hann_window=use_hann_window,
        low_fraction=low_fraction,
        high_fraction=high_fraction,
    )


@st.cache_data(show_spinner=False, max_entries=6)
def compute_example(name, max_samples=600):
    """A bundled dataset for the case pages, independent of the sidebar selection."""
    spec = EXAMPLE_DATASETS[name]
    return run_spectral_pipeline(
        str(REPOSITORY_ROOT / spec["abundance"]),
        metadata=str(REPOSITORY_ROOT / spec["metadata"]) if spec.get("metadata") else None,
        group_column=spec["group_column"],
        phylogeny=str(REPOSITORY_ROOT / DEFAULT_PHYLOGENY),
        root=REPOSITORY_ROOT,
        sample_column=spec["sample_column"],
        max_samples=max_samples,
    )


# ── figures: phylogeny-ordered signal and its transform ───────────────────────


def component_waves(signal, count):
    """The individual lowest-frequency components of a signal.

    Each wave is the inverse transform of a single Fourier coefficient, so by
    linearity their sum is exactly the low-pass reconstruction.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    coefficients = np.fft.rfft(signal)
    waves = []

    for index in range(min(count, len(coefficients))):
        single = np.zeros_like(coefficients)
        single[index] = coefficients[index]
        waves.append(np.fft.irfft(single, n=n))

    return waves


def reconstruct_signal(signal, keep):
    """Low-pass the signal by keeping the ``keep`` lowest non-DC coefficients."""
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    coefficients = np.fft.rfft(signal)
    retained = np.zeros_like(coefficients)
    retained[: keep + 1] = coefficients[: keep + 1]
    return np.fft.irfft(retained, n=n), retained


def energy_fraction(coefficients, kept_coefficients):
    total = float(np.sum(np.abs(coefficients) ** 2))
    if total <= 0:
        return 1.0
    return float(np.sum(np.abs(kept_coefficients) ** 2) / total)


def figure_signal_panel(result, sample_id, view, clr_reconstruction, mode):
    """The community as a signal along the phylogeny.

    Black is the signal the transform actually sees — the centered log-ratio of the
    abundance, one value per taxon in phylogenetic order. The blue dashed line is
    what survives once only the low-frequency components are kept, so the gap between
    them is exactly the fine-scale detail that was discarded. With every component
    kept the two coincide.
    """
    ink = ink_for(mode)
    figure = go.Figure()
    positions = np.arange(result.n_taxa)

    if view == CLR_LABEL:
        figure.add_trace(
            go.Scatter(
                x=positions, y=result.clr.loc[sample_id].to_numpy(), mode="lines",
                line=dict(color=ink["primary"], width=2.5),
                name="CLR signal",
                hovertemplate="taxon #%{x}<br>CLR %{y:.2f}<extra>signal</extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=positions, y=clr_reconstruction, mode="lines",
                line=dict(color=SEQUENTIAL_BLUE[8], width=2.5, dash="dash"),
                name="rebuilt from the kept components",
                hovertemplate="taxon #%{x}<br>CLR %{y:.2f}<extra>reconstruction</extra>",
            )
        )
        figure.update_yaxes(title_text="Centered log-ratio")
    else:
        reconstructed = inverse_centered_log_ratio(pd.DataFrame([clr_reconstruction])).to_numpy()[0]
        figure.add_trace(
            go.Scatter(
                x=positions, y=result.abundance.loc[sample_id].to_numpy(), mode="lines",
                line=dict(color=ink["primary"], width=2),
                fill="tozeroy", fillcolor=with_alpha(ink["primary"], 0.10),
                name="relative abundance",
                hovertemplate="taxon #%{x}<br>abundance %{y:.4f}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=positions, y=reconstructed, mode="lines",
                line=dict(color=SEQUENTIAL_BLUE[8], width=2.5, dash="dash"),
                name="rebuilt from the kept components",
                hovertemplate="taxon #%{x}<br>abundance %{y:.4f}<extra>reconstruction</extra>",
            )
        )
        figure.update_yaxes(title_text="Relative abundance")

    # Fewer ticks than the wide charts get: this panel shares the row with the 3D
    # scene, so its rotated labels would otherwise crowd into each other.
    taxon_tick_axis(figure, result.taxa_order, count=14)
    # Matches the 3D scene's height exactly: Streamlit sizes a row by its tallest
    # column, so any difference shows up as a dead band under the shorter one.
    return style_figure(figure, height=560, legend=True, mode=mode)


CAMERA_VIEWS = {
    "Signal behind, components forward": dict(eye=dict(x=1.35, y=1.6, z=0.95)),
    "Signal in front": dict(eye=dict(x=1.05, y=-1.9, z=1.0)),
    "Looking down the stack": dict(eye=dict(x=0.2, y=0.6, z=2.6)),
}


def figure_decomposition_3d(result, sample_id, keep, mode, view="Signal behind, components forward", max_components=ORDINAL_RAMP_LIMIT):
    """Panel-a style decomposition in 3D.

    CLR is the vertical axis, so every curve is read the same way as the signal in
    the neighbouring panel; the frequency components sit at increasing depth, the
    broadest wave nearest the signal and the finest furthest away.
    """
    ink = ink_for(mode)
    surface = surface_for(mode)
    signal = result.clr.loc[sample_id].to_numpy()
    positions = np.arange(len(signal))
    reconstruction, _ = reconstruct_signal(signal, keep)
    drawn = min(keep, max_components)

    figure = go.Figure()

    figure.add_trace(
        go.Scatter3d(
            x=positions, y=np.zeros_like(signal, dtype=float), z=signal,
            mode="lines", line=dict(color=ink["primary"], width=5),
            name="original signal",
            hovertemplate="taxon #%{x}<br>CLR %{z:.2f}<extra>original</extra>",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=positions, y=np.zeros_like(signal, dtype=float), z=reconstruction,
            mode="lines", line=dict(color=SEQUENTIAL_BLUE[6], width=4, dash="dash"),
            name=f"sum of the {keep} kept components",
            hovertemplate="taxon #%{x}<br>CLR %{z:.2f}<extra>reconstruction</extra>",
        )
    )

    if drawn > 0:
        waves = component_waves(signal, drawn + 1)[1:]  # skip the DC offset
        ramp = ordinal_ramp(len(waves))
        for index, wave in enumerate(waves):
            figure.add_trace(
                go.Scatter3d(
                    x=positions, y=np.full_like(wave, float(index + 1)), z=wave,
                    mode="lines", line=dict(color=ramp[index], width=4),
                    # The depth axis already labels each component, so the legend
                    # carries one entry for the whole family instead of five rows
                    # that squeeze the scene out of the card.
                    showlegend=index == 0,
                    name=f"components 1–{len(waves)} (broad → fine)",
                    hovertemplate=(
                        f"component {index + 1} (f={(index + 1) / len(signal):.3f})"
                        "<br>taxon #%{x}<br>contribution %{z:.2f}<extra></extra>"
                    ),
                )
            )

    axis_common = dict(
        showbackground=True, backgroundcolor=surface,
        gridcolor=ink["grid"], gridwidth=1,
        zeroline=False, showspikes=False,
        color=ink["muted"], tickfont=dict(color=ink["muted"], size=10),
    )
    figure.update_layout(
        scene=dict(
            # Short titles: a 3D axis title is placed by projecting the axis, and a
            # long one runs off the card at some camera angles.
            xaxis=dict(title=dict(text="Phylogeny", font=dict(color=ink["secondary"], size=11)), **axis_common),
            yaxis=dict(
                title=dict(text="Component", font=dict(color=ink["secondary"], size=11)),
                tickmode="array",
                tickvals=[0] + list(range(1, drawn + 1)),
                ticktext=["signal"] + [f"c{index}" for index in range(1, drawn + 1)],
                **axis_common,
            ),
            zaxis=dict(title=dict(text="CLR", font=dict(color=ink["secondary"], size=11)), **axis_common),
            camera=CAMERA_VIEWS.get(view, CAMERA_VIEWS["Signal behind, components forward"]),
            aspectmode="manual",
            aspectratio=dict(x=1.7, y=1.05, z=0.9),
        ),
        margin=dict(l=10, r=10, t=80, b=10),
    )
    return style_figure(figure, height=560, mode=mode)


def figure_signal_decomposition(result, sample_id, keep, show_components, mode):
    ink = ink_for(mode)
    surface = surface_for(mode)
    signal = result.clr.loc[sample_id].to_numpy()
    positions = np.arange(len(signal))
    reconstruction, _ = reconstruct_signal(signal, keep)

    figure = go.Figure()
    surface_color = surface

    if show_components:
        # Skip the DC term: CLR rows are centered, so it is exactly zero and would
        # draw a flat line at the bottom of every stack. Component numbering then
        # matches the 3D view.
        waves = component_waves(signal, min(keep, 5) + 1)[1:]
        ramp = ordinal_ramp(len(waves))
        for index, wave in enumerate(waves):
            frequency = (index + 1) / len(signal)
            figure.add_trace(
                go.Scatter(
                    x=positions, y=wave, mode="lines",
                    line=dict(color=ramp[index], width=1.5),
                    name=f"component {index + 1} · f={frequency:.3f}",
                    hovertemplate=(
                        f"component {index + 1} (f={frequency:.3f})"
                        "<br>taxon #%{x}<br>contribution %{y:.2f}<extra></extra>"
                    ),
                )
            )

    figure.add_trace(
        go.Scatter(
            x=positions, y=signal, mode="lines",
            line=dict(color=ink["primary"], width=2.5),
            name="original signal",
            hovertemplate="taxon #%{x}<br>CLR %{y:.2f}<extra>original</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=positions, y=reconstruction, mode="lines",
            line=dict(color=SEQUENTIAL_BLUE[6], width=2.5, dash="dash"),
            name=f"sum of the {keep} kept components",
            hovertemplate="taxon #%{x}<br>CLR %{y:.2f}<extra>reconstruction</extra>",
        )
    )

    figure.update_yaxes(title_text="Centered log-ratio")
    taxon_tick_axis(figure, result.taxa_order)
    return style_figure(figure, height=440, mode=mode)


def figure_amplitude_spectrum(result, sample_id, keep, mode, show_windowed=False):
    ink = ink_for(mode)
    signal = result.clr.loc[sample_id].to_numpy()
    n = len(signal)
    coefficients = np.fft.rfft(signal)
    frequency = np.fft.rfftfreq(n, d=1.0)
    amplitude = np.abs(coefficients)

    keep_mask = np.zeros_like(amplitude, dtype=bool)
    keep_mask[: keep + 1] = True
    dropped_mask = ~keep_mask
    dropped_mask[0] = False

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=frequency[dropped_mask], y=amplitude[dropped_mask],
            marker=dict(color=ink["muted"], line=dict(width=0)),
            name="dropped", hovertemplate="f=%{x:.4f}<br>|X|=%{y:.2f}<extra>dropped</extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=frequency[keep_mask], y=amplitude[keep_mask],
            marker=dict(color=SEQUENTIAL_BLUE[6], line=dict(width=0)),
            name=f"kept (lowest {keep})",
            hovertemplate="f=%{x:.4f}<br>|X|=%{y:.2f}<extra>kept</extra>",
        )
    )

    if show_windowed:
        # Dividing by the window's mean (its coherent gain) puts the tapered spectrum
        # back on the same amplitude scale as the plain one, so the two are comparable.
        window = np.hanning(n)
        windowed = np.abs(np.fft.rfft(signal * window)) / float(window.mean())
        figure.add_trace(
            go.Scatter(
                x=frequency, y=windowed, mode="lines",
                line=dict(color=SEQUENTIAL_BLUE[10], width=2, dash="dot"),
                name="Hann-windowed (as in the analysis)",
                hovertemplate="f=%{x:.4f}<br>|X|=%{y:.2f}<extra>windowed</extra>",
            )
        )

    fmin, fmax = result.fmin, result.params["fmax"]
    figure.add_vrect(
        x0=fmin, x1=fmax, fillcolor=ink["grid"], opacity=0.45, line_width=0, layer="below",
        annotation_text="slope fitted here", annotation_position="bottom left",
        annotation_font=dict(color=ink["muted"], size=11, family=FONT_STACK),
    )

    figure.update_xaxes(title_text="Phylogenetic frequency")
    figure.update_yaxes(title_text="Amplitude |X| of the Fourier coefficient")
    figure.update_layout(barmode="overlay", bargap=0.15)
    return style_figure(figure, height=440, mode=mode)


def figure_power_spectrum(result, mode, show_samples=True, show_band=True):
    ink = ink_for(mode)
    colors = group_color_map(result.groups(), mode=mode)
    figure = go.Figure()
    frequency = result.analysis_frequency
    groups = result.groups()

    if show_samples and result.group is not None:
        for group in groups:
            sample_ids = result.group.index[result.group.astype(str) == group]
            block = result.power.loc[sample_ids].iloc[:, result.frequency_mask]
            for _, row in block.iterrows():
                figure.add_trace(
                    go.Scatter(
                        x=frequency, y=row.to_numpy(), mode="lines",
                        line=dict(color=colors[group], width=1),
                        opacity=0.22, name=str(group), legendgroup=str(group),
                        showlegend=False,
                        hovertemplate="%{y:.3g}<extra>" + str(group) + "</extra>",
                    )
                )
    elif show_samples:
        for _, row in result.power.iloc[: min(len(result.power), 200)].iterrows():
            figure.add_trace(
                go.Scatter(
                    x=frequency, y=row.to_numpy(), mode="lines",
                    line=dict(color=ink["muted"], width=1), opacity=0.22,
                    showlegend=False, hoverinfo="skip",
                )
            )

    summaries = result.group_summaries()
    for group in groups:
        block = summaries[summaries["group"] == group]
        if len(block) == 0:
            continue

        if show_band:
            figure.add_trace(
                go.Scatter(
                    x=pd.concat([block["frequency"], block["frequency"][::-1]]),
                    y=pd.concat([block["q3_power"], block["q1_power"][::-1]]),
                    fill="toself", fillcolor=colors[group], opacity=0.16,
                    line=dict(width=0), hoverinfo="skip",
                    legendgroup=str(group), showlegend=False,
                )
            )

        figure.add_trace(
            go.Scatter(
                x=block["frequency"], y=block["median_power"], mode="lines",
                line=dict(color=colors[group], width=2),
                name=str(group), legendgroup=str(group), showlegend=True,
                hovertemplate="%{y:.3g}<extra>" + str(group) + "</extra>",
            )
        )

        beta = float(block["group_beta"].iloc[0])
        intercept = float(block["group_intercept"].iloc[0])
        figure.add_trace(
            go.Scatter(
                x=block["frequency"], y=10 ** (intercept - beta * np.log10(block["frequency"])),
                mode="lines", line=dict(color=colors[group], width=1.5, dash="dash"),
                legendgroup=str(group), showlegend=False, hoverinfo="skip",
            )
        )
        figure.add_annotation(
            x=float(block["frequency"].iloc[-1]), y=float(block["median_power"].iloc[-1]),
            text=f"β={beta:.2f}", showarrow=False, xanchor="left", xshift=6,
            font=dict(color=ink["primary"], size=12, family=FONT_STACK),
        )

    ticks = [value for value in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
             if frequency.min() * 0.9 <= value <= frequency.max() * 1.1]
    figure.update_xaxes(
        type="log", title_text="Phylogenetic frequency",
        tickmode="array", tickvals=ticks, ticktext=[f"{value:g}" for value in ticks],
    )
    figure.update_yaxes(type="log", title_text="Power spectral density")
    figure.update_layout(hovermode="x unified")
    return style_figure(figure, mode=mode)


def figure_slope_distribution(result, mode, compact=False):
    ink = ink_for(mode)
    colors = group_color_map(result.groups(), mode=mode)
    symbols = group_symbol_map(result.groups())
    surface = surface_for(mode)
    groups = result.groups()
    figure = go.Figure()

    for group in groups:
        sample_ids = result.group.index[result.group.astype(str) == group]
        values = result.slopes.loc[sample_ids.intersection(result.slopes.index), "beta"].dropna()
        if len(values) == 0:
            continue

        figure.add_trace(
            go.Box(
                y=values.to_numpy(), name=str(group), boxpoints=False, width=0.5,
                marker=dict(color=colors[group]),
                line=dict(color=colors[group], width=2),
                fillcolor=with_alpha(colors[group], 0.22),
                hoverinfo="y",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=np.random.default_rng(0).normal(0, 0.055, len(values)) + groups.index(group),
                y=values.to_numpy(), mode="markers",
                marker=dict(color=colors[group], size=8, symbol=symbols[group],
                            line=dict(width=2, color=surface)),
                name=str(group), legendgroup=str(group), showlegend=False,
                hovertemplate="β=%{y:.3f}<extra>" + str(group) + "</extra>",
            )
        )

    figure.update_yaxes(title_text="Spectral slope (β)")
    figure.update_xaxes(showgrid=False, title_text="")
    figure.update_layout(showlegend=False, hovermode="closest")
    return style_figure(figure, height=320 if compact else 430, legend=False, mode=mode)


def figure_macro_micro(result, mode):
    ink = ink_for(mode)
    colors = group_color_map(result.groups(), mode=mode)
    symbols = group_symbol_map(result.groups())
    surface = surface_for(mode)
    figure = go.Figure()

    for group in result.groups():
        sample_ids = result.group.index[result.group.astype(str) == group]
        block = result.macro_micro.loc[sample_ids.intersection(result.macro_micro.index)]
        if len(block) == 0:
            continue

        figure.add_trace(
            go.Scatter(
                x=block["low_frequency_macro_organization"],
                y=block["high_frequency_micro_fragmentation"],
                mode="markers", name=str(group),
                marker=dict(color=colors[group], size=9, symbol=symbols[group],
                            line=dict(width=2, color=surface)),
                hovertemplate="macro %{x:.3f}<br>micro %{y:.3f}<extra>" + str(group) + "</extra>",
            )
        )
        figure.add_annotation(
            x=float(block["low_frequency_macro_organization"].median()),
            y=float(block["high_frequency_micro_fragmentation"].median()),
            text=f"<b>{group}</b>", showarrow=False, yshift=16,
            bgcolor=surface, borderpad=2, opacity=0.9,
            font=dict(color=ink["primary"], size=12, family=FONT_STACK),
        )

    figure.update_xaxes(title_text="Low-frequency macro-organization (log₁₀)")
    figure.update_yaxes(title_text="High-frequency micro-fragmentation (log₁₀)")
    return style_figure(figure, mode=mode)


def figure_cumulative_energy(result, mode):
    ink = ink_for(mode)
    colors = group_color_map(result.groups(), mode=mode)
    figure = go.Figure()
    mode_fraction = result.mode_fraction

    for group in result.groups():
        sample_ids = result.group.index[result.group.astype(str) == group]
        block = result.cumulative.loc[sample_ids.intersection(result.cumulative.index)]
        if len(block) == 0:
            continue

        median = block.median(axis=0)
        lower = block.quantile(0.25, axis=0)
        upper = block.quantile(0.75, axis=0)

        figure.add_trace(
            go.Scatter(
                x=np.concatenate([mode_fraction, mode_fraction[::-1]]),
                y=np.concatenate([upper.to_numpy(), lower.to_numpy()[::-1]]),
                fill="toself", fillcolor=with_alpha(colors[group], 0.16),
                line=dict(width=0), hoverinfo="skip",
                legendgroup=str(group), showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=mode_fraction, y=median.to_numpy(), mode="lines",
                line=dict(color=colors[group], width=2),
                name=str(group), legendgroup=str(group),
                hovertemplate="%{y:.3f}<extra>" + str(group) + "</extra>",
            )
        )

    figure.add_hline(y=0.80, line=dict(color=ink["axis"], width=1),
                     annotation_text="80% of spectral energy", annotation_position="bottom right",
                     annotation_font=dict(color=ink["muted"], size=11))
    figure.update_xaxes(title_text="Fraction of frequency modes, low to high", range=[0, 1])
    figure.update_yaxes(title_text="Cumulative spectral energy", range=[0, 1.02])
    figure.update_layout(hovermode="x unified")
    return style_figure(figure, mode=mode)


def figure_c80_richness(result, mode):
    colors = group_color_map(result.groups(), mode=mode)
    symbols = group_symbol_map(result.groups())
    surface = surface_for(mode)
    figure = go.Figure()
    table = result.compressibility.join(result.diversity["Richness"])

    for group in result.groups():
        sample_ids = result.group.index[result.group.astype(str) == group]
        block = table.loc[sample_ids.intersection(table.index)]
        if len(block) == 0:
            continue

        figure.add_trace(
            go.Scatter(
                x=block["Richness"], y=block["C80"], mode="markers", name=str(group),
                marker=dict(color=colors[group], size=9, symbol=symbols[group], opacity=0.55,
                            line=dict(width=2, color=surface)),
                customdata=block.index.to_numpy(),
                hovertemplate="%{customdata}<br>richness %{x}<br>C80 %{y:.3f}<extra>" + str(group) + "</extra>",
            )
        )

    figure.update_xaxes(title_text="Taxonomic richness (phylo-ordered taxa observed)")
    figure.update_yaxes(title_text="C80 · modes needed for 80% of energy", range=[0.3, 1.02])
    return style_figure(figure, mode=mode)


def figure_signal_heatmap(result, mode, max_rows, sort_by_group=True):
    ink = ink_for(mode)
    surface = surface_for(mode)
    table = result.clr

    if result.group is not None and sort_by_group:
        order = result.group.reindex(table.index).fillna("").astype(str)
        table = table.loc[order.sort_values(kind="stable").index]

    if len(table) > max_rows:
        table = table.iloc[:: max(1, len(table) // max_rows)]

    positions = np.arange(result.n_taxa)
    values = table.to_numpy()

    # CLR is centered on zero but not symmetric: absent taxa pile up just below the
    # sample mean while dominant taxa reach far above it. Each arm gets its own
    # percentile, with the zero point pinned, so the long upper tail cannot wash the
    # whole matrix into one pale band.
    lower_clip = min(float(np.nanpercentile(values, 2)), -1e-6)
    upper_clip = max(float(np.nanpercentile(values, 98)), 1e-6)

    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=positions,
            y=[str(label) for label in table.index],
            colorscale=colorscale(DIVERGING_RED_BLUE),
            zmin=lower_clip,
            zmid=0.0,
            zmax=upper_clip,
            colorbar=dict(
                title=dict(text=f"CLR (clipped {lower_clip:.1f} … {upper_clip:.1f})",
                           font=dict(color=ink["secondary"], family=FONT_STACK)),
                tickfont=dict(color=ink["muted"], family=FONT_STACK),
                outlinewidth=0, thickness=12, len=0.72,
            ),
            hovertemplate="taxon #%{x}<br>%{y}<br>CLR %{z:.2f}<extra></extra>",
            hoverongaps=False,
        )
    )
    taxon_tick_axis(figure, result.taxa_order, count=24)
    figure.update_yaxes(title_text="Sample", showticklabels=False, showgrid=False,
                        ticks="", linecolor=surface)
    return style_figure(figure, height=560, legend=False, mode=mode)


def figure_sample_profile(result, mode, samples):
    ink = ink_for(mode)
    colors = group_color_map(result.groups(), mode=mode)
    figure = go.Figure()
    positions = np.arange(result.n_taxa)

    for sample in samples:
        if sample not in result.clr.index:
            continue
        group = str(result.group.loc[sample]) if result.group is not None and sample in result.group.index else None
        figure.add_trace(
            go.Scatter(
                x=positions, y=result.clr.loc[sample].to_numpy(), mode="lines",
                line=dict(color=colors[group] if group else SEQUENTIAL_BLUE[7], width=2),
                name=str(sample),
                hovertemplate="taxon #%{x}<br>CLR %{y:.2f}<extra>" + str(sample) + "</extra>",
            )
        )

    figure.update_yaxes(title_text="Centered log-ratio", zeroline=True,
                        zerolinecolor=ink["axis"], zerolinewidth=1)
    taxon_tick_axis(figure, result.taxa_order, count=20)
    return style_figure(figure, mode=mode)


# ── sidebar ───────────────────────────────────────────────────────────────────


def render_sidebar():
    st.sidebar.subheader("Data")

    source_kind = st.sidebar.radio(
        "Source",
        ["Bundled example", "Upload your own"],
        help="Bundled datasets come from the repository's data/ folder.",
    )

    payloads = {
        "abundance": None, "metadata": None, "phylogeny": None,
        "sample_column": None, "taxa_as_rows": None, "group_column": None,
        "cache_token": None, "dataset_name": None,
    }

    if source_kind == "Bundled example":
        catalogue = available_example_datasets()
        choice = st.sidebar.selectbox(
            "Dataset",
            catalogue["name"].tolist(),
            format_func=lambda name: catalogue.loc[catalogue["name"] == name, "label"].iloc[0],
        )
        spec = EXAMPLE_DATASETS[choice]
        st.sidebar.caption(spec["description"])

        payloads.update(
            abundance=str(REPOSITORY_ROOT / spec["abundance"]),
            metadata=str(REPOSITORY_ROOT / spec["metadata"]) if spec.get("metadata") else None,
            phylogeny=str(REPOSITORY_ROOT / DEFAULT_PHYLOGENY),
            sample_column=spec["sample_column"],
            taxa_as_rows=spec["taxa_as_rows"],
            group_column=spec["group_column"],
            cache_token=f"example::{choice}",
            dataset_name=choice,
        )
    else:
        abundance_file = st.sidebar.file_uploader(
            "Abundance table (CSV) *",
            type=["csv", "txt", "tsv"],
            help="Taxa × samples or samples × taxa. Orientation is detected and can be overridden below.",
        )
        metadata_file = st.sidebar.file_uploader(
            "Sample metadata (CSV, optional)",
            type=["csv", "txt", "tsv"],
            help="Needed to colour samples by group. The first column should hold sample identifiers.",
        )
        phylogeny_file = st.sidebar.file_uploader(
            "Phylogeny / taxon order (CSV, optional)",
            type=["csv", "txt", "tsv"],
            help="A single column of taxonomy paths, root to genus, in phylogenetic order. Defaults to the bundled phylogeny.",
        )

        if abundance_file is not None:
            abundance_bytes = abundance_file.getvalue()
            payloads["abundance"] = abundance_bytes
            payloads["metadata"] = metadata_file.getvalue() if metadata_file is not None else None
            payloads["phylogeny"] = (
                phylogeny_file.getvalue() if phylogeny_file is not None
                else str(REPOSITORY_ROOT / DEFAULT_PHYLOGENY)
            )
            payloads["cache_token"] = f"upload::{len(abundance_bytes)}::{abundance_bytes[:64].hex()}"

            header = pd.read_csv(io.BytesIO(abundance_bytes), nrows=0).columns.tolist()
            first_column = st.sidebar.selectbox(
                "What is the table's first column?",
                ["Sample identifiers", "Taxon names"],
                help="Only used to read the file; the orientation of the matrix is detected next.",
            )
            payloads["sample_column"] = header[0] if first_column == "Sample identifiers" else None

            orientation_choice = st.sidebar.radio(
                "Table orientation",
                ["Detect automatically", "Samples are rows", "Taxa are rows"],
                help="Detection uses shared sample identifiers, then how many labels resolve onto the phylogeny.",
            )
            payloads["taxa_as_rows"] = {
                "Detect automatically": None, "Samples are rows": False, "Taxa are rows": True,
            }[orientation_choice]

            if payloads["metadata"] is not None:
                try:
                    columns = pd.read_csv(io.BytesIO(payloads["metadata"]), nrows=0).columns.tolist()
                except Exception:
                    columns = []
                # The first metadata column becomes the sample index, so it cannot
                # also be the grouping column.
                candidates = columns[1:] or columns
                if candidates:
                    payloads["group_column"] = st.sidebar.selectbox(
                        "Group column", candidates, index=0,
                        help="Samples are coloured and compared by this column.",
                    )

    st.sidebar.divider()
    st.sidebar.subheader("Spectral parameters")

    params = {
        "pseudocount": st.sidebar.number_input(
            "Pseudocount", min_value=1e-12, max_value=1e-1, value=1e-9, format="%.1e",
            help="Added before the log transform, as in the manuscript analysis.",
        ),
        "fmax": st.sidebar.slider(
            "fmax · highest retained frequency", min_value=0.05, max_value=0.50, value=0.20, step=0.01,
            help="The spectral slope is fitted inside 2/n_taxa … fmax.",
        ),
        "use_hann_window": st.sidebar.checkbox("Hann window before the transform", value=True),
        "low_fraction": st.sidebar.slider("Low-frequency fraction", 0.05, 0.50, 0.25, 0.05),
        "high_fraction": st.sidebar.slider("High-frequency fraction", 0.05, 0.50, 0.35, 0.05),
        "max_samples": st.sidebar.select_slider(
            "Maximum samples", options=[100, 200, 400, 800, 1600, 3200, 999999], value=800,
            format_func=lambda value: "all" if value == 999999 else str(value),
            help="Caps the computation for very large tables; sampling is random with a fixed seed.",
        ),
        "min_samples_per_group": st.sidebar.number_input("Minimum samples per group", min_value=1, value=1, step=1),
    }

    return payloads, params


def load_result(payloads, params):
    if payloads["abundance"] is None:
        return None, None

    try:
        result = compute_bundle(
            payloads["abundance"], payloads["metadata"], payloads["phylogeny"],
            payloads["sample_column"], payloads["taxa_as_rows"], payloads["group_column"],
            float(params["pseudocount"]), float(params["fmax"]), bool(params["use_hann_window"]),
            float(params["low_fraction"]), float(params["high_fraction"]),
            None if params["max_samples"] == 999999 else int(params["max_samples"]),
            int(params["min_samples_per_group"]), payloads["cache_token"],
        )
        return result, None
    except Exception as error:
        return None, str(error)


# ── pages ─────────────────────────────────────────────────────────────────────


def page_overview():
    result = STATE["result"]
    summary = result.summary()

    st.title("🧬 Phylogeny-ordered spectral explorer")
    st.caption(
        "Order a microbial community along a phylogenetic axis, decompose it into spectral "
        f"modes, and compare the result across groups. [Repository]({REPOSITORY_URL})"
    )

    tiles = [
        ("Samples", f"{summary['n_samples']:,}"),
        ("Taxa on the phylogeny", f"{summary['n_taxa']:,}"),
        ("Input taxa matched", f"{summary['matched_taxa']:,} / {summary['input_taxa']:,}"),
        ("Groups", f"{summary['n_groups']}" if summary["n_groups"] else "—"),
        ("Median spectral slope β", f"{summary['median_spectral_slope']:.3f}"),
        ("Median C80", f"{summary['median_c80']:.3f}"),
    ]
    for column, (label, value) in zip(st.columns(len(tiles)), tiles):
        column.metric(label, value)

    st.markdown("---")

    # Three short cards rather than text beside a tall chart: a row of uneven
    # columns leaves a dead band under whichever side is shorter.
    cards = st.columns(3)
    with cards[0]:
        st.markdown(
            "##### 1 · Transform\n"
            "Pick a sample and watch it come apart. Signal on the left, Fourier "
            "decomposition on the right; keep fewer components and the signal "
            "rebuilds from the broadest ones."
        )
    with cards[1]:
        st.markdown(
            "##### 2 · Spectra\n"
            "The same decomposition for every sample at once — power spectra by "
            "group, spectral-slope distributions with their statistics, the "
            "macro–micro space and compressibility."
        )
    with cards[2]:
        st.markdown(
            "##### 3 · Case studies\n"
            "Worked examples on the bundled datasets with live numbers, a contrast "
            "drawn from your own data, and a guide to reading the quantities. "
            "The **Guide** page covers inputs and every parameter."
        )

    report = result.match_report
    with st.expander("How the taxa were placed on the phylogeny", expanded=False):
        st.markdown(
            f"**{report['n_matched']:,} of {report['n_input_taxa']:,} input taxa** were placed on the "
            f"phylogeny — {report['n_by_path']:,} by full taxonomy path and {report['n_by_genus']:,} by "
            f"genus name. The {result.n_taxa:,} retained entries form the analysis axis."
        )
        if report["n_unmatched"]:
            st.caption(
                f"{report['n_unmatched']:,} taxa could not be placed and were dropped. They are listed "
                "below — usually renamed genera, or ranks below genus."
            )
            show_table(unmatched_taxa(result.details), "Unmatched taxa", "unmatched_taxa.csv", "unmatched")
        st.caption(
            f"Orientation: samples as rows = {not result.params['taxa_as_rows']} — "
            f"{result.params['orientation_reason']}."
        )

    st.subheader("The whole dataset at a glance")
    st.caption(
        "Every sample as a row, ordered along the phylogeny. Broad blocks of colour are "
        "low-frequency organization; rapid switching is high-frequency fragmentation."
    )
    st.plotly_chart(
        figure_signal_heatmap(result, STATE["mode"], max_rows=200, sort_by_group=result.group is not None),
        width="stretch",
    )


def page_transform():
    result = STATE["result"]
    mode = STATE["mode"]
    samples = result.clr.index.tolist()

    st.title("🔬 How the transform works")
    st.caption(
        "One community, decomposed. On the left, the sample as a signal along the phylogeny: the "
        "centered log-ratio of its abundance, one value per taxon in phylo order. On the right, that "
        "signal taken apart into frequency components — the broad waves that span whole clades and the "
        "fine ones that move between neighbouring taxa. Drag the slider to keep only the lowest "
        "components and watch the signal rebuild out of them."
    )

    key = "transform_sample_index"
    if key not in st.session_state:
        st.session_state[key] = 0

    previous, following, selector, beta_column = st.columns([1, 1, 6, 2])
    if previous.button("◀ previous", width="stretch"):
        st.session_state[key] = (st.session_state[key] - 1) % len(samples)
    if following.button("next ▶", width="stretch"):
        st.session_state[key] = (st.session_state[key] + 1) % len(samples)

    index = selector.selectbox(
        "Sample", range(len(samples)), key=key,
        format_func=lambda position: (
            f"{samples[position]}"
            + (f"  ·  {result.group.loc[samples[position]]}" if result.group is not None else "")
        ),
    )
    sample_id = samples[index]

    signal = result.clr.loc[sample_id].to_numpy()
    coefficients = np.fft.rfft(signal)
    n_modes = len(coefficients) - 1
    beta = float(result.slopes.loc[sample_id, "beta"])
    c80 = float(result.compressibility.loc[sample_id, "C80"])
    default_keep = int(np.clip(round(c80 * n_modes), 1, n_modes))
    beta_column.metric("Sample β", f"{beta:.3f}" if np.isfinite(beta) else "n/a")

    # Labelled widgets share one row (their labels sit above the control), and the
    # checkboxes take their own: a checkbox puts its label beside the box, so mixing
    # the two kinds in one row leaves it visibly ragged.
    controls = st.columns([4, 2, 2])
    keep = controls[0].slider(
        "Frequency components kept", min_value=0, max_value=n_modes, value=default_keep,
        help="Starts at the sample's own C80 — the number of modes that carry 80% of its spectral energy.",
    )
    left_view = controls[1].radio(
        "Left signal", [CLR_LABEL, "Abundance"], index=0,
        help="The transform is applied to the centered log-ratio, so the CLR view is the exact signal "
             "being decomposed; the abundance view is there to check it against the raw table.",
    )
    camera_view = controls[2].selectbox(
        "Camera", list(CAMERA_VIEWS), index=0,
        help="Only affects the 3D view; the chart is drag-rotatable too.",
    )

    toggles = st.columns([2, 2, 6])
    three_d = toggles[0].checkbox("3D view", value=True, help="Uncheck for a flat spectrum view.")
    show_windowed = toggles[1].checkbox("Hann-windowed spectrum", value=False,
                                        help="Overlay the windowed spectrum in the amplitude view below.")

    reconstruction, retained = reconstruct_signal(signal, keep)
    captured = energy_fraction(coefficients, retained)
    rmse = float(np.sqrt(np.mean((signal - reconstruction) ** 2)))
    dominant = int(np.argmax(np.abs(coefficients[1:])) + 1)

    metrics = st.columns(4)
    metrics[0].metric("Spectral energy kept", f"{captured:.1%}")
    metrics[1].metric("Reconstruction error (RMSE)", f"{rmse:.3f}")
    metrics[2].metric("Strongest component", f"#{dominant} · f={dominant / len(signal):.3f}")
    metrics[3].metric("Sample C80", f"{c80:.3f}")

    signal_column, spectrum_column = st.columns([1, 1.25])
    with signal_column:
        st.plotly_chart(
            figure_signal_panel(result, sample_id, left_view, reconstruction, mode),
            width="stretch",
        )
    with spectrum_column:
        if three_d:
            st.plotly_chart(
                figure_decomposition_3d(result, sample_id, keep, mode, view=camera_view),
                width="stretch",
            )
        else:
            st.plotly_chart(
                figure_signal_decomposition(result, sample_id, keep, True, mode),
                width="stretch",
            )

    with st.expander("Amplitude spectrum · which components are kept"):
        st.plotly_chart(
            figure_amplitude_spectrum(result, sample_id, keep, mode, show_windowed=show_windowed),
            width="stretch",
        )

    st.info(
        "**Reading it.** The component nearest the signal is the broadest wave across the whole "
        "phylogeny; the ones further back oscillate faster, down to variation between neighbouring "
        "taxa. Keeping the lowest few reproduces the community's broad organization, and what they "
        f"leave behind is the fine-scale detail. With all {n_modes} components kept the dashed line "
        "sits exactly on the black one."
        + (
            f"\n\nThe stack draws the first 5 of the {keep} kept components; the amplitude spectrum in the "
            f"expander below shows all {n_modes}." if keep > 5 else ""
        )
        + "\n\nBlack is the signal being decomposed — the centered log-ratio of the abundance, which is "
          "what every spectral number in this app is computed on — and the blue dashed line is what the "
          "kept components rebuild from it. No window is applied here, so the round trip is exact; the "
          "slopes and compressibility elsewhere use a Hann window, as the manuscript does."
    )

    table = pd.DataFrame(
        {
            "component": np.arange(len(coefficients)),
            "frequency": np.fft.rfftfreq(len(signal)),
            "amplitude": np.abs(coefficients),
            "kept": np.arange(len(coefficients)) <= keep,
        }
    )
    show_table(table, "Table view · the Fourier coefficients of this sample",
               f"coefficients_{sample_id}.csv", "coefficients")


def page_spectra():
    result = STATE["result"]
    mode = STATE["mode"]

    st.title("📊 Spectra and group comparisons")

    if result.group is None:
        st.warning(
            "This dataset has no grouping column, so the group comparisons below are unavailable. "
            "Upload a metadata table and pick a **Group column** in the sidebar to enable them."
        )

    tabs = st.tabs(["Power spectrum", "Spectral slope", "Macro–micro space", "Compressibility"])

    with tabs[0]:
        st.caption(
            "Density against phylogenetic frequency, log–log. Thin lines are individual samples, thick "
            "lines the group median, bands the interquartile range, and dashed lines the fitted β."
        )
        toggles = st.columns(2)
        show_samples = toggles[0].checkbox("Show individual samples", value=True)
        show_band = toggles[1].checkbox("Show interquartile band", value=True)
        st.plotly_chart(
            figure_power_spectrum(result, mode, show_samples=show_samples, show_band=show_band),
            width="stretch",
        )
        if result.group is not None:
            show_table(result.group_summaries(), "Table view · group spectra", "group_spectra.csv", "group_spectra")

    with tabs[1]:
        st.caption(
            "One point per sample. β is the fitted exponent of the power spectrum over the analysis "
            "window — higher means energy concentrated in the broadest phylogenetic scales."
        )
        if result.group is None:
            st.info("A grouping column is required for this comparison.")
        else:
            st.plotly_chart(figure_slope_distribution(result, mode), width="stretch")
            stats = result.group_slope_stats()
            reference = stats["reference_group"].iloc[0] if len(stats) else None
            st.caption(
                f"Two-sided Mann–Whitney against **{reference}**, with Cliff's δ as the effect size. "
                "A small P with a δ near zero is a difference that is real but negligible."
            )
            st.dataframe(stats, width="stretch", hide_index=True)
            st.download_button(
                "Download group statistics (CSV)",
                stats.to_csv(index=False).encode(),
                file_name="group_slope_statistics.csv",
                mime="text/csv",
            )

    with tabs[2]:
        st.caption(
            "The manuscript's host spectral space: the low-frequency share of power against the "
            "high-frequency share. Symbols and labels carry group identity alongside colour."
        )
        if result.group is None:
            st.info("A grouping column is required for this view.")
        else:
            st.plotly_chart(figure_macro_micro(result, mode), width="stretch")
            show_table(result.macro_micro.reset_index(names="sample"),
                       "Table view · macro–micro coordinates", "macro_micro_scores.csv", "macro_micro")

    with tabs[3]:
        st.caption(
            "How many Fourier modes are needed to recover a given share of the spectral energy. "
            "Lower C80 means a more compressible community."
        )
        st.plotly_chart(figure_cumulative_energy(result, mode), width="stretch")
        if result.group is not None:
            st.plotly_chart(figure_c80_richness(result, mode), width="stretch")
        show_table(result.compressibility.reset_index(names="sample"),
                   "Table view · compressibility metrics", "spectral_compressibility.csv", "compressibility")


def page_cases():
    st.title("🔎 Case studies")
    st.caption(
        "Worked examples. The first uses whatever you have loaded; the rest run on bundled datasets, "
        "live — the numbers below are computed now, not quoted from the paper."
    )

    result = STATE["result"]
    mode = STATE["mode"]

    # ── contrast on the current data ─────────────────────────────────────────
    st.subheader("1 · The two extremes of your own data")
    slopes = result.slopes["beta"].dropna()
    if len(slopes) < 4:
        st.info("Load a dataset with more samples to see this contrast.")
    else:
        high_id = slopes.idxmax()
        low_id = slopes.idxmin()
        st.caption(
            f"**{high_id}** has the highest slope (β={slopes.loc[high_id]:.2f}) and **{low_id}** the "
            f"lowest (β={slopes.loc[low_id]:.2f}). Same pipeline, opposite ends of the spectral range."
        )

        left, right = st.columns(2)
        with left:
            figure = go.Figure()
            positions = np.arange(result.n_taxa)
            for sample_id, colour, label in [
                (high_id, SEQUENTIAL_BLUE[8], f"{high_id} · β={slopes.loc[high_id]:.2f} (most organized)"),
                (low_id, SEQUENTIAL_BLUE[2], f"{low_id} · β={slopes.loc[low_id]:.2f} (most fragmented)"),
            ]:
                figure.add_trace(
                    go.Scatter(
                        x=positions, y=result.clr.loc[sample_id].to_numpy(), mode="lines",
                        line=dict(color=colour, width=2), name=label,
                        hovertemplate="taxon #%{x}<br>CLR %{y:.2f}<extra>" + label + "</extra>",
                    )
                )
            figure.update_yaxes(title_text="Centered log-ratio")
            taxon_tick_axis(figure, result.taxa_order, count=14)
            st.plotly_chart(
                style_figure(figure, height=380, mode=mode).update_layout(title_text="Their signals"),
                width="stretch",
            )

        with right:
            figure = go.Figure()
            for sample_id, colour, label in [
                (high_id, SEQUENTIAL_BLUE[8], f"{high_id} · β={slopes.loc[high_id]:.2f}"),
                (low_id, SEQUENTIAL_BLUE[2], f"{low_id} · β={slopes.loc[low_id]:.2f}"),
            ]:
                signal = result.clr.loc[sample_id].to_numpy()
                frequency = np.fft.rfftfreq(len(signal))
                power = np.abs(np.fft.rfft(signal)) ** 2 / len(signal)
                figure.add_trace(
                    go.Scatter(
                        x=frequency[1:], y=power[1:], mode="lines",
                        line=dict(color=colour, width=2), name=label,
                        hovertemplate="f=%{x:.4f}<br>power %{y:.3g}<extra></extra>",
                    )
                )
            figure.update_xaxes(type="log", title_text="Phylogenetic frequency")
            figure.update_yaxes(type="log", title_text="Power spectral density")
            st.plotly_chart(
                style_figure(figure, height=380, mode=mode).update_layout(title_text="Their spectra", hovermode="x unified"),
                width="stretch",
            )

        st.info(
            "The organized sample's power falls away steeply with frequency, so a handful of low "
            "components carry most of its structure. The fragmented sample stays energetic at high "
            "frequency: its signal is dominated by taxon-to-taxon jumps rather than broad clades."
        )

    st.markdown("---")
    st.subheader("2 · Bundled examples")
    st.caption("Each loads a dataset from `data/` and runs the same pipeline. Pick one to see it in context.")

    for name in ("infant", "palleja", "hmp_site", "crc"):
        spec = EXAMPLE_DATASETS[name]
        with st.expander(f"**{spec['label']}** — {spec['description']}"):
            try:
                example = compute_example(name)
            except Exception as error:
                st.error(f"Could not run this example: {error}")
                continue

            stats = example.group_slope_stats()
            reference = stats["reference_group"].iloc[0]
            st.plotly_chart(figure_slope_distribution(example, mode, compact=True), width="stretch")

            framing = {
                "infant": (
                    "Gut communities that mature early look spectrally *smoother*: their power is "
                    "concentrated in broad, low-frequency structure. Compare the medians below — and "
                    "note how much of the spread is within-group, not between-group."
                ),
                "palleja": (
                    "An antibiotic perturbation is a directional spectral move rather than a change in "
                    "overall variability. The unstable phase sits at the fragmented end; recovery "
                    "returns toward the baseline range."
                ),
                "hmp_site": (
                    "Body sites differ enormously in spectral organization — oral communities are far "
                    "more fragmented than gut ones. This is the largest between-group separation in "
                    "the bundled data, so it is a good reference for what a strong effect looks like."
                ),
                "crc": (
                    "A caution about effect sizes. The CRC difference is statistically detectable at "
                    "this sample size but tiny compared with the within-group spread — exactly the "
                    "case where a P value alone would mislead."
                ),
            }[name]
            st.markdown(framing)

            display = stats[["group", "n", "median_slope", "p_vs_reference", "cliffs_delta"]].rename(
                columns={
                    "group": "group", "n": "samples", "median_slope": "median β",
                    "p_vs_reference": "P vs " + reference, "cliffs_delta": "Cliff's δ",
                }
            )
            st.dataframe(display, width="stretch", hide_index=True)

            if st.button(f"Load this dataset in the sidebar", key=f"load_{name}"):
                st.info(
                    f"Set **Source → Bundled example → {spec['label']}** in the sidebar to explore it "
                    "across all pages."
                )

    st.markdown("---")
    st.subheader("3 · What to look for")
    st.markdown(
        """
| Quantity | Reads as | Worth a second look when |
|---|---|---|
| **β** (spectral slope) | Steeper = power concentrated at broad phylogenetic scales | β near 0 or negative: the signal is dominated by fine-scale noise, so a group difference may just reflect sequencing depth or richness |
| **Sample spread** | The width of a group's β distribution | The between-group gap is smaller than the within-group spread — the group label explains little |
| **P with Cliff's δ** | P tests for a difference, δ for its size | P significant but \\|δ\\| < 0.15: real but negligible (see the CRC example) |
| **Macro–micro position** | Low-frequency share vs high-frequency share | Groups separate along the diagonal rather than into a region of their own |
| **C80** | Modes needed for 80% of the energy | C80 tracks richness closely, so compare at matched richness before concluding |
        """
    )


def page_guide():
    st.title("📖 Guide")

    st.subheader("Quick start")
    st.markdown(
        """
1. In the sidebar, keep **Source → Bundled example** to try the framework, or switch to
   **Upload your own** and drop in a CSV.
2. If you upload a metadata table, pick the **Group column** that should colour and split the samples.
3. Adjust **pseudocount** and **fmax** if needed — the defaults follow the manuscript.
4. Read the result on **Transform**, **Spectra** and **Case studies**.

Nothing is written to disk and nothing leaves your browser session; uploads are held in memory only.
        """
    )

    st.subheader("Input formats")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
**Abundance table** — one CSV, samples × taxa or taxa × samples:

|  | Bacteroides | Faecalibacterium | Prevotella |
|---|---|---|---|
| S1 | 10 | 25 | 3 |
| S2 | 14 | 18 | 1 |

- Counts or relative abundances; rows are normalised internally.
- The first column may hold sample identifiers (default) or taxon names.
- Orientation is detected automatically and can be overridden in the sidebar.

**Metadata table** — optional, one row per sample, first column = sample identifier:

| | label | study_name |
|---|---|---|
| S1 | healthy | CohortA |
| S2 | disease | CohortA |
            """
        )
    with right:
        st.markdown(
            """
**Phylogeny table** — optional, defaults to the bundled `data/phylogeny.csv`:

| taxon |
|---|
| k__Bacteria;p__Firmicutes;...;g__Faecalibacterium |
| k__Bacteria;p__Bacteroidota;...;g__Bacteroides |

A single column of taxonomy paths in phylogenetic order. Any table with a
phylogenetic ordering of taxa works — a tree export's tip order, for instance.

**How labels are matched**

- Full paths are matched first, exactly.
- Anything left over is matched by genus name, so `Bacteroides`,
  `Bacteroides vulgatus` and `g__Bacteroides` all resolve.
- Taxa that match nothing are dropped, and the Overview page reports how many and
  which ones. **Check that count first** when a result looks wrong.
            """
        )

    st.subheader("Parameters")
    st.markdown(
        """
| Parameter | Default | Meaning | Change it when |
|---|---|---|---|
| `pseudocount` | 1e-9 | Added before the log transform | Your table has many zeros and the spectrum looks like pure noise; larger values smooth rare taxa |
| `fmax` | 0.20 | Upper end of the fitted frequency window | Your table has few taxa (roughly < 20) and the analysis refuses to run — raise it |
| Hann window | on | Tapers the signal before transforming | You are reproducing the Transform page's exact round trip |
| Low / high fraction | 0.25 / 0.35 | Share of modes counted as "macro" and "micro" | You want a stricter or looser definition of the ends of the spectrum |
| Maximum samples | 800 | Random cap on samples per run | Your table is large and the app feels slow; raise for the full analysis |
| Minimum samples per group | 1 | Drops tiny groups | A group has too few samples to say anything about it |
        """
    )

    st.subheader("When something fails")
    st.markdown(
        """
| Message | Cause | Fix |
|---|---|---|
| *Only N taxa matched the phylogeny* | Taxon labels and the phylogeny describe different taxonomies | Check the unmatched list on the Overview page; supply your own phylogeny table |
| *Only N frequency mode(s) fall between fmin … and fmax …* | Too few taxa for the chosen band | Raise `fmax`, or use a table with more taxa |
| *Only N frequency mode(s) were retained … needs at least 6* | Fewer than ~14 taxa | Above, plus: the low/middle/high split needs at least six modes |
| *Fewer than two groups have at least N samples* | A small group was filtered out | Lower **Minimum samples per group**, or pick another group column |
| *No sample identifier is shared…* | Metadata identifiers do not match abundance identifiers | Make the two tables use the same sample names |
| An odd number of taxa on the axis | The two values disagree about orientation | Set **Table orientation** explicitly in the sidebar |
| The run is slow | A large table | Lower **Maximum samples** |
        """
    )

    st.subheader("Reproducing a result outside the app")
    result = STATE["result"]
    st.markdown("The app is a thin layer over the package, so the same numbers come out of four lines of Python:")
    st.code(
        "from phylospectra.pipeline import run_spectral_pipeline\n\n"
        "result = run_spectral_pipeline(\n"
        '    "your_abundance.csv",\n'
        '    metadata="your_metadata.csv",\n'
        f"    group_column={result.group_column!r},\n"
        f"    pseudocount={float(STATE['params']['pseudocount']):.1e},\n"
        f"    fmax={float(STATE['params']['fmax'])},\n"
        f"    use_hann_window={bool(STATE['params']['use_hann_window'])},\n"
        ")\n\n"
        "result.summary()            # counts, match report, medians\n"
        "result.slopes               # per-sample spectral slope\n"
        "result.compressibility      # C50 / C80 / C90, effective spectral dimension\n"
        "result.macro_micro          # the two macro–micro coordinates\n"
        "result.group_slope_stats()  # per-group medians, P values, Cliff's delta",
        language="python",
    )

    st.subheader("Other ways to run this")
    st.markdown(
        f"""
- **Command line** — the manuscript scripts in `paper_code/` accept `--abundance`,
  `--metadata`, `--phylogeny` and the same spectral parameters.
- **Deploy your own copy** — push the repository to GitHub and point
  [share.streamlit.io](https://share.streamlit.io) at `webapp/streamlit_app.py`; the repository
  root's `requirements.txt` is installed automatically.
- **Cite** — see the repository [README]({REPOSITORY_URL}) for the manuscript reference.
        """
    )


def page_data():
    result = STATE["result"]
    params = dict(result.params)

    st.title("💾 Data and downloads")
    st.caption(
        "Diversity, spectral slope, compressibility and macro–micro coordinates in one table — "
        "the table view behind every chart in the app."
    )

    per_sample = result.sample_table()
    st.dataframe(per_sample, width="stretch", height=420, hide_index=True)
    st.download_button(
        "Download per-sample metrics (CSV)",
        per_sample.to_csv(index=False).encode(),
        file_name="per_sample_spectral_metrics.csv",
        mime="text/csv",
    )

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("The taxon axis")
        st.caption("The phylo-ordered entries every spectrum is computed on, in order.")
        st.dataframe(result.taxa_order, width="stretch", height=300, hide_index=True)
        st.download_button(
            "Download the taxon axis (CSV)",
            result.taxa_order.to_csv(index=False).encode(),
            file_name="taxa_phylogeny_order.csv",
            mime="text/csv",
        )

    with right:
        st.subheader("Parameters in force")
        st.caption("What this run used, including the orientation and windowing decisions.")
        st.dataframe(
            pd.DataFrame({"parameter": list(params), "value": [str(value) for value in params.values()]}),
            width="stretch", height=300, hide_index=True,
        )
        if result.metadata is not None:
            st.download_button(
                "Download the aligned metadata (CSV)",
                result.metadata.to_csv().encode(),
                file_name="aligned_metadata.csv",
                mime="text/csv",
            )


# ── run ───────────────────────────────────────────────────────────────────────

navigation = st.navigation(
    [
        st.Page(page_overview, title="Overview", icon="🧬", default=True),
        st.Page(page_guide, title="Guide", icon="📖"),
        st.Page(page_transform, title="Transform", icon="🔬"),
        st.Page(page_spectra, title="Spectra", icon="📊"),
        st.Page(page_cases, title="Case studies", icon="🔎"),
        st.Page(page_data, title="Data", icon="💾"),
    ]
)

payloads, params = render_sidebar()
result, error = load_result(payloads, params)

STATE["mode"] = active_mode()
STATE["params"] = params

if error is not None:
    st.error(f"**Could not build the analysis:** {error}")
    st.caption(
        "Common causes, all adjustable in the sidebar: the taxon labels do not overlap the phylogeny "
        "table; the metadata carries no shared sample identifiers; the table orientation is the wrong "
        "way round; or the table has so few taxa that fmax falls below 2/n_taxa — raise **fmax** to fit "
        "at least three modes. The **Guide** page lists every message with its fix."
    )
elif result is None:
    st.title("🧬 Phylogeny-ordered spectral explorer")
    st.info(
        "**Upload an abundance table in the sidebar to begin**, or switch the source to a bundled "
        "example dataset. Expected layout: a CSV whose rows are samples and columns are taxa (a "
        "transposed table works too — the orientation is detected). A metadata CSV with a group "
        "column enables the group comparisons."
    )
else:
    STATE["result"] = result
    navigation.run()
