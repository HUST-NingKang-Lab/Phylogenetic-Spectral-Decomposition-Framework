"""Turn an abundance table into phylogeny-ordered spectra in one call.

This is the shared backend of the interactive tools -- ``webapp/streamlit_app.py``
and ``notebooks/phylospectra_interactive_visualization.ipynb`` both import it, so
the browser app and the notebook always report identical numbers.

A user's table rarely arrives in the shape the manuscript scripts assume: samples
may be rows or columns, taxon labels may be full ``k__;p__;...;g__`` paths or bare
genus names. The loaders here detect the orientation and match labels to the
phylogeny along both routes, then report how many taxa were matched.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu

from .evaluation import alpha_diversity, cliffs_delta
from .io import (
    collapse_to_genus,
    normalize_taxon_name,
    read_abundance_table,
    read_phylogeny_order,
    relative_abundance,
    centered_log_ratio,
)
from .spectral import (
    compressibility_metrics,
    cumulative_energy,
    macro_micro_scores,
    spectral_power,
    spectral_slope,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

EXAMPLE_DATASETS = {
    "ibd": {
        "label": "IBD vs healthy (multi-cohort)",
        "description": "Human gut 16S/metagenomes; healthy vs IBD across several studies.",
        "abundance": "data/IBD/abundance.csv",
        "metadata": "data/IBD/meta.csv",
        "sample_column": "sample_id",
        "group_column": "label",
        "secondary_group_column": "study_name",
        "taxa_as_rows": False,
    },
    "infant": {
        "label": "Infant gut maturation",
        "description": "Infant gut communities labelled immature vs mature.",
        "abundance": "data/infant/abundance.csv",
        "metadata": "data/infant/metadata.csv",
        "sample_column": "sample_id",
        "group_column": "group",
        "secondary_group_column": "country",
        "taxa_as_rows": False,
    },
    "crc": {
        "label": "Colorectal cancer (7 cohorts)",
        "description": "Multi-batch CRC metagenomes; CRC vs healthy.",
        "abundance": "data/7CRC/abundance.csv",
        "metadata": "data/7CRC/meta.csv",
        "sample_column": None,
        "group_column": "disease",
        "secondary_group_column": "batch",
        "taxa_as_rows": False,
    },
    "hmp_site": {
        "label": "HMP body sites",
        "description": "Human Microbiome Project samples across five body sites.",
        "abundance": "data/HMP_site/abundance.csv",
        "metadata": "data/HMP_site/metadata.csv",
        "sample_column": None,
        "group_column": "SITE",
        "secondary_group_column": "SEX",
        "taxa_as_rows": True,
    },
    "etec": {
        "label": "ETEC challenge",
        "description": "Controlled ETEC infection; two dose arms sampled over time.",
        "abundance": "data/etec/etec16s_abundance_genus.csv",
        "metadata": "data/etec/etec16s_metadata.csv",
        "sample_column": None,
        "group_column": "Dose",
        "secondary_group_column": "Day",
        "taxa_as_rows": True,
    },
    "okeefe": {
        "label": "O'Keefe diet swap",
        "description": "Diet-swap intervention; HE/ED/DI arms.",
        "abundance": "data/Okeefe/OKeefe_dietswap_abundance.csv",
        "metadata": "data/Okeefe/OKeefe_dietswap_metadata.csv",
        "sample_column": None,
        "group_column": "group",
        "secondary_group_column": "bmi_group",
        "taxa_as_rows": True,
    },
    "palleja": {
        "label": "Palleja antibiotic recovery",
        "description": "Antibiotic perturbation and recovery; stable vs unstable phases.",
        "abundance": "data/Palleja/abundance.csv",
        "metadata": "data/Palleja/meta.csv",
        "sample_column": "sample_id",
        "group_column": "label",
        "secondary_group_column": "timepoint",
        "taxa_as_rows": False,
    },
}

DEFAULT_PHYLOGENY = "data/phylogeny.csv"

_TAXONOMY_TOKENS = (";", "|", "__")
_SAMPLE_COLUMN_NAMES = {"sample", "sample_id", "sampleid", "sample-id", "id", "sample_name"}


def available_example_datasets():
    return pd.DataFrame(
        [
            {
                "name": name,
                "label": spec["label"],
                "description": spec["description"],
                "group_column": spec["group_column"],
                "taxa_as_rows": spec["taxa_as_rows"],
            }
            for name, spec in EXAMPLE_DATASETS.items()
        ]
    )


def repository_root():
    return REPOSITORY_ROOT


def _looks_taxonomic(labels):
    labels = [str(label) for label in labels]
    if len(labels) == 0:
        return 0.0
    flagged = sum(any(token in label for token in _TAXONOMY_TOKENS) for label in labels)
    return flagged / len(labels)


def _looks_numeric(labels):
    labels = [str(label) for label in labels]
    if len(labels) == 0:
        return 0.0
    numeric = sum(bool(label.strip()) and label.strip().replace(".", "", 1).isdigit() for label in labels)
    return numeric / len(labels)


def detect_orientation(table, metadata=None, phylogeny=None, root=None):
    """Decide whether samples are rows.

    Three signals, strongest first: sample identifiers shared with the metadata,
    how many labels on each axis resolve onto the phylogeny, then the shape of the
    labels themselves. Returns True when the table must be transposed (taxa are
    rows). The guess is only a default -- every caller exposes it as a
    user-overridable switch, and the reason is returned so the interface can
    explain itself.
    """
    column_labels = [str(column) for column in table.columns]

    if isinstance(metadata, pd.DataFrame) and len(metadata) > 0:
        row_labels = pd.Index([str(label) for label in table.index])
        metadata_ids = pd.Index(metadata.index.astype(str))
        as_rows = len(row_labels.intersection(metadata_ids))
        as_columns = len(pd.Index(column_labels).intersection(metadata_ids))
        if max(as_rows, as_columns) > 0:
            if as_columns > as_rows:
                return True, f"{as_columns} column labels match the metadata sample identifiers"
            return False, f"{as_rows} row labels match the metadata sample identifiers"

    if phylogeny is not None:
        try:
            path_positions, genus_positions = _phylogeny_lookups(phylogeny, root=root)
            column_score = _phylogeny_match_fraction(column_labels, path_positions, genus_positions)
            index_score = _phylogeny_match_fraction([str(label) for label in table.index], path_positions, genus_positions)
        except Exception:
            column_score = index_score = 0.0

        # Compare the axes rather than clearing an absolute bar: a table whose
        # labels only partly resolve (species names, renamed genera) still points
        # at the right axis as long as the other axis resolves less.
        if max(column_score, index_score) >= 0.2 and abs(column_score - index_score) >= 0.1:
            if index_score > column_score:
                return True, f"{index_score:.0%} of row labels resolve onto the phylogeny, so taxa are rows"
            return False, f"{column_score:.0%} of column labels resolve onto the phylogeny, so samples are rows"

    lowered = {label.strip().lower() for label in column_labels}
    if lowered & _SAMPLE_COLUMN_NAMES:
        return False, "a sample identifier column is present, so samples are rows"

    column_taxonomy = _looks_taxonomic(column_labels)
    index_taxonomy = _looks_taxonomic(table.index)

    if column_taxonomy >= 0.5 and index_taxonomy < column_taxonomy:
        return False, "column labels look like taxonomy paths, so samples are rows"

    if index_taxonomy >= 0.5 and column_taxonomy < index_taxonomy:
        return True, "row labels look like taxonomy paths, so taxa are rows"

    column_numeric = _looks_numeric(column_labels)
    index_numeric = _looks_numeric(table.index)

    if column_numeric >= 0.5 and index_numeric < column_numeric:
        return True, "column labels look like sample identifiers, so taxa are rows"

    return False, "defaulting to samples as rows"


def read_abundance_any(source, sample_column=None, index_col=0):
    """Read an abundance table from a path or an uploaded file object."""
    if isinstance(source, (str, Path)):
        return read_abundance_table(source, sample_column=sample_column, index_col=index_col)

    table = pd.read_csv(source, low_memory=False, index_col=None if sample_column is not None else index_col)
    if sample_column is not None:
        if sample_column not in table.columns:
            raise ValueError(f"Sample column '{sample_column}' was not found in the abundance table.")
        table[sample_column] = table[sample_column].astype(str)
        table = table.set_index(sample_column)

    table.index = table.index.astype(str)
    return table.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def load_abundance_any(source, sample_column=None, taxa_as_rows=None, index_col=0, metadata=None, phylogeny=None, root=None):
    """Read an abundance table and settle its orientation.

    ``taxa_as_rows`` of None means "detect it"; pass True/False to override.
    """
    table = read_abundance_any(source, sample_column=sample_column, index_col=index_col)
    reason = "given by the caller"

    if taxa_as_rows is None:
        taxa_as_rows, reason = detect_orientation(table, metadata=metadata, phylogeny=phylogeny, root=root)

    if taxa_as_rows:
        table = table.T

    table.index = table.index.astype(str)
    table.columns = [str(column) for column in table.columns]
    return table, {"taxa_as_rows": bool(taxa_as_rows), "orientation_reason": reason}


def load_metadata_any(source, sample_column=None, index_col=0):
    metadata = pd.read_csv(source, low_memory=False)
    if sample_column is not None:
        if sample_column not in metadata.columns:
            raise ValueError(f"Sample column '{sample_column}' was not found in the metadata table.")
        metadata[sample_column] = metadata[sample_column].astype(str)
        metadata = metadata.set_index(sample_column)
    elif index_col is not None:
        metadata = metadata.set_index(metadata.columns[0])

    metadata.index = metadata.index.astype(str)
    return metadata


def load_example_dataset(name, root=None, max_samples=None, random_state=42):
    """Load a bundled dataset; returns (abundance, metadata, spec)."""
    if name not in EXAMPLE_DATASETS:
        raise KeyError(f"Unknown example dataset '{name}'. Available: {sorted(EXAMPLE_DATASETS)}")

    root = Path(root) if root is not None else REPOSITORY_ROOT
    spec = dict(EXAMPLE_DATASETS[name])
    abundance, _ = load_abundance_any(
        root / spec["abundance"],
        sample_column=spec["sample_column"],
        taxa_as_rows=spec["taxa_as_rows"],
    )
    metadata = None
    if spec.get("metadata"):
        metadata = load_metadata_any(root / spec["metadata"], sample_column=spec["sample_column"])

    if max_samples is not None and len(abundance) > max_samples:
        generator = np.random.default_rng(random_state)
        kept = generator.choice(abundance.index.to_numpy(), size=max_samples, replace=False)
        abundance = abundance.loc[kept]
        if metadata is not None:
            metadata = metadata.loc[metadata.index.intersection(abundance.index)]

    if metadata is not None:
        shared = abundance.index.intersection(metadata.index)
        abundance = abundance.loc[shared]
        metadata = metadata.loc[shared]

    return abundance, metadata, spec


@lru_cache(maxsize=8)
def _read_phylogeny_entries(path_string):
    table = pd.read_csv(path_string, low_memory=False)
    return tuple(table.iloc[:, 0].astype(str).tolist())


def phylogeny_entries(source, root=None):
    """Read the first column of a phylogeny table from a path, frame or upload."""
    if isinstance(source, pd.DataFrame):
        return source.iloc[:, 0].astype(str).tolist()

    if isinstance(source, (str, Path)):
        path = Path(source)
        if root is not None and not path.is_absolute():
            path = Path(root) / path
        return list(_read_phylogeny_entries(str(path)))

    return pd.read_csv(source, low_memory=False).iloc[:, 0].astype(str).tolist()


def build_phylogeny_orders(phylogeny_path, root=None):
    """Return the full-path order and a genus-name lookup built from it.

    Manuscript tables label taxa with the full ``k__;p__;...;g__`` path, while
    published OTU tables often carry only a genus or a species name. Both routes
    resolve to a position on the same phylogeny, so a table of either kind can be
    ordered.
    """
    seen = set()
    full_order = []

    for raw_value in phylogeny_entries(phylogeny_path, root=root):
        value = collapse_to_genus(raw_value)
        if value and value not in seen:
            seen.add(value)
            full_order.append(value)

    genus_positions = {}
    for position, entry in enumerate(full_order):
        genus = short_taxon_name(entry).strip()
        if genus and genus not in genus_positions:
            genus_positions[genus] = position

    return full_order, genus_positions


@lru_cache(maxsize=8)
def _phylogeny_lookups_cached(path_string):
    full_order, genus_positions = build_phylogeny_orders(Path(path_string))
    return (
        {entry: position for position, entry in enumerate(full_order)},
        genus_positions,
    )


def _phylogeny_lookups(phylogeny, root=None):
    if isinstance(phylogeny, (str, Path)):
        path = Path(phylogeny)
        if root is not None and not path.is_absolute():
            path = Path(root) / path
        return _phylogeny_lookups_cached(str(path))

    full_order, genus_positions = build_phylogeny_orders(phylogeny, root=root)
    return {entry: position for position, entry in enumerate(full_order)}, genus_positions


def _phylogeny_match_fraction(labels, path_positions, genus_positions):
    if len(labels) == 0:
        return 0.0

    matched = 0
    for label in labels:
        if collapse_to_genus(label) in path_positions or _genus_key(label) in genus_positions:
            matched += 1

    return matched / len(labels)


def _genus_key(label):
    collapsed = collapse_to_genus(label)
    if collapsed:
        genus = collapsed.split(";")[-1]
        if "__" in genus:
            genus = genus.split("__")[-1]
        return genus.strip()
    return ""


def match_taxa_to_phylogeny(taxon_labels, phylogeny_path=DEFAULT_PHYLOGENY, root=None):
    """Resolve each taxon label to a position on the phylogeny.

    Full-path labels are matched first; anything left over falls back to a
    genus-name match. Returns a frame with one row per input taxon.
    """
    path_positions, genus_positions = _phylogeny_lookups(phylogeny_path, root=root)

    records = []
    for label in taxon_labels:
        collapsed = collapse_to_genus(label)
        position = path_positions.get(collapsed)
        mode = "path"

        if position is None:
            genus = _genus_key(label)
            position = genus_positions.get(genus)
            mode = "genus" if position is not None else "unmatched"

        if position is None:
            collapsed = normalize_taxon_name(label)

        records.append(
            {
                "input_taxon": str(label),
                "matched_taxon": collapsed if position is not None else "",
                "phylogeny_position": position if position is not None else -1,
                "match_mode": mode,
            }
        )

    return pd.DataFrame(records)


def unmatched_taxa(details, limit=200):
    """The taxa that could not be placed on the phylogeny, for the user to inspect."""
    missing = details.loc[details["match_mode"] == "unmatched", ["input_taxon"]]
    return missing.head(limit).reset_index(drop=True)


def align_to_phylogeny(abundance, phylogeny_path=DEFAULT_PHYLOGENY, root=None, min_taxa=8, aggregation="sum"):
    """Collapse taxa to their phylogeny entry and sort columns along the phylogeny."""
    matches = match_taxa_to_phylogeny(abundance.columns, phylogeny_path=phylogeny_path, root=root)
    matched = matches[matches["match_mode"] != "unmatched"].copy()

    report = {
        "n_input_taxa": int(len(matches)),
        "n_matched": int(len(matched)),
        "n_unmatched": int((matches["match_mode"] == "unmatched").sum()),
        "n_by_path": int((matches["match_mode"] == "path").sum()),
        "n_by_genus": int((matches["match_mode"] == "genus").sum()),
    }

    if len(matched) < min_taxa:
        raise ValueError(
            f"Only {len(matched)} of {len(matches)} taxa matched the phylogeny, "
            f"but at least {min_taxa} are needed. Check that the taxon labels and the "
            "phylogeny table describe the same taxonomy."
        )

    retained = abundance.loc[:, matched["input_taxon"].values]
    retained.columns = matched["matched_taxon"].values

    if aggregation == "sum":
        retained = retained.T.groupby(level=0).sum().T
    else:
        retained = retained.T.groupby(level=0).mean().T

    positions = matched.drop_duplicates("matched_taxon").set_index("matched_taxon")["phylogeny_position"]
    ordered_taxa = [taxon for taxon in positions.sort_values().index if taxon in retained.columns]
    retained = retained.loc[:, ordered_taxa]

    taxa_order = pd.DataFrame(
        {
            "taxon_order": np.arange(1, retained.shape[1] + 1),
            "taxon_fullname": retained.columns,
            "taxon_short": [short_taxon_name(column) for column in retained.columns],
        }
    )

    retained_taxa = set(retained.columns)
    details = matches.assign(retained=matches["matched_taxon"].isin(retained_taxa))

    return retained, taxa_order, report, details


def short_taxon_name(value):
    last = str(value).split(";")[-1]
    if "__" in last:
        last = last.split("__")[-1]
    return last or str(value)


@dataclass
class SpectralResult:
    """Everything the interactive views need, computed once."""

    abundance: pd.DataFrame
    clr: pd.DataFrame
    power: pd.DataFrame
    frequency: np.ndarray
    frequency_mask: np.ndarray
    slopes: pd.DataFrame
    compressibility: pd.DataFrame
    cumulative: pd.DataFrame
    mode_fraction: np.ndarray
    macro_micro: pd.DataFrame
    diversity: pd.DataFrame
    taxa_order: pd.DataFrame
    metadata: pd.DataFrame = None
    group: pd.Series = None
    group_column: str = None
    params: dict = field(default_factory=dict)
    match_report: dict = field(default_factory=dict)
    details: pd.DataFrame = None

    @property
    def n_samples(self):
        return int(self.abundance.shape[0])

    @property
    def n_taxa(self):
        return int(self.abundance.shape[1])

    @property
    def analysis_frequency(self):
        return self.frequency[self.frequency_mask]

    @property
    def fmin(self):
        return 2.0 / self.n_taxa if self.n_taxa else np.nan

    def groups(self):
        if self.group is None:
            return []
        return sorted(self.group.dropna().astype(str).unique().tolist())

    def sample_table(self):
        """Per-sample metrics in one table, the table view behind every chart."""
        table = self.diversity.join(self.slopes).join(self.compressibility).join(self.macro_micro)
        if self.group is not None:
            table = table.join(self.group.rename("group"))
        return table.reset_index(names="sample")

    def group_summaries(self):
        """Median power spectrum and its interquartile band for each group."""
        records = []

        for group in self.groups():
            sample_ids = self.group.index[self.group.astype(str) == group]
            sample_ids = [sample for sample in sample_ids if sample in self.power.index]
            if len(sample_ids) == 0:
                continue

            block = self.power.loc[sample_ids].iloc[:, self.frequency_mask]
            median = block.median(axis=0)
            lower = block.quantile(0.25, axis=0)
            upper = block.quantile(0.75, axis=0)
            fit = self.slopes.loc[sample_ids, ["beta", "intercept"]].median()
            band = self.macro_micro.loc[sample_ids, ["low_frequency_macro_organization", "high_frequency_micro_fragmentation"]].median()

            records.append(
                pd.DataFrame(
                    {
                        "group": group,
                        "frequency": self.analysis_frequency,
                        "median_power": median.to_numpy(),
                        "q1_power": lower.to_numpy(),
                        "q3_power": upper.to_numpy(),
                        "group_beta": float(fit["beta"]),
                        "group_intercept": float(fit["intercept"]),
                        "n_samples": len(sample_ids),
                        "macro_organization": float(band["low_frequency_macro_organization"]),
                        "micro_fragmentation": float(band["high_frequency_micro_fragmentation"]),
                    }
                )
            )

        if len(records) == 0:
            return pd.DataFrame(
                columns=[
                    "group", "frequency", "median_power", "q1_power", "q3_power",
                    "group_beta", "group_intercept", "n_samples", "macro_organization", "micro_fragmentation",
                ]
            )

        return pd.concat(records, ignore_index=True)

    def group_slope_stats(self, reference=None):
        """Per-group slope summary, with effect sizes against a reference group."""
        groups = self.groups()
        if len(groups) == 0:
            return pd.DataFrame(columns=["group", "n", "median_slope", "q1_slope", "q3_slope", "p_vs_reference", "cliffs_delta"])
        if reference is None or reference not in groups:
            reference = groups[0]

        reference_values = self.slopes.loc[
            self.group.index[self.group.astype(str) == reference].intersection(self.slopes.index), "beta"
        ].dropna()

        records = []
        for group in groups:
            sample_ids = self.group.index[self.group.astype(str) == group].intersection(self.slopes.index)
            values = self.slopes.loc[sample_ids, "beta"].dropna()

            if group == reference or len(values) == 0 or len(reference_values) == 0:
                p_value = np.nan if group != reference else 1.0
                effect = 0.0 if group == reference else np.nan
            else:
                try:
                    p_value = float(mannwhitneyu(values, reference_values, alternative="two-sided").pvalue)
                except ValueError:
                    p_value = np.nan
                effect = cliffs_delta(values, reference_values)

            records.append(
                {
                    "group": group,
                    "n": int(len(values)),
                    "median_slope": float(values.median()) if len(values) else np.nan,
                    "q1_slope": float(values.quantile(0.25)) if len(values) else np.nan,
                    "q3_slope": float(values.quantile(0.75)) if len(values) else np.nan,
                    "p_vs_reference": p_value,
                    "cliffs_delta": effect,
                }
            )

        return pd.DataFrame(records).assign(reference_group=reference)

    def summary(self):
        return {
            "n_samples": self.n_samples,
            "n_taxa": self.n_taxa,
            "n_groups": len(self.groups()),
            "group_column": self.group_column,
            "median_spectral_slope": float(self.slopes["beta"].median()),
            "median_c80": float(self.compressibility["C80"].median()),
            "median_effective_spectral_dimension": float(self.compressibility["effective_spectral_dimension"].median()),
            "matched_taxa": self.match_report.get("n_matched"),
            "input_taxa": self.match_report.get("n_input_taxa"),
            "taxa_matched_by_path": self.match_report.get("n_by_path"),
            "taxa_matched_by_genus": self.match_report.get("n_by_genus"),
            "fmin": float(self.fmin) if self.n_taxa else np.nan,
            "fmax": self.params.get("fmax"),
        }


def run_spectral_pipeline(
    abundance,
    metadata=None,
    group_column=None,
    phylogeny=DEFAULT_PHYLOGENY,
    root=None,
    sample_column=None,
    taxa_as_rows=None,
    max_samples=None,
    min_samples_per_group=1,
    pseudocount=1e-9,
    fmax=0.20,
    use_hann_window=True,
    low_fraction=0.25,
    high_fraction=0.35,
    random_state=42,
):
    """Order taxa along the phylogeny and decompose every sample into spectra.

    ``abundance`` may be a path, a file-like object or a DataFrame; when a
    DataFrame is given its orientation is taken as already settled unless
    ``taxa_as_rows`` says otherwise.
    """
    orientation = {
        "taxa_as_rows": bool(taxa_as_rows) if taxa_as_rows is not None else None,
        "orientation_reason": "given as an already-oriented frame",
    }

    if metadata is not None and not isinstance(metadata, pd.DataFrame):
        metadata = load_metadata_any(metadata, sample_column=sample_column)

    if isinstance(abundance, (str, Path)):
        abundance_path = Path(abundance)
        if root is not None and not abundance_path.is_absolute():
            abundance_path = Path(root) / abundance_path
        abundance, orientation = load_abundance_any(
            abundance_path, sample_column=sample_column, taxa_as_rows=taxa_as_rows,
            metadata=metadata, phylogeny=phylogeny, root=root,
        )
    elif isinstance(abundance, pd.DataFrame):
        abundance = abundance.copy()
        abundance.index = abundance.index.astype(str)
        abundance.columns = [str(column) for column in abundance.columns]
        abundance = abundance.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    else:
        abundance, orientation = load_abundance_any(
            abundance, sample_column=sample_column, taxa_as_rows=taxa_as_rows,
            metadata=metadata, phylogeny=phylogeny, root=root,
        )

    if metadata is not None and not isinstance(metadata, pd.DataFrame):
        metadata = load_metadata_any(metadata, sample_column=sample_column)

    if metadata is not None:
        metadata.index = metadata.index.astype(str)
        shared = abundance.index.intersection(metadata.index)
        if len(shared) == 0:
            raise ValueError("No sample identifier is shared between the abundance table and the metadata.")
        abundance = abundance.loc[shared]
        metadata = metadata.loc[shared]

    if max_samples is not None and len(abundance) > max_samples:
        generator = np.random.default_rng(random_state)
        kept = np.sort(generator.choice(abundance.index.to_numpy(), size=max_samples, replace=False))
        abundance = abundance.loc[kept]
        if metadata is not None:
            metadata = metadata.loc[kept]

    ordered, taxa_order, match_report, details = align_to_phylogeny(abundance, phylogeny_path=phylogeny, root=root)

    relative = relative_abundance(ordered)
    relative = relative.loc[:, relative.sum(axis=0) > 0]
    if relative.shape[1] < 4:
        raise ValueError(f"Too few taxa survived filtering: {relative.shape[1]}")

    taxa_order = taxa_order[taxa_order["taxon_fullname"].isin(relative.columns)].reset_index(drop=True)
    taxa_order["taxon_order"] = np.arange(1, len(taxa_order) + 1)

    clr = centered_log_ratio(relative, pseudocount=pseudocount)
    n_taxa = clr.shape[1]
    fmin = 2.0 / n_taxa
    frequency_mask = (np.fft.rfftfreq(n_taxa, d=1.0) >= fmin) & (np.fft.rfftfreq(n_taxa, d=1.0) <= fmax)

    if int(frequency_mask.sum()) < 3:
        raise ValueError(
            f"Only {int(frequency_mask.sum())} frequency mode(s) fall between fmin={fmin:.4f} "
            f"and fmax={fmax:.2f} for {n_taxa} taxa. Raise fmax, or provide a table with more taxa."
        )

    power_values, frequency = spectral_power(clr.to_numpy(dtype=float), use_hann_window=use_hann_window)
    power = pd.DataFrame(power_values, index=clr.index, columns=frequency)

    slopes = spectral_slope(clr.to_numpy(dtype=float), fmax=fmax, use_hann_window=use_hann_window)
    slopes.index = clr.index

    diversity = alpha_diversity(relative)
    richness = diversity["Richness"].astype(int)
    compressibility = compressibility_metrics(
        clr.to_numpy(dtype=float), richness=richness.to_numpy(), fmax=fmax, use_hann_window=use_hann_window
    )
    compressibility.index = clr.index

    cumulative_values, mode_fraction, _ = cumulative_energy(clr.to_numpy(dtype=float), fmax=fmax, use_hann_window=use_hann_window)
    cumulative = pd.DataFrame(cumulative_values, index=clr.index, columns=mode_fraction)

    macro_micro = macro_micro_scores(
        clr.to_numpy(dtype=float), fmax=fmax, low_fraction=low_fraction, high_fraction=high_fraction, use_hann_window=use_hann_window
    )
    macro_micro.index = clr.index

    group = None
    if metadata is not None and group_column is not None:
        if group_column not in metadata.columns:
            raise ValueError(f"Group column '{group_column}' was not found in the metadata table.")
        group = metadata[group_column].astype(str).reindex(clr.index)
        counts = group.value_counts()
        retained = counts[counts >= min_samples_per_group].index
        kept = group[group.isin(retained)].index
        group = group.loc[kept]
        if group.nunique() < 2:
            sizes = ", ".join(f"{label}: {size}" for label, size in counts.items())
            raise ValueError(
                f"Fewer than two groups have at least {min_samples_per_group} samples "
                f"({sizes}). Lower the minimum, or group by a different column."
            )
        clr = clr.loc[kept]
        power = power.loc[kept]
        slopes = slopes.loc[kept]
        compressibility = compressibility.loc[kept]
        cumulative = cumulative.loc[kept]
        macro_micro = macro_micro.loc[kept]
        diversity = diversity.loc[kept]
        metadata = metadata.loc[kept]
        relative = relative.loc[kept]

    params = {
        "pseudocount": pseudocount,
        "fmax": fmax,
        "use_hann_window": bool(use_hann_window),
        "low_fraction": low_fraction,
        "high_fraction": high_fraction,
        "n_taxa": n_taxa,
        "max_samples": max_samples,
        "min_samples_per_group": min_samples_per_group,
    }
    params.update(orientation)

    return SpectralResult(
        abundance=relative,
        clr=clr,
        power=power,
        frequency=frequency,
        frequency_mask=frequency_mask,
        slopes=slopes,
        compressibility=compressibility,
        cumulative=cumulative,
        mode_fraction=mode_fraction,
        macro_micro=macro_micro,
        diversity=diversity,
        taxa_order=taxa_order,
        metadata=metadata,
        group=group,
        group_column=group_column,
        params=params,
        match_report=match_report,
        details=details,
    )
