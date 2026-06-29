from pathlib import Path
import re
import numpy as np
import pandas as pd


def normalize_taxon_name(value):
    value = str(value).strip().strip('"').strip("'").replace("|", ";")

    if value.lower() in {"nan", "none", "uncultured", "outgrouping", "incertae", ""}:
        return ""

    corrections = {
        "Allistipes": "Alistipes",
        "Klebisiella": "Klebsiella",
        "Fusobacteria": "Fusobacterium",
        "Escherichia/Shigella": "Escherichia",
        "Clostridiales": "Clostridium",
    }

    if value.startswith("sk__"):
        value = "k__" + value[4:]

    value = value.replace(" et rel.", "").replace(" et rel", "")
    value = re.sub(r"\s+cluster.*$", "", value, flags=re.I)
    value = re.sub(r"\s+group.*$", "", value, flags=re.I)
    value = re.sub(r"\s+sensu.*$", "", value, flags=re.I)
    value = value.strip()

    if ";" not in value:
        for prefix in ["k__", "p__", "c__", "o__", "f__", "g__", "s__"]:
            if value.startswith(prefix):
                value = value[len(prefix):]

    if " " in value and ";" not in value:
        value = value.split()[0]

    return corrections.get(value, value)


def collapse_to_genus(value):
    value = normalize_taxon_name(value)
    parts = [part for part in str(value).split(";") if part]
    retained = []

    for part in parts:
        if part.startswith(("s__", "t__")):
            break
        retained.append(part)
        if part.startswith("g__"):
            break

    return ";".join(retained)


def read_phylogeny_order(path, genus_level=True):
    phylogeny = pd.read_csv(path, low_memory=False)
    values = phylogeny.iloc[:, 0].astype(str)

    if genus_level:
        values = values.map(collapse_to_genus)
    else:
        values = values.map(normalize_taxon_name)

    return values.dropna().loc[lambda x: x != ""].drop_duplicates().tolist()


def read_abundance_table(path, sample_column=None, index_col=0, hdf_key=None, taxa_as_rows=False):
    path = Path(path)

    if hdf_key is not None:
        table = pd.read_hdf(path, hdf_key)
    else:
        table = pd.read_csv(path, low_memory=False, index_col=None if sample_column is not None else index_col)

    if sample_column is not None:
        if sample_column not in table.columns:
            raise ValueError(f"Sample column '{sample_column}' was not found.")
        table[sample_column] = table[sample_column].astype(str)
        table = table.set_index(sample_column)

    if taxa_as_rows:
        table = table.T

    table.index = table.index.astype(str)
    table = table.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return table


def collapse_abundance_to_genus(abundance):
    abundance = abundance.copy()
    abundance.columns = [collapse_to_genus(column) for column in abundance.columns]
    abundance = abundance.loc[:, [column != "" for column in abundance.columns]]
    abundance = abundance.T.groupby(level=0).sum().T
    return abundance


def order_taxa_by_phylogeny(abundance, phylogeny_order, min_taxa=1):
    ordered = [taxon for taxon in phylogeny_order if taxon in abundance.columns]

    if len(ordered) < min_taxa:
        raise ValueError(f"Only {len(ordered)} taxa matched the phylogenetic order.")

    return abundance.loc[:, ordered]


def align_samples(abundance, metadata):
    shared = abundance.index.intersection(metadata.index)

    if len(shared) == 0:
        raise ValueError("No overlapping samples were found.")

    return abundance.loc[shared].copy(), metadata.loc[shared].copy()


def relative_abundance(abundance):
    abundance = abundance.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    row_sums = abundance.sum(axis=1)
    abundance = abundance.loc[row_sums > 0].copy()
    row_sums = abundance.sum(axis=1)
    return abundance.div(row_sums, axis=0)


def centered_log_ratio(abundance, pseudocount=1e-9):
    abundance = relative_abundance(abundance) + pseudocount
    log_abundance = np.log(abundance)
    return log_abundance.sub(log_abundance.mean(axis=1), axis=0)


def inverse_centered_log_ratio(clr_table):
    values = np.exp(clr_table)
    return values.div(values.sum(axis=1), axis=0)
