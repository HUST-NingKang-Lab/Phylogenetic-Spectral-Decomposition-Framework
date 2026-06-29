# PhyloSpectra

PhyloSpectra is a phylogeny-aware spectral framework for microbiome community analysis. It represents a microbial community as an abundance signal ordered along a phylogenetic axis and decomposes this signal into Fourier spectral components. The framework is designed to characterize scale-dependent microbiome organization, host-associated stability transitions, perturbation response axes, spectral compressibility, and downstream applications including batch harmonization and synthetic community generation.

This repository contains two complementary layers:

```text
paper_code/        Scripts for reproducing the analyses and figures in the manuscript
src/phylospectra/  Reusable Python functions that implement the core framework
data/              Input data tables used by the analysis scripts
```

## Overview

Microbiome profiles are usually represented as high-dimensional taxon-by-sample matrices. PhyloSpectra instead treats each sample as a one-dimensional abundance signal after taxa are ordered by phylogenetic relatedness. Fourier decomposition is then used to separate broad, low-frequency phylogenetic-scale organization from fine, high-frequency fragmentation.

The main analyses in this repository support five components of the manuscript:

1. MGnify biome-level spectral organization
2. Host-associated stability transitions
3. Perturbation response axes and spectral seesaw dynamics
4. Emergent spectral compressibility
5. Applications to cohort harmonization and synthetic community generation

## Repository structure

```text
PhyloSpectra/
├── paper_code/
│   ├── 01_mgnify_spectral_organization/
│   ├── 02_host_stability_transitions/
│   ├── 03_pertubation_response_axes/
│   ├── 04_spectral_compressibility/
│   └── 05_applications/
│
├── src/
│   └── phylospectra/
│       ├── __init__.py
│       ├── io.py
│       ├── spectral.py
│       ├── response.py
│       ├── harmonization.py
│       ├── generation.py
│       ├── evaluation.py
│       └── visualization.py
│
├── data/
├── outputs/
├── figures/
├── docs/
├── .gitignore
├── README.md
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

`paper_code/` contains manuscript-specific workflows. These scripts are intended for reproducing the main results and figure panels.

`src/phylospectra/` contains reusable software components. These modules are designed to be imported by analysis scripts or reused in new microbiome spectral analyses.

`data/` stores the input abundance, metadata and phylogeny tables required by the workflows.

`outputs/` is the default location for generated tables, metrics and figures. It is recommended to keep large outputs out of version control unless they are small summary files.

## Installation

Clone the repository and install the package locally:

```bash
git clone https://github.com/<your-user-name>/PhyloSpectra.git
cd PhyloSpectra

python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
git clone https://github.com/<your-user-name>/PhyloSpectra.git
cd PhyloSpectra

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements.txt
```

## Python dependencies

The core package uses:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
```

Several figure-generation and application scripts additionally use:

```text
seaborn
scikit-bio
umap-learn
tables
```

A minimal `requirements.txt` can be:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
scikit-bio
umap-learn
tables
```

## Data organization

The analysis scripts assume the following default data layout. Paths can be changed through command-line arguments in each script.

```text
data/
├── phylogeny.csv
│
├── mgnify/
│   ├── abu.h5
│   └── metadata.csv
│
├── ibd/
│   ├── abundance.csv
│   └── metadata.csv
│
├── infant/
│   ├── abundance.csv
│   └── metadata.csv
│
├── palleja/
│   ├── abundance_phylogeny_ordered.csv
│   └── metadata.csv
│
├── etec/
│   ├── etec_abundance_genus.csv
│   └── etec_metadata.csv
│
├── okeefe/
│   ├── okeefe_dietswap_abundance.csv
│   └── okeefe_dietswap_metadata.csv
│
├── crc_7_cohorts/
│   ├── classification.txt
│   ├── d_crc1.txt
│   ├── h_crc1.txt
│   ├── d_crc2.txt
│   ├── h_crc2.txt
│   ├── ...
│   ├── d_crc7.txt
│   └── h_crc7.txt
│
└── generation/
    └── mgnify_biome_subset/
        ├── abundance.h5
        └── metadata.csv
```

The expected abundance format is generally samples by taxa, except for some legacy CRC and perturbation datasets where scripts perform the required transpose internally. The phylogeny file should contain a taxonomic path or genus-level ordering in its first column.

## Quick start

Run the MGnify spectral organization workflow:

```bash
python paper_code/01_mgnify_spectral_organization/spectral_slope.py
```

Run the host macro-micro spectral space workflow:

```bash
python paper_code/02_host_stability_transitions/host_macro_micro_spectral_space.py
```

Run the perturbation response-axis workflow:

```bash
python paper_code/03_pertubation_response_axes/spectral_response_axis.py
```

Run the spectral compressibility workflow:

```bash
python paper_code/04_spectral_compressibility/spectral_compressibility_metrics.py
```

Run the low-order biome classification workflow:

```bash
python paper_code/04_spectral_compressibility/low_order_biome_classification.py
```

Run the CRC harmonization workflow:

```bash
python paper_code/05_applications/crc_fourier_harmonization.py
```

Run the synthetic community generation workflow:

```bash
python paper_code/05_applications/spectral_community_generation.py
```

## Manuscript analysis map

### 1. MGnify spectral organization

Folder:

```text
paper_code/01_mgnify_spectral_organization/
```

Suggested files:

```text
spectral_slope.py
random_order_permutation.py
dominant_peak.py
```

Purpose:

- Compute biome-level phylogenetic spectral slopes.
- Test whether the observed spectral organization depends on the true phylogenetic order.
- Estimate dominant spectral peak frequencies as a supplementary analysis.

Typical outputs:

```text
outputs/01_mgnify_spectral_organization/
├── spectral_slope_per_sample.csv
├── group_spectra_summary.csv
├── random_order_permutation_summary.csv
├── dominant_peak_per_sample.csv
└── figures
```

### 2. Host-associated stability transitions

Folder:

```text
paper_code/02_host_stability_transitions/
```

Suggested files:

```text
ibd_spectral_slope.py
antibiotic_perturbation_spectral_slope.py
infant_maturation_spectral_slope.py
host_macro_micro_spectral_space.py
```

Purpose:

- Compare spectral slopes between healthy and disease-associated states.
- Quantify spectral shifts during infant gut maturation.
- Characterize perturbation and recovery following antibiotic exposure.
- Combine IBD, infant maturation and antibiotic perturbation into a macro-micro spectral space.

Typical outputs:

```text
outputs/02_host_stability_transitions/
├── ibd_spectral_slope/
├── infant_maturation_spectral_slope/
├── antibiotic_perturbation_spectral_slope/
└── host_macro_micro_spectral_space/
```

### 3. Perturbation response axes

Folder:

```text
paper_code/03_pertubation_response_axes/
```

Suggested files:

```text
spectral_response_axis.py
control_axis_specificity.py
```

Purpose:

- Construct dataset-specific spectral response axes for ETEC and O'Keefe datasets.
- Project response axes back to taxon space using inverse Fourier transformation.
- Extract positive and negative taxonomic poles.
- Validate reciprocal pole-balance responses.
- Test whether the learned response axes are more biologically specific than residual control axes.

Typical outputs:

```text
outputs/03_spectral_seesaw_response/
├── spectral_response_axis/
│   ├── spectral_response_axis_integrated.png
│   ├── spectral_response_axis_integrated.pdf
│   ├── etec_pole_validation_summary.csv
│   ├── okeefe_pole_validation_summary.csv
│   ├── etec_axis_frequency_profile.csv
│   └── okeefe_axis_frequency_profile.csv
│
└── control_axis_specificity/
    ├── control_axis_specificity.png
    ├── control_axis_specificity.pdf
    └── control_axis_specificity_summary.csv
```

### 4. Spectral compressibility

Folder:

```text
paper_code/04_spectral_compressibility/
```

Suggested files:

```text
spectral_compressibility_metrics.py
low_order_biome_classification.py
```

Purpose:

- Compute C50, C80, C90, low-order energy concentration, spectral entropy and effective spectral dimension.
- Test whether richer communities require fewer low-order modes to explain spectral energy.
- Evaluate whether a small fraction of low-frequency Fourier modes preserves biome identity.
- Compare low-order spectral features against alpha-diversity and full-taxon baselines.

Typical outputs:

```text
outputs/04_spectral_compressibility/
├── spectral_compressibility_metrics/
│   ├── spectral_compressibility_metrics.csv
│   ├── spectral_compressibility_metrics.png
│   ├── spectral_compressibility_metrics.pdf
│   ├── low_order_energy_concentration.png
│   └── richness_tier_difference_curves.png
│
└── low_order_biome_classification/
    ├── low_order_classification_benchmark.csv
    ├── macro_roc_curves.csv
    ├── macro_roc_aucs.csv
    ├── per_biome_roc_curves.csv
    ├── per_biome_roc_aucs.csv
    ├── low_order_biome_classification.png
    └── low_order_biome_classification.pdf
```

### 5. Applications

Folder:

```text
paper_code/05_applications/
```

Suggested files:

```text
crc_fourier_harmonization.py
spectral_community_generation.py
```

Purpose:

- Apply Fourier-domain harmonization to multi-cohort CRC microbiome datasets.
- Reduce low-frequency batch-associated trends while preserving disease-relevant signal.
- Generate synthetic microbiome profiles by perturbing Fourier-domain components within biome classes.

Typical outputs:

```text
outputs/05_harmonization/crc_fourier_harmonization/
├── crc_pcoa_harmonization.png
├── crc_pcoa_harmonization.pdf
├── crc_batch_silhouette_comparison.png
├── crc_lobo_auc_comparison.png
├── crc_cutoff_sensitivity_auc.png
├── harmonization_summary.csv
├── corrected_relative_abundance.csv
└── corrected_clr_abundance.csv

outputs/06_generation/spectral_community_generation/
├── synthetic_profiles_embedding_grid.png
├── synthetic_profiles_embedding_grid.pdf
├── generation_summary.csv
├── real_profiles_subset.csv
└── synthetic_profiles_*x.csv
```

## Software package modules

The `src/phylospectra/` folder contains reusable components.

### `io.py`

Data input and preprocessing utilities:

```python
from phylospectra.io import (
    normalize_taxon_name,
    collapse_to_genus,
    read_phylogeny_order,
    read_abundance_table,
    align_samples,
    relative_abundance,
    centered_log_ratio,
    inverse_centered_log_ratio,
)
```

### `spectral.py`

Fourier transform and spectral metric utilities:

```python
from phylospectra.spectral import (
    fourier_coefficients,
    spectral_power,
    spectral_slope,
    cumulative_energy,
    compressibility_metrics,
    macro_micro_scores,
)
```

### `response.py`

Perturbation response-axis utilities:

```python
from phylospectra.response import (
    baseline_deltas,
    mean_by_subject,
    response_axis,
    project_on_axis,
    inverse_project_axis,
    extract_axis_poles,
    pole_balance,
    residual_axes,
)
```

### `harmonization.py`

Fourier-domain batch harmonization:

```python
from phylospectra.harmonization import (
    lowpass_signal,
    fourier_batch_harmonization,
)
```

### `generation.py`

Synthetic community generation:

```python
from phylospectra.generation import (
    enforce_simplex,
    generate_from_class,
    generate_by_group,
)
```

### `evaluation.py`

Evaluation, classification and validation utilities:

```python
from phylospectra.evaluation import (
    alpha_diversity,
    batch_r2_from_distance,
    batch_silhouette_values,
    leave_one_group_auc,
    roc_tables,
)
```

### `visualization.py`

General plotting and embedding helpers:

```python
from phylospectra.visualization import (
    configure_matplotlib,
    style_axis,
    embed_profiles,
    save_figure,
)
```

## Minimal example

```python
import pandas as pd
from phylospectra.io import read_phylogeny_order, order_taxa_by_phylogeny, centered_log_ratio
from phylospectra.spectral import spectral_slope, compressibility_metrics

abundance = pd.read_csv("data_demo/demo_abundance.csv", index_col=0)
phylogeny_order = read_phylogeny_order("data_demo/demo_phylogeny_order.csv")

abundance = order_taxa_by_phylogeny(abundance, phylogeny_order)
clr = centered_log_ratio(abundance)

slope = spectral_slope(clr, fmax=0.20)
metrics = compressibility_metrics(clr, fmax=0.20)

print(slope.head())
print(metrics.head())
```

## Command-line usage examples

Most scripts support command-line arguments. For example:

```bash
python paper_code/04_spectral_compressibility/spectral_compressibility_metrics.py \
  --abundance data/mgnify/abu.h5 \
  --metadata data/mgnify/metadata.csv \
  --phylogeny data/phylogeny.csv \
  --output-dir outputs/04_spectral_compressibility/spectral_compressibility_metrics
```

```bash
python paper_code/05_applications/crc_fourier_harmonization.py \
  --data-dir data/crc_7_cohorts \
  --taxonomy data/crc_7_cohorts/classification.txt \
  --output-dir outputs/05_harmonization/crc_fourier_harmonization
```

```bash
python paper_code/05_applications/spectral_community_generation.py \
  --abundance data/generation/mgnify_biome_subset/abundance.h5 \
  --metadata data/generation/mgnify_biome_subset/metadata.csv \
  --class-col class \
  --output-dir outputs/06_generation/spectral_community_generation
```

## Key parameters

### Spectral decomposition

```text
--pseudocount
```

Small value added before log transformation.

```text
--fmax
```

Maximum normalized phylogenetic frequency retained for spectral slope and spectral metrics.

```text
--use-hann-window / --no-hann-window
```

Whether to apply a Hann window before Fourier decomposition.

### Compressibility

```text
--energy-threshold
```

Energy threshold used for C80-like compressibility metrics. The default is 0.80.

```text
--low-order-fractions
```

Fractions of the lowest-frequency Fourier modes used in low-order classification.

```text
--focal-low-order-fraction
```

Main low-order fraction highlighted in classification and ROC plots. The default is 0.005, corresponding to 0.5% of Fourier modes.

### Harmonization

```text
--cutoff-frequency
```

Fourier cutoff frequency used to estimate low-frequency batch-associated trends.

```text
--correction-strength
```

Scaling factor applied to the estimated batch trend before subtraction.

```text
--per-condition / --no-per-condition
```

Whether batch correction is estimated separately within each biological condition.

## Outputs and reproducibility

Each analysis script writes results to `outputs/` by default. Typical outputs include:

```text
*.csv   numeric results and summary tables
*.png   raster figures
*.pdf   vector figures
```

The scripts are written so that paths can be changed using command-line arguments. Randomized steps use explicit random seeds where applicable.

## Notes on data size

Some public microbiome datasets can be large. If this repository is shared publicly, consider excluding large raw data files from Git tracking and placing them in an external archive such as Zenodo, Figshare, OSF or a project data repository. In that case, keep small demo files in `data_demo/` and describe the full data download process in `docs/data_availability.md`.

## Suggested citation

If you use this repository, please cite the associated manuscript:

```text
Phylogenetic spectral compressibility of microbiome organization.
```

A full citation will be added after publication.

## License

This repository is released under the license specified in `LICENSE`.

## Contact

For questions, issues or suggestions, please open an issue in the GitHub repository.
