<div align="center">

# 🧬 Phylogenetic Spectral Decomposition Framework

### A phylogeny-aware Fourier framework for scale-resolved microbiome community analysis

**PhyloSpectra** represents microbiomes as phylogeny-ordered abundance signals and decomposes them into spectral modes to reveal stability, perturbation dynamics, spectral compressibility, batch harmonization and synthetic community generation.

<br>

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research%20code-7A3446?style=flat-square)
![Topic](https://img.shields.io/badge/topic-microbiome%20spectral%20analysis-44757A?style=flat-square)
![Package](https://img.shields.io/badge/package-phylospectra-B7B5A0?style=flat-square)

</div>

---
<img width="1430" height="1060" alt="image" src="https://github.com/user-attachments/assets/85b881f3-34e4-4997-8509-b3113aeec219" />

## ✨ Overview

Microbiome profiles are usually represented as high-dimensional taxon-by-sample matrices. This repository implements a phylogeny-aware spectral view: each microbial community is treated as a one-dimensional abundance signal after taxa are ordered along a phylogenetic axis. Fourier decomposition is then used to separate broad, low-frequency phylogenetic-scale organization from fine, high-frequency fragmentation.

This repository contains two complementary layers:

```text
paper_code/        Manuscript analysis scripts and figure-reproduction workflows
src/phylospectra/  Reusable Python package functions
data/              Input abundance, metadata and phylogeny tables
outputs/           Default output directory for generated tables and figures
```
---

## 🧠 Core concept

```text
microbiome abundance table
        ↓
taxa ordered by phylogeny
        ↓
phylogeny-ordered abundance signal
        ↓
Fourier spectral decomposition
        ↓
low-frequency organization / high-frequency fragmentation
```

| Spectral quantity | Interpretation |
|---|---|
| Low-frequency modes | Broad phylogenetic-scale organization |
| High-frequency modes | Fine-scale taxonomic fragmentation |
| Spectral amplitude | Contribution of each frequency mode |
| Spectral phase | Positional or directional information along the phylogenetic axis |
| Inverse transform | Projection from spectral space back to taxa |

---

## 🔬 Main analyses

| Folder | Main question | Main outputs |
|---|---|---|
| `01_mgnify_spectral_organization` | Do environments differ in phylogenetic spectral structure? | Spectral slope, dominant peak, permutation tests |
| `02_host_stability_transitions` | Do stable and unstable host microbiomes occupy different spectral regimes? | Host macro-micro spectral space |
| `03_pertubation_response_axes` | Do perturbations follow directional spectral response axes? | Spectral seesaw axes, microbial poles, control axes |
| `04_spectral_compressibility` | Are microbial communities spectrally compressible? | C80, effective spectral dimension, low-order classification |
| `05_applications` | Can spectral representations support applications? | CRC harmonization, synthetic generation |

---

## ⚙️ Installation

Clone this repository:

```bash
git clone https://github.com/<your-github-name>/Phylogenetic-Spectral-Decomposition-Framework.git
cd Phylogenetic-Spectral-Decomposition-Framework
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```
---

## 📦 Dependencies

Core dependencies:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
```
---

## 🧩 Manuscript-to-code map

### 🟦 01. MGnify spectral organization

```text
paper_code/01_mgnify_spectral_organization/
```

| Script | Purpose |
|---|---|
| `spectral_slope.py` | Computes sample-level and biome-level phylogenetic spectral slopes |
| `random_order_permutation.py` | Tests whether spectral organization depends on true phylogenetic ordering |
| `dominant_peak.py` | Estimates dominant spectral peak frequencies as a supplementary analysis |

Default output:

```text
outputs/01_mgnify_spectral_organization/
```

---

### 🟩 02. Host stability transitions

```text
paper_code/02_host_stability_transitions/
```

| Script | Purpose |
|---|---|
| `ibd_spectral_slope.py` | Compares spectral slopes between healthy and IBD-associated microbiomes |
| `antibiotic_perturbation_spectral_slope.py` | Quantifies spectral changes during antibiotic perturbation and recovery |
| `infant_maturation_spectral_slope.py` | Evaluates spectral shifts during infant gut maturation |
| `host_macro_micro_spectral_space.py` | Builds the low-frequency macro-organization vs high-frequency micro-fragmentation space |

Default output:

```text
outputs/02_host_stability_transitions/
```

---

### 🟨 03. Perturbation response axes

```text
paper_code/03_pertubation_response_axes/
```

| Script | Purpose |
|---|---|
| `spectral_response_axis.py` | Constructs ETEC and O'Keefe dataset-specific spectral response axes |
| `control_axis_specificity.py` | Tests whether residual control axes reproduce the same biological trajectories |

Default output:

```text
outputs/03_spectral_seesaw_response/
```

Key outputs:

```text
spectral_response_axis_integrated.png
spectral_response_axis_integrated.pdf
control_axis_specificity.png
control_axis_specificity.pdf
```

---

### 🟥 04. Spectral compressibility

```text
paper_code/04_spectral_compressibility/
```

| Script | Purpose |
|---|---|
| `spectral_compressibility_metrics.py` | Computes C50, C80, C90, E10, E20, spectral entropy and effective spectral dimension |
| `low_order_biome_classification.py` | Tests whether 0.5% low-order Fourier modes preserve biome identity |

Default output:

```text
outputs/04_spectral_compressibility/
```

Key outputs:

```text
spectral_compressibility_metrics.csv
low_order_classification_benchmark.csv
macro_roc_curves.csv
macro_roc_aucs.csv
per_biome_roc_curves.csv
per_biome_roc_aucs.csv
```

---

### 🟪 05. Applications

```text
paper_code/05_applications/
```

| Script | Purpose |
|---|---|
| `crc_fourier_harmonization.py` | Applies Fourier-domain harmonization to multi-cohort CRC profiles |
| `spectral_community_generation.py` | Generates synthetic microbiome profiles by perturbing Fourier-domain components |

Default outputs:

```text
outputs/05_harmonization/crc_fourier_harmonization/
outputs/06_generation/spectral_community_generation/
```

---

## 🛠️ Reusable Python package

The reusable package is located in:

```text
src/phylospectra/
```

Import examples:

```python
from phylospectra.io import centered_log_ratio, read_phylogeny_order
from phylospectra.spectral import spectral_slope, compressibility_metrics
from phylospectra.response import response_axis, inverse_project_axis, pole_balance
from phylospectra.harmonization import fourier_batch_harmonization
from phylospectra.generation import generate_by_group
```

### `io.py`

Input, taxon processing and normalization utilities:

```text
normalize_taxon_name
collapse_to_genus
read_phylogeny_order
read_abundance_table
align_samples
relative_abundance
centered_log_ratio
inverse_centered_log_ratio
```

### `spectral.py`

Fourier decomposition and spectral metrics:

```text
fourier_coefficients
spectral_power
spectral_slope
cumulative_energy
compressibility_metrics
macro_micro_scores
```

### `response.py`

Perturbation response-axis utilities:

```text
baseline_deltas
mean_by_subject
response_axis
project_on_axis
inverse_project_axis
extract_axis_poles
pole_balance
residual_axes
```

### `harmonization.py`

Fourier-domain harmonization:

```text
lowpass_signal
fourier_batch_harmonization
```

### `generation.py`

Spectral synthetic community generation:

```text
enforce_simplex
generate_from_class
generate_by_group
```

### `evaluation.py`

Evaluation and machine-learning utilities:

```text
alpha_diversity
batch_r2_from_distance
batch_silhouette_values
leave_one_group_auc
roc_tables
paired_or_unpaired_p
```

### `visualization.py`

General plotting helpers:

```text
configure_matplotlib
style_axis
embed_profiles
save_figure
```

---

## 🎚️ Key parameters

| Parameter | Meaning |
|---|---|
| `--pseudocount` | Small value added before log transformation |
| `--fmax` | Maximum normalized phylogenetic frequency retained |
| `--use-hann-window` | Apply Hann window before Fourier decomposition |
| `--energy-threshold` | Energy threshold for C80-like compressibility metrics |
| `--low-order-fractions` | Fractions of lowest-frequency Fourier modes used for classification |
| `--focal-low-order-fraction` | Main low-order fraction highlighted in classification plots |
| `--cutoff-frequency` | Fourier cutoff used to estimate batch-associated trends |
| `--correction-strength` | Scaling factor applied to the estimated batch trend |
| `--per-condition` | Estimate batch trends separately within each biological condition |

---

## 📤 Outputs

Each script writes results to `outputs/` by default.

```text
*.csv   numeric tables and summary results
*.png   raster figures
*.pdf   vector figures
```
---

## ⚠️ Notes

- `paper_code/` is designed for manuscript reproduction.
- `src/phylospectra/` is designed for reusable method development.
- The GitHub repository name contains hyphens, but the Python package name does not.

---

## 📄 License

This repository is distributed under the license specified in:

```text
LICENSE
```

---

## Maintainers

| Name | Email | Organization |
|---|---|---|
| Yuli Zhang | yulizhang@hust.edu.cn | PhD student, School of Life Science and Technology, Huazhong University of Science & Technology |
| Haohong Zhang | haohongzh@gmail.com | PhD student, School of Life Science and Technology, Huazhong University of Science & Technology |
| Kang Ning | ningkang@hust.edu.cn | Professor, School of Life Science and Technology, Huazhong University of Science & Technology |


