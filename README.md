# Alcohol Brain Connectomes
<p align="center">
  <a href="https://creativecommons.org/licenses/by-nc/4.0/deed.en"><img src="https://img.shields.io/badge/license-CC--BY--NC%204.0-lightgrey"/></a>
  <a href="https://arxiv.org/">
  <img alt="Documentation" src="https://img.shields.io/badge/arXiv-2507.02554-b31b1b.svg">
  </a>
</p>

Code, saved outputs, and figure-generation utilities for the manuscript on
"Predicting future alcohol use from baseline brain connectomes"
- June 2024 - Present
- Manuscript in progress

## Repository layout
- `environment.yml` — conda environment for all Python runs.
- `scripts/` — data prep utilities (subject grouping, matrix flattening,
  demographics processing, alignment, permutation helpers).
- `train/` — logistic regression pipelines (combined, cross-sex, sex-specific,
  undersampled).
- `visualization_scripts/` — plotting utilities and master figure runner.
- `final_figures/` — paper-ready PNGs generated from the saved results.
- `results/` — example reports/metrics to accompany the figures.

## Setup
```bash
conda env create -f environment.yml
conda activate alcohol
# brainmontage (for brain region montages) — install if you need to regenerate those plots
git clone https://github.com/kjamison/brainmontageplot.git
cd brainmontageplot
pip install .
```

## Data (not distributed)
Raw participant-level data are not included. To re-run training you need:
- SC matrices: `data/tractography_subcortical (SC)/*baseline.mat`
- FC matrices: `data/FC/NCANDA_FC.mat`, `data/FC/NCANDA_FCgsr.mat`
- Demography ordering: `data/FC/NCANDA_demos.csv`
- Baseline labels and covariates: CSVs in `data/csv/` (e.g.,
  `cahalan_plus_drugs.csv`, `demographics_short.csv`, `clinical_short.csv`,
  `sri24_parc116_gm.csv`, `tzo116plus_yeo7xhemi.csv`)

## Workflow
1) **Group assignment (control vs heavy outcome)**  
   `python scripts/split_into_groups.py`  
   Toggle `CONTROL_ONLY` inside to switch between control-only and control+moderate cohorts.

2) **Flatten connectivity matrices**  
   `python scripts/process_conn_matrices.py`

3) **Prepare demographics**  
   `python scripts/process_demographics.py`

4) **Align modalities and split by sex**  
   `python scripts/align_training_data.py`  
   Outputs go to `data/training_data/aligned/` and `data/data_with_subject_ids/aligned/`.

5) **Train/evaluate**  
   - Combined within-sex CV: `python train/logreg_master.py`
   - Cross-sex generalization: `python train/logreg_cross_sex_repeated.py`
   - Sex-balanced/undersampled runs: `python train/logreg_undersample.py`
   - Sex-specific evaluation from combined models: `python train/logreg_sex_specific.py`

   Script-level flags (`CONTROL_ONLY`, `PERMUTE`, `UNDERSAMPLE`, `MALE`/`FEMALE`, etc.)
   control cohort and analysis settings.

6) **Generate figures**  
   From the saved results, run `python visualization_scripts/generate_all_figures.py`
   to populate `figures/final_figures/` (individual scripts are also available in that folder).

## Quick look
- Ready-to-use figures: `final_figures/*.png`
- Example report: `results/reports/sex_comparison_report_control_moderate.txt`

## Support
Issues or questions: please open a GitHub issue on this repository.
