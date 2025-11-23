# Figure Generation Guide

## Overview

This project includes 6 scripts for generating publication-quality figures, plus a master script that runs them all.

## Quick Start

To generate all final figures at once:

```
python visualization_scripts/generate_all_figures.py
```

All figures will be saved to `figures/final_figures/`

## Individual Scripts

Each script can also be run individually:

### 1. Combined Violin Plots (`create_combined_violin.py`)
- **Purpose**: Model performance metrics across different modalities
- **Generates**: 2x3 grid with overall and sex-stratified violin plots
- **Output**: `figures/combined_violin/control_moderate/combined_figure_control_moderate.png`

```bash
python scripts/create_combined_violin.py
```

### 2. Edges Heatmap (`edges_heatmap.py`)
- **Purpose**: Network connectivity patterns (SC and FC)
- **Generates**: 2x2 square format heatmaps (positive/negative coefficients)
- **Output**: 3 files for all/male/female subjects
  - `figures/edges_heatmap/square_format_network_means_control_moderate.png`
  - `figures/edges_heatmap_male/square_format_network_means_control_moderate_male.png`
  - `figures/edges_heatmap_female/square_format_network_means_control_moderate_female.png`

```bash
python scripts/edges_heatmap.py
```

### 3. Cross-Sex Comparison Plots (`crosssex_comparison_plots.py`)
- **Purpose**: Within-sex vs cross-sex generalization performance
- **Generates**:
  - Full comparison plot (4 conditions)
  - Cross-sex only plot (2 directions)
- **Output**:
  - `figures/crosssex_comparison/control_moderate/combined_full_comparison_control_moderate.png`
  - `figures/crosssex_comparison/control_moderate/combined_crosssex_only_control_moderate.png`

```bash
python scripts/crosssex_comparison_plots.py
```

### 4. Brain Region Visualization (`visualize_brain_regions.py`)
- **Purpose**: Spatial distribution of Haufe coefficients across brain regions
- **Generates**: 2x2 brain montages (SC/FC × positive/negative)
- **Output**: 3 files for all/male/female subjects
  - `figures/brain_regions_2/square_format_brain_regions_control_moderate.png`
  - `figures/brain_regions_2/square_format_brain_regions_control_moderate_male.png`
  - `figures/brain_regions_2/square_format_brain_regions_control_moderate_female.png`

```bash
python scripts/visualize_brain_regions.py
```

### 5. Yeo Network Assignments (`yeo_functional_assignments.py`)
- **Purpose**: Network-level influences across Yeo functional networks
- **Generates**: 3-panel bar plot (combined/male/female)
- **Output**: `figures/yeo_network_barplots/yeo_network_influences_combined_control_moderate.png`

```bash
python scripts/yeo_functional_assignments.py
```

### 6. ROC and PR Curves Grid (`roc_curves.py`)
- **Purpose**: Model discrimination and precision-recall performance visualization
- **Generates**: 2x3 grid with ROC curves (top row) and PR curves (bottom row) for combined/male/female
- **Output**: `figures/roc_pr_grid/roc_pr_grid_control_moderate.png`
- **Function**: `plot_roc_pr_combined_grid(['SC', 'FC', 'demos', 'simple_ensemble'], control_only=False, save_fig=True)`

```bash
python scripts/roc_curves.py
```

## Key Features

### Dynamic Range Adjustment
All scripts now automatically calculate and use dynamic axis ranges/colormaps based on the actual data:

- **Violin plots**: Y-axis dynamically adjusts to data range (all 3 metrics share the same range)
- **Heatmaps**: Color scale is symmetric around 0 and spans actual min/max values **across all/male/female**
- **Brain montages**: Color limits adapt to coefficient distributions **across all/male/female**
- **Bar plots**: Y-axis adjusts to accommodate all network values **across combined/male/female**

### Consistency Within and Across Sex Stratifications
- **Within each script**: All plots use **identical ranges** for proper comparison
  - In `create_combined_violin.py`: All 3 metrics (Balanced Accuracy, ROC AUC, PR AUC) share the same y-axis range
  - In `edges_heatmap.py`: All 4 heatmaps (SC/FC × pos/neg) share the same color scale

- **Across sex stratifications**: Figures 2a/2b/2c and 4a/4b/4c share ranges
  - **Edges heatmaps (2a/2b/2c)**: All three (all/male/female) use the same colormap range, calculated from global min/max across all conditions
  - **Brain montages (4a/4b/4c)**: All three (all/male/female) use the same color limits, calculated from global min/max across all conditions
  - **Yeo bar plots (5)**: All three panels (combined/male/female) share the same y-axis range

This ensures that visual comparisons between sex-stratified figures are meaningful and quantitatively accurate.

## Configuration

To change between control-only and control+moderate datasets, edit the `CONTROL_ONLY` variable in each script or in `generate_all_figures.py`:

```python
CONTROL_ONLY = False  # Use control+moderate (default)
# CONTROL_ONLY = True  # Use control-only
```

## Output Directory Structure

```
figures/
├── final_figures/                      # Master output directory
│   ├── 1_combined_violin_*.png         # * will be either control or control_moderate
│   ├── 2a_edges_heatmap_all_*.png
│   ├── 2b_edges_heatmap_male_*.png
│   ├── 2c_edges_heatmap_female_*.png
│   ├── 3a_crosssex_full_comparison_*.png
│   ├── 3b_crosssex_only_*.png
│   ├── 4a_brain_regions_all_*.png
│   ├── 4b_brain_regions_male_*.png
│   ├── 4c_brain_regions_female_*.png
│   ├── 5_yeo_network_influences_*.png
│   └── 6_roc_pr_grid_*.png
├── combined_violin/
├── edges_heatmap/
├── crosssex_comparison/
├── brain_regions_2/
├── yeo_network_barplots/
└── roc_pr_grid/
```

## Dependencies

All scripts require:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn (for ROC and PR curve calculations)

Additional dependencies for specific scripts:
- PIL (for brain montages in visualize_brain_regions.py)
- `brainmontage` (for visualize_brain_regions.py)
- Custom modules: `get_haufe_coefs`, `sig_coefs`, `edges_heatmap`

## Notes

- Figures are generated at 300 DPI for publication quality
- All scripts save figures automatically (SAVE_FIG = True by default)
- The master script copies all figures to `figures/final_figures/` with numbered prefixes for easy identification
- Temporary files (for brain montages) are automatically cleaned up after generation
