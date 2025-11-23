"""
Master script to generate all final figures for the alcohol prediction project.

This script runs all 6 figure generation scripts in sequence:
1. create_combined_violin.py - Combined violin plots for model performance metrics
2. edges_heatmap.py - Square format heatmaps showing network connectivity
3. crosssex_comparison_plots.py - Cross-sex generalization comparison plots
4. visualize_brain_regions.py - Brain region montages showing coefficient distributions
5. yeo_functional_assignments.py - Yeo network functional assignment bar plots
6. roc_curves.py - ROC and Precision-Recall curves in 2x3 grid format

All figures will be saved to final_figures/ directory.
"""

import os
import sys
import shutil
from pathlib import Path

# Add the scripts directory to the path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

# Import the generation functions from each script
from create_combined_violin import create_combined_figure
from edges_heatmap import create_square_format_heatmap, get_global_heatmap_range
from crosssex_comparison_plots import create_combined_full_comparison_plot, create_combined_crosssex_only_plot
from visualize_brain_regions import create_square_format_brain_montage, get_global_brain_region_range
from yeo_functional_assignments import create_combined_yeo_barplot
from roc_curves import plot_roc_pr_combined_grid

def main():
    """
    Main function to generate all final figures.
    """
    print("=" * 80)
    print("GENERATING ALL FINAL FIGURES")
    print("=" * 80)

    # Configuration
    CONTROL_ONLY = False
    file_name = 'control' if CONTROL_ONLY else 'control_moderate'

    # Create final figures directory
    final_figures_dir = Path('figures/final_figures')
    final_figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nUsing data: {file_name}")
    print(f"Output directory: {final_figures_dir}")
    print("=" * 80)

    # ========================================================================
    # 1. Create Combined Violin Plots
    # ========================================================================
    print("\n[1/6] Generating combined violin plots...")
    print("-" * 80)
    try:
        create_combined_figure(control_only=CONTROL_ONLY, save_fig=True)

        # Copy to final_figures
        src = f'figures/combined_violin/{file_name}/combined_figure_{file_name}.png'
        dst = final_figures_dir / f'1_combined_violin_{file_name}.png'
        shutil.copy2(src, dst)
        print(f"✓ Saved to: {dst}")
    except Exception as e:
        print(f"✗ Error generating combined violin plots: {e}")

    # ========================================================================
    # 2. Create Edges Heatmaps
    # ========================================================================
    print("\n[2/6] Generating edges heatmaps...")
    print("-" * 80)
    try:
        # Calculate global color range across all sex conditions
        print("  - Calculating global color range across all/male/female...")
        vmin, vmax = get_global_heatmap_range(control_only=CONTROL_ONLY)
        print(f"    Global color range: [{vmin:.4f}, {vmax:.4f}]")

        # Generate for all subjects
        print("  - All subjects...")
        create_square_format_heatmap(control_only=CONTROL_ONLY, male=False, female=False, save_fig=True, vmin=vmin, vmax=vmax)
        src = f'figures/edges_heatmap/square_format_network_means_{file_name}.png'
        dst = final_figures_dir / f'2a_edges_heatmap_all_{file_name}.png'
        shutil.copy2(src, dst)
        print(f"    ✓ Saved to: {dst}")

        # Generate for male only
        print("  - Male subjects...")
        create_square_format_heatmap(control_only=CONTROL_ONLY, male=True, female=False, save_fig=True, vmin=vmin, vmax=vmax)
        src = f'figures/edges_heatmap_male/square_format_network_means_{file_name}_male.png'
        dst = final_figures_dir / f'2b_edges_heatmap_male_{file_name}.png'
        shutil.copy2(src, dst)
        print(f"    ✓ Saved to: {dst}")

        # Generate for female only
        print("  - Female subjects...")
        create_square_format_heatmap(control_only=CONTROL_ONLY, male=False, female=True, save_fig=True, vmin=vmin, vmax=vmax)
        src = f'figures/edges_heatmap_female/square_format_network_means_{file_name}_female.png'
        dst = final_figures_dir / f'2c_edges_heatmap_female_{file_name}.png'
        shutil.copy2(src, dst)
        print(f"    ✓ Saved to: {dst}")
    except Exception as e:
        print(f"✗ Error generating edges heatmaps: {e}")

    # ========================================================================
    # 3. Create Cross-Sex Comparison Plots
    # ========================================================================
    print("\n[3/6] Generating cross-sex comparison plots...")
    print("-" * 80)
    try:
        # Full comparison (within-sex vs cross-sex)
        print("  - Full comparison plot...")
        create_combined_full_comparison_plot(control_only=CONTROL_ONLY, save_fig=True)
        src = f'figures/crosssex_comparison/{file_name}/combined_full_comparison_{file_name}.png'
        dst = final_figures_dir / f'3a_crosssex_full_comparison_{file_name}.png'
        shutil.copy2(src, dst)
        print(f"    ✓ Saved to: {dst}")

        # Cross-sex only
        print("  - Cross-sex only plot...")
        create_combined_crosssex_only_plot(control_only=CONTROL_ONLY, save_fig=True)
        src = f'figures/crosssex_comparison/{file_name}/combined_crosssex_only_{file_name}.png'
        dst = final_figures_dir / f'3b_crosssex_only_{file_name}.png'
        shutil.copy2(src, dst)
        print(f"    ✓ Saved to: {dst}")
    except Exception as e:
        print(f"✗ Error generating cross-sex comparison plots: {e}")

    # ========================================================================
    # 4. Create Brain Region Montages
    # ========================================================================
    print("\n[4/6] Generating brain region montages...")
    print("-" * 80)
    try:
        # Calculate global color limits across all sex conditions
        print("  - Calculating global color limits across all/male/female...")
        clim = get_global_brain_region_range(control_only=CONTROL_ONLY)
        print(f"    Global color limits: [{clim[0]:.4f}, {clim[1]:.4f}]")

        # Generate for all subjects
        print("  - All subjects...")
        create_square_format_brain_montage(control_only=CONTROL_ONLY, male=False, female=False, clim=clim)
        src = f'figures/brain_regions_2/square_format_brain_regions_{file_name}.png'
        dst = final_figures_dir / f'4a_brain_regions_all_{file_name}.png'
        shutil.copy2(src, dst)
        print(f"    ✓ Saved to: {dst}")

        # Generate for male only
        print("  - Male subjects...")
        create_square_format_brain_montage(control_only=CONTROL_ONLY, male=True, female=False, clim=clim)
        src = f'figures/brain_regions_2/square_format_brain_regions_{file_name}_male.png'
        dst = final_figures_dir / f'4b_brain_regions_male_{file_name}.png'
        shutil.copy2(src, dst)
        print(f"    ✓ Saved to: {dst}")

        # Generate for female only
        print("  - Female subjects...")
        create_square_format_brain_montage(control_only=CONTROL_ONLY, male=False, female=True, clim=clim)
        src = f'figures/brain_regions_2/square_format_brain_regions_{file_name}_female.png'
        dst = final_figures_dir / f'4c_brain_regions_female_{file_name}.png'
        shutil.copy2(src, dst)
        print(f"    ✓ Saved to: {dst}")
    except Exception as e:
        print(f"✗ Error generating brain region montages: {e}")

    # ========================================================================
    # 5. Create Yeo Network Bar Plots
    # ========================================================================
    print("\n[5/6] Generating Yeo network bar plots...")
    print("-" * 80)
    try:
        create_combined_yeo_barplot(file_name)
        src = f'figures/yeo_network_barplots/yeo_network_influences_combined_{file_name}.png'
        dst = final_figures_dir / f'5_yeo_network_influences_{file_name}.png'
        shutil.copy2(src, dst)
        print(f"✓ Saved to: {dst}")
    except Exception as e:
        print(f"✗ Error generating Yeo network bar plots: {e}")

    # ========================================================================
    # 6. Create ROC and PR Curves Grid
    # ========================================================================
    print("\n[6/6] Generating ROC and PR curves grid...")
    print("-" * 80)
    try:
        # Make sure the directory exists
        Path('figures/roc_pr_grid').mkdir(parents=True, exist_ok=True)

        plot_roc_pr_combined_grid(['SC', 'FC', 'demos', 'simple_ensemble'], control_only=CONTROL_ONLY, save_fig=True)
        src = f'figures/roc_pr_grid/roc_pr_grid_{file_name}.png'
        dst = final_figures_dir / f'6_roc_pr_grid_{file_name}.png'
        shutil.copy2(src, dst)
        print(f"✓ Saved to: {dst}")
    except Exception as e:
        print(f"✗ Error generating ROC and PR curves grid: {e}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nAll final figures have been saved to: {final_figures_dir.resolve()}")
    print("\nGenerated figures:")
    print("  1. Combined violin plots (model performance metrics)")
    print("  2. Edges heatmaps (network connectivity) - all/male/female")
    print("  3. Cross-sex comparison plots - full comparison & cross-sex only")
    print("  4. Brain region montages - all/male/female")
    print("  5. Yeo network bar plots (functional assignments)")
    print("  6. ROC and PR curves grid (2x3 grid: combined/male/female)")
    print("=" * 80)


if __name__ == "__main__":
    main()
