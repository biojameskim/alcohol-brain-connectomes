import numpy as np
import pandas as pd
from edges_heatmap import upper_tri_to_matrix
from brainmontage import create_montage_figure
from get_haufe_coefs import get_haufe_coefs
from sig_coefs import get_sig_indices

def get_coefs_matrices(control_only, male, female):
    """
    [get_coefs_matrices] loads the Haufe coefficients for SC, FC, and FCgsr matrices, calculates the average coefficients, and converts the upper triangular coefficients to a square matrix
    """
    if control_only:
        file_name = 'control'
    else:
        file_name = 'control_moderate'
    
    if male:
        sex = '_male'
    elif female:
        sex = '_female'
    else:
        sex = ''

    # Load the Haufe coefficients
    SC_coefficients = get_haufe_coefs(matrix_type='SC', file_name=file_name, sex=sex)
    FC_coefficients = get_haufe_coefs(matrix_type='FC', file_name=file_name, sex=sex)
    FCgsr_coefficients = get_haufe_coefs(matrix_type='FCgsr', file_name=file_name, sex=sex)

    # Find indices where the coefficients are not significant (failed to reject --> False)
    SC_reject, FC_reject, FCgsr_reject = get_sig_indices(control_only=control_only, male=male, female=female, p_value_threshold=0.05)
    SC_false_indices = np.where(SC_reject == False)
    FC_false_indices = np.where(FC_reject == False)
    FCgsr_false_indices = np.where(FCgsr_reject == False)
    # Set non-significant coefficients to 0
    SC_coefficients[SC_false_indices] = 0
    FC_coefficients[FC_false_indices] = 0
    FCgsr_coefficients[FCgsr_false_indices] = 0

    # Convert the upper triangular coefficients to a square matrix
    SC_coefs_matrix = upper_tri_to_matrix(SC_coefficients, 90)
    FC_coefs_matrix = upper_tri_to_matrix(FC_coefficients, 109)
    FCgsr_coefs_matrix = upper_tri_to_matrix(FCgsr_coefficients, 109)

    return SC_coefs_matrix, FC_coefs_matrix, FCgsr_coefs_matrix

def get_aal_roi_vals(SC_coefs_matrix, FC_coefs_matrix, FCgsr_coefs_matrix):
    """
    [get_aal_roi_vals] calculates the average positive and negative coefficients for each AAL region and returns a dictionary of 1x116 aal vectors for {SC, FC, FCgsr} x {pos, neg} 
    """
    # Initialize with nan because any ROIs with "nan" will not be displayed
    SC_pos_coefs = np.full((116,), np.nan) 
    SC_neg_coefs = np.full((116,), np.nan)
    FC_pos_coefs = np.full((116,), np.nan) 
    FC_neg_coefs = np.full((116,), np.nan)
    FCgsr_pos_coefs = np.full((116,), np.nan) 
    FCgsr_neg_coefs = np.full((116,), np.nan)

    # SC - Cerebellar regions and Vermis set to nan (regions 91-116)
    for roi_idx in range(SC_coefs_matrix.shape[0]):
        SC_region = SC_coefs_matrix[roi_idx]
        pos_avg = np.sum(SC_region[SC_region > 0]) / len(SC_region)
        neg_avg = np.sum(SC_region[SC_region < 0]) / len(SC_region)
        SC_pos_coefs[roi_idx] = pos_avg
        SC_neg_coefs[roi_idx] = neg_avg

    # FC and FCgsr - Pallidum_L and Pallidum_R and Vermis set to nan (regions 75-76, 109-116)
    # Pallidum_L is at index 74, Pallidum_R is at index 75 so we need to skip over those (leave as nan)
    # FC matrix size is 109x109 but the last 3 regions are Vermis regions that we want to skip (so only index up to 105)
    # We skip Vermis regions because the AAL atlas has 8 Vermis regions but our data only has 3

    for idx in range(106): # 0 to 105 (We skip over the last 3 regions). Set range to 109 if you want to include the Vermis regions
        roi_idx = idx
        FC_region = FC_coefs_matrix[roi_idx]
        FCgsr_region = FCgsr_coefs_matrix[roi_idx]

        FC_pos_avg = np.sum(FC_region[FC_region > 0]) / len(FC_region)
        FC_neg_avg = np.sum(FC_region[FC_region < 0]) / len(FC_region)
        FCgsr_pos_avg = np.sum(FCgsr_region[FCgsr_region > 0]) / len(FCgsr_region)
        FCgsr_neg_avg = np.sum(FCgsr_region[FCgsr_region < 0]) / len(FCgsr_region)
        
        # Shift the indices by 2 after skipping over Pallidum_L and Pallidum_R
        if roi_idx >= 74:
            roi_idx = idx + 2

        FC_pos_coefs[roi_idx] = FC_pos_avg
        FC_neg_coefs[roi_idx] = FC_neg_avg
        FCgsr_pos_coefs[roi_idx] = FCgsr_pos_avg
        FCgsr_neg_coefs[roi_idx] = FCgsr_neg_avg


    aal_roi_vals = {
        "SC_pos_coefs": SC_pos_coefs,
        "SC_neg_coefs": SC_neg_coefs,
        "FC_pos_coefs": FC_pos_coefs,
        "FC_neg_coefs": FC_neg_coefs,
        "FCgsr_pos_coefs": FCgsr_pos_coefs,
        "FCgsr_neg_coefs": FCgsr_neg_coefs
    }

    return aal_roi_vals

def get_global_brain_region_range(control_only):
    """
    Calculate global color limits across all sex conditions (all/male/female)
    to ensure consistent colormaps.
    """
    all_values = []

    for male, female in [(False, False), (True, False), (False, True)]:
        SC_coefs_matrix, FC_coefs_matrix, FCgsr_coefs_matrix = get_coefs_matrices(control_only=control_only, male=male, female=female)
        aal_roi_vals = get_aal_roi_vals(SC_coefs_matrix=SC_coefs_matrix, FC_coefs_matrix=FC_coefs_matrix, FCgsr_coefs_matrix=FCgsr_coefs_matrix)

        for key in ["SC_pos_coefs", "SC_neg_coefs", "FC_pos_coefs", "FC_neg_coefs"]:
            vals = aal_roi_vals[key]
            all_values.extend(vals[~np.isnan(vals)])

    vmin = np.min(all_values)
    vmax = np.max(all_values)
    # Make symmetric around 0
    vmax_abs = max(abs(vmin), abs(vmax))
    return [-vmax_abs, vmax_abs]

def visualize_brain_regions(matrix_type, aal_roi_vals, positive, control_only, male, female, clim=None):
    """
    [visualize_brain_regions] creates and saves a montage figure of the AAL regions with the average positive or negative coefficients for a given matrix type
    """

    if control_only:
        file_name = 'control'
    else:
        file_name = 'control_moderate'

    if male:
        sex = '_male'
    elif female:
        sex = '_female'
    else:
        sex = ''

    if positive:
        sign = 'positive'
    else:
        sign = 'negative'

    # Use provided color limits or calculate from this dataset
    if clim is None:
        vals = aal_roi_vals[~np.isnan(aal_roi_vals)]
        if len(vals) > 0:
            vmin = np.min(vals)
            vmax = np.max(vals)
            vmax_abs = max(abs(vmin), abs(vmax))
            clim = [-vmax_abs, vmax_abs]
        else:
            clim = [-0.1, 0.1]  # fallback if no valid values

    # Create the montage figure
    create_montage_figure(aal_roi_vals,roilutfile="data/aal116_brainmontage/AAL116_LUT.tsv",lhannotfile="data/aal116_brainmontage/fsaverage.lh.AAL116.label.gii",rhannotfile="data/aal116_brainmontage/fsaverage.rh.AAL116.label.gii",annotsurfacename="fsaverage",subcorticalvolume="data/aal116_brainmontage/AAL116_subcortex.nii.gz",colormap="coolwarm",slice_dict={'axial':[23,33,43,53]},mosaic_dict={'axial':[-1,1]},add_colorbar=True,clim=clim,outputimagefile=f"figures/brain_regions_2/{matrix_type}_brain_regions_{sign}_{file_name}{sex}.png")

def create_square_format_brain_montage(control_only, male, female, clim=None):
    """
    Creates a 2x2 square format brain montage visualization:
    - Top left: SC positive coefficients
    - Top right: SC negative coefficients
    - Bottom left: FC positive coefficients
    - Bottom right: FC negative coefficients

    Similar to the square format heatmap in edges_heatmap.py

    Parameters:
    - clim: Optional color limits [vmin, vmax]. If not provided, calculated from this dataset only.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import os

    if control_only:
        file_name = 'control'
    else:
        file_name = 'control_moderate'

    if male:
        sex = '_male'
        title_suffix = ' (Male-only)'
    elif female:
        sex = '_female'
        title_suffix = ' (Female-only)'
    else:
        sex = ''
        title_suffix = ' (Combined)'

    # Get the coefficient matrices and AAL ROI values
    SC_coefs_matrix, FC_coefs_matrix, FCgsr_coefs_matrix = get_coefs_matrices(control_only=control_only, male=male, female=female)
    aal_roi_vals = get_aal_roi_vals(SC_coefs_matrix=SC_coefs_matrix, FC_coefs_matrix=FC_coefs_matrix, FCgsr_coefs_matrix=FCgsr_coefs_matrix)

    # Use provided color limits or calculate from this dataset
    if clim is None:
        all_values = []
        for key in ["SC_pos_coefs", "SC_neg_coefs", "FC_pos_coefs", "FC_neg_coefs"]:
            vals = aal_roi_vals[key]
            all_values.extend(vals[~np.isnan(vals)])

        vmin = np.min(all_values)
        vmax = np.max(all_values)
        # Make symmetric around 0
        vmax_abs = max(abs(vmin), abs(vmax))
        clim = [-vmax_abs, vmax_abs]

    # Create temporary directory for individual subplot images
    temp_dir = f'figures/brain_regions_2/temp{sex}'
    os.makedirs(temp_dir, exist_ok=True)

    # Generate individual brain montages
    temp_files = []

    # Top left: SC positive
    temp_file_sc_pos = f"{temp_dir}/sc_pos_temp.png"
    create_montage_figure(aal_roi_vals["SC_pos_coefs"], roilutfile="data/aal116_brainmontage/AAL116_LUT.tsv",
                         lhannotfile="data/aal116_brainmontage/fsaverage.lh.AAL116.label.gii",
                         rhannotfile="data/aal116_brainmontage/fsaverage.rh.AAL116.label.gii",
                         annotsurfacename="fsaverage", subcorticalvolume="data/aal116_brainmontage/AAL116_subcortex.nii.gz",
                         colormap="coolwarm", slice_dict={'axial':[23,33,43,53]}, mosaic_dict={'axial':[-1,1]},
                         add_colorbar=False, clim=clim, outputimagefile=temp_file_sc_pos)
    temp_files.append(temp_file_sc_pos)

    # Top right: SC negative
    temp_file_sc_neg = f"{temp_dir}/sc_neg_temp.png"
    create_montage_figure(aal_roi_vals["SC_neg_coefs"], roilutfile="data/aal116_brainmontage/AAL116_LUT.tsv",
                         lhannotfile="data/aal116_brainmontage/fsaverage.lh.AAL116.label.gii",
                         rhannotfile="data/aal116_brainmontage/fsaverage.rh.AAL116.label.gii",
                         annotsurfacename="fsaverage", subcorticalvolume="data/aal116_brainmontage/AAL116_subcortex.nii.gz",
                         colormap="coolwarm", slice_dict={'axial':[23,33,43,53]}, mosaic_dict={'axial':[-1,1]},
                         add_colorbar=False, clim=clim, outputimagefile=temp_file_sc_neg)
    temp_files.append(temp_file_sc_neg)

    # Bottom left: FC positive
    temp_file_fc_pos = f"{temp_dir}/fc_pos_temp.png"
    create_montage_figure(aal_roi_vals["FC_pos_coefs"], roilutfile="data/aal116_brainmontage/AAL116_LUT.tsv",
                         lhannotfile="data/aal116_brainmontage/fsaverage.lh.AAL116.label.gii",
                         rhannotfile="data/aal116_brainmontage/fsaverage.rh.AAL116.label.gii",
                         annotsurfacename="fsaverage", subcorticalvolume="data/aal116_brainmontage/AAL116_subcortex.nii.gz",
                         colormap="coolwarm", slice_dict={'axial':[23,33,43,53]}, mosaic_dict={'axial':[-1,1]},
                         add_colorbar=False, clim=clim, outputimagefile=temp_file_fc_pos)
    temp_files.append(temp_file_fc_pos)

    # Bottom right: FC negative (without colorbar - we'll add our own)
    temp_file_fc_neg = f"{temp_dir}/fc_neg_temp.png"
    create_montage_figure(aal_roi_vals["FC_neg_coefs"], roilutfile="data/aal116_brainmontage/AAL116_LUT.tsv",
                         lhannotfile="data/aal116_brainmontage/fsaverage.lh.AAL116.label.gii",
                         rhannotfile="data/aal116_brainmontage/fsaverage.rh.AAL116.label.gii",
                         annotsurfacename="fsaverage", subcorticalvolume="data/aal116_brainmontage/AAL116_subcortex.nii.gz",
                         colormap="coolwarm", slice_dict={'axial':[23,33,43,53]}, mosaic_dict={'axial':[-1,1]},
                         add_colorbar=False, clim=clim, outputimagefile=temp_file_fc_neg)
    temp_files.append(temp_file_fc_neg)

    # Create the combined figure
    from PIL import Image
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(2, 2, figure=fig, hspace=0.15, wspace=0.0)

    # Titles for each subplot
    titles = [
        'SC (Positive Coefficients)',
        'SC (Negative Coefficients)',
        'FC (Positive Coefficients)',
        'FC (Negative Coefficients)'
    ]

    # Load and display each image (without the colorbar from the last image)
    for idx, (temp_file, title) in enumerate(zip(temp_files, titles)):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])

        # Load the image and crop white space
        img = Image.open(temp_file)
        # Convert to numpy array to find non-white pixels
        import numpy as np
        img_array = np.array(img)
        # Find rows and columns that aren't all white (assuming white is 255,255,255)
        non_white_rows = np.where(~np.all(img_array == 255, axis=(1, 2)))[0]
        non_white_cols = np.where(~np.all(img_array == 255, axis=(0, 2)))[0]

        if len(non_white_rows) > 0 and len(non_white_cols) > 0:
            # Crop to non-white content
            img_cropped = img.crop((non_white_cols[0], non_white_rows[0],
                                   non_white_cols[-1]+1, non_white_rows[-1]+1))
        else:
            img_cropped = img

        ax.imshow(img_cropped)
        ax.axis('off')
        ax.set_title(title, fontsize=30, pad=15)

    # Adjust layout manually to control spacing
    # Remove tight_layout as it overrides wspace
    # Images are now cropped, so use small wspace for tight columns
    # Remove horizontal padding between columns so montages sit flush next to each other
    fig.subplots_adjust(left=0.04, right=0.92, top=0.95, bottom=0.05, hspace=0.15, wspace=0.0)

    # Add colorbar to the right of the entire figure (like edges_heatmap)
    # Adjusted to be about 80% of vertical height, centered
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.70])  # [left, bottom, width, height]
    norm = Normalize(vmin=clim[0], vmax=clim[1])
    sm = cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=18)

    # Save the combined figure
    output_dir = 'figures/brain_regions_2'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/square_format_brain_regions_{file_name}{sex}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)

    return output_file

if __name__ == "__main__":
    CONTROL_ONLY = False

    print("Generating square format brain region montages...")
    print("=" * 80)

    # Calculate global color limits across all sex conditions
    print("\nCalculating global color limits across all/male/female...")
    clim = get_global_brain_region_range(control_only=CONTROL_ONLY)
    print(f"Global color limits: [{clim[0]:.4f}, {clim[1]:.4f}]")

    # Generate square format brain montages for all subjects
    print("\n1. Generating square format brain montage for all subjects...")
    create_square_format_brain_montage(control_only=CONTROL_ONLY, male=False, female=False, clim=clim)

    # Generate square format brain montages for male only
    print("\n2. Generating square format brain montage for male subjects only...")
    create_square_format_brain_montage(control_only=CONTROL_ONLY, male=True, female=False, clim=clim)

    # Generate square format brain montages for female only
    print("\n3. Generating square format brain montage for female subjects only...")
    create_square_format_brain_montage(control_only=CONTROL_ONLY, male=False, female=True, clim=clim)

    print("\n" + "=" * 80)
    print("All square format brain region montages generated successfully!")
    print("=" * 80)

    # OLD CODE - Commented out but kept for reference
    # if CONTROL_ONLY:
    #     file_name = 'control'
    # else:
    #     file_name = 'control_moderate'
    #
    # if MALE:
    #     sex = '_male'
    # elif FEMALE:
    #     sex = '_female'
    # else:
    #     sex = ''
    #
    # SC_coefs_matrix, FC_coefs_matrix, FCgsr_coefs_matrix = get_coefs_matrices(control_only=CONTROL_ONLY, male=MALE, female=FEMALE)
    # aal_roi_vals = get_aal_roi_vals(SC_coefs_matrix=SC_coefs_matrix, FC_coefs_matrix=FC_coefs_matrix, FCgsr_coefs_matrix=FCgsr_coefs_matrix)
    #
    # pd.DataFrame(aal_roi_vals).to_csv(f'data/regions_info/aal_roi_vals/{file_name}{sex}_aal_roi_vals.csv', index=False)
    #
    # visualize_brain_regions(matrix_type='SC', aal_roi_vals=aal_roi_vals["SC_pos_coefs"], positive=True, control_only=CONTROL_ONLY, male=MALE, female=FEMALE)
    # visualize_brain_regions(matrix_type='SC', aal_roi_vals=aal_roi_vals["SC_neg_coefs"], positive=False, control_only=CONTROL_ONLY, male=MALE, female=FEMALE)
    # visualize_brain_regions(matrix_type='FC', aal_roi_vals=aal_roi_vals["FC_pos_coefs"], positive=True, control_only=CONTROL_ONLY, male=MALE, female=FEMALE)
    # visualize_brain_regions(matrix_type='FC', aal_roi_vals=aal_roi_vals["FC_neg_coefs"], positive=False, control_only=CONTROL_ONLY, male=MALE, female=FEMALE)
    # visualize_brain_regions(matrix_type='FCgsr', aal_roi_vals=aal_roi_vals["FCgsr_pos_coefs"], positive=True, control_only=CONTROL_ONLY, male=MALE, female=FEMALE)
    # visualize_brain_regions(matrix_type='FCgsr', aal_roi_vals=aal_roi_vals["FCgsr_neg_coefs"], positive=False, control_only=CONTROL_ONLY, male=MALE, female=FEMALE)

