"""
Logic:

Let's take the example of SC matrices.
The raw data for an SC matrix is a 90x90 matrix. (denoted as raw_SC_matrix)
The order of regions in this matrix is detailed in the "data/csv/tzo116plus_yeo7xhemi.csv" file.
We can visualize the coefficients between each region as a heatmap without doing any reordering.

But each region also belongs to a yeo network (1-9) and has a hierarchy (hist_g2_value) based on the hist_g2 scores.
We can reorder the regions based on the network and hierarchy to get a more meaningful visualization of the coefficients.

The code below (get_sorted_matrix_indices) reads the Yeo7 assignment file and the hist_g2 values and returns a dictionary of the sorted indices for each network.
So for example, SC_sorted_indices_dict[1] is a list of indices of the original raw_SC_matrix where the regions are in network 1 sorted by hist_g2_value.
So we can use these indices to directly index the raw_SC_matrix to extract the regions in a sorted manner for each network.
This is what is creating the SC_ordered_coefs_matrix which we can then visualize as a heatmap.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from get_haufe_coefs import get_haufe_coefs
from sig_coefs import get_sig_indices

def upper_tri_to_matrix(vector, size):
  """
  Converts a vector representing the upper triangular part of a matrix into a full square matrix.
  
  Parameters:
    vector (numpy.ndarray): Vector of upper triangular values.
    size (int): The size of the resulting square matrix (number of rows/columns).
    
  Returns:
    numpy.ndarray: Square matrix with the upper triangular values filled.
  """
  # Initialize an empty square matrix
  square_matrix = np.zeros((size, size))
  
  # Indices for the upper triangular part (excluding the diagonal)
  upper_triangle_indices = np.triu_indices(size, k=1)
  
  # Fill in the upper triangular part
  square_matrix[upper_triangle_indices] = vector

  # Fill in the lower triangular part (symmetric)
  square_matrix = square_matrix + square_matrix.T
  
  return square_matrix

def get_sorted_matrix_indices():
  """
  [get_sorted_matrix_indices] reads the Yeo7 assignment file and the hist_g2 values and returns a dictionary of the sorted indices for each network.
  These indices are sorted by the network (1-9) and then by the hierarchy (hist_g2_value) of the regions in each network.
  This is so that we can use these indices to directly index the unordered coefficients matrix to extract the regions in a sorted manner.
    e.g., SC_sorted_indices_dict = {1: [indices of regions of the original (unordered) coefficients matrix that are in network 1 sorted by hist_g2_value], 2: [indices of ...], ...}
  """
  # Load the Yeo7 assignment file
  df_yeo = pd.read_csv('data/csv/tzo116plus_yeo7xhemi.csv')
  # Adjust indices for 9-network parcellation
  df_yeo.loc[df_yeo['Yeo7_Index'] > 7, 'Yeo7_Index'] = df_yeo.loc[df_yeo['Yeo7_Index'] > 7, 'Yeo7_Index'] - 7 

  # Load the hist_g2 values
  df_g2 = pd.read_csv('data/hist_g2_scores/histg2_mean_tzo116plus.csv')
  df_g2.drop(columns=['region_label'], inplace=True)
  df_g2.rename(columns={"region_name": "ROI"}, inplace=True)
  # Reverse the sign to sort from low to high levels
  df_g2['hist_g2_value'] = df_g2['hist_g2_value'] * -1 

  # Merge the Yeo7 assignment and hist_g2 values
  df = df_yeo.merge(df_g2, on='ROI')
  df = df.drop(columns=['Used for tractography cortical', 'Index'])

  # Separate the SC and FC regions
  df_SC = df[df['Used for tractography subcortical'] == 'Y']
  # Reset indices to get a dataframe from 0-89 (for the 90 regions) --> This is how the SC matrix is ordered in the "tractography subcortical" folder (the raw data before flattening)
  df_SC.reset_index(drop=True, inplace=True)

  df_FC = df[df['Used for rs-fMRI gm signal'] == 'Y']
  # Reset indices to get a dataframe from 0-108 (for the 109 regions) --> This is how the FC matrix is ordered in the "NCANDA_FC.mat" (the raw data before flattening)
  df_FC.reset_index(drop=True, inplace=True)

  # Sort the regions based on the Yeo7_Index and hist_g2_value
  df_SC_sorted = df_SC.sort_values(by=['Yeo7_Index', 'hist_g2_value'])
  df_SC_sorted = df_SC_sorted.drop(columns=['Used for rs-fMRI gm signal', 'Used for tractography subcortical'])

  df_FC_sorted = df_FC.sort_values(by=['Yeo7_Index', 'hist_g2_value'])
  df_FC_sorted = df_FC_sorted.drop(columns=['Used for rs-fMRI gm signal', 'Used for tractography subcortical'])

  SC_sorted_indices_dict = {}
  FC_sorted_indices_dict = {}

  # For each network, store the sorted indices
  for i in [1,2,3,4,5,6,7,8,9]:
    SC_sorted_indices_dict[i] = df_SC_sorted[df_SC_sorted['Yeo7_Index'] == i].index.values
    FC_sorted_indices_dict[i] = df_FC_sorted[df_FC_sorted['Yeo7_Index'] == i].index.values

  return SC_sorted_indices_dict, FC_sorted_indices_dict

def get_ordered_coef_matrices(control_only, male, female):
  """
  [get_sorted_coef_matrices] loads the Haufe coefficients and returns the ordered and unordered matrices for SC, FC, and FCgsr.
  Ordered based on network (1-9) and hierarchy (hist_g2_value) of the regions in each network (low to high levels).
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
  SC_unordered_coefs_matrix = upper_tri_to_matrix(SC_coefficients, 90)
  FC_unordered_coefs_matrix = upper_tri_to_matrix(FC_coefficients, 109)
  FCgsr_unordered_coefs_matrix = upper_tri_to_matrix(FCgsr_coefficients, 109)

  # Reorder the matrix based on the network (1-9) and hierarchy (hist_g2_value) of the regions in each network (low to high levels)
  SC_sorted_indices_dict, FC_sorted_indices_dict = get_sorted_matrix_indices()

  # Initialize matrices ordered by network (1-9) and hist_g2_value (low to high)
  SC_ordered_coefs_matrix = np.zeros((90, 90))
  FC_ordered_coefs_matrix = np.zeros((109, 109))
  FCgsr_ordered_coefs_matrix = np.zeros((109, 109))

  # Combine sorted indices for easy reordering
  # New order: Visual, SomMot, DorsAttn, VentAttn, Limbic, Control, Default, Subcortex, Cerebellum
  # Network mapping: 1=Visual, 2=SomMot, 3=DorsAttn, 4=SalVentAttn, 5=Limbic, 6=Cont, 7=Default, 8=Subcortex, 9=Cerebellum
  # SC_sorted_indices = np.concatenate([SC_sorted_indices_dict[network] for network in [1,2,3,4,5,6,7,8]])
  # FC_sorted_indices = np.concatenate([FC_sorted_indices_dict[network] for network in [1,2,3,4,5,6,7,8,9]])

  SC_sorted_indices = np.concatenate([SC_sorted_indices_dict[network] for network in [1,2,3,4,5,6,7,8]])
  FC_sorted_indices = np.concatenate([FC_sorted_indices_dict[network] for network in [1,2,3,4,5,6,7,8,9]])

  # Order the coefficients matrices based on the sorted indices (rows and columns)
  SC_ordered_coefs_matrix = SC_unordered_coefs_matrix[SC_sorted_indices, :][:, SC_sorted_indices]
  FC_ordered_coefs_matrix = FC_unordered_coefs_matrix[FC_sorted_indices, :][:, FC_sorted_indices]
  FCgsr_ordered_coefs_matrix = FCgsr_unordered_coefs_matrix[FC_sorted_indices, :][:, FC_sorted_indices]

  return SC_unordered_coefs_matrix, FC_unordered_coefs_matrix, FCgsr_unordered_coefs_matrix, SC_ordered_coefs_matrix, FC_ordered_coefs_matrix, FCgsr_ordered_coefs_matrix

def network_block_means(matrix_type, ordered_matrix, sign=None):
  """
  [network_block_means] calculates the mean of each network block in the reordered matrices (SC, FC, FCgsr).
  New order: Visual, SomMot, DorsAttn, VentAttn, Limbic, Control, Default, Subcortex, Cerebellum
  """
  # Initialize empty network matrices and network indices
  if matrix_type == 'SC':
    network_matrix = np.zeros((8,8))
    # Network order: Visual, SomMot, DorsAttn, VentAttn, Limbic, Control, Default, Subcortex
    # Each network starts at the indices listed. (e.g., Visual starts at 0, SomMot starts at 14, etc.)
    network_indices = [0, 14, 28, 32, 38, 50, 57, 76, 90]
  elif matrix_type == 'FC' or matrix_type == 'FCgsr':
    network_matrix = np.zeros((9,9))
    # Network order: Visual, SomMot, DorsAttn, VentAttn, Limbic, Control, Default, Subcortex, Cerebellum
    # Each network starts at the indices listed. (e.g., Visual starts at 0, SomMot starts at 14, etc.)
    network_indices = [0, 14, 28, 32, 38, 50, 57, 76, 88, 109]

  for i in range(len(network_indices) - 1):
    for j in range(len(network_indices) - 1):
      block = ordered_matrix[network_indices[i]:network_indices[i+1], network_indices[j]:network_indices[j+1]]
      if sign == 'positive':
        block_values = block[block > 0]  # Extract positive values from the block
      elif sign == 'negative':
        block_values = block[block < 0]  # Extract negative values from the block
      else:
        block_values = block
      network_matrix[i, j] = np.sum(block_values) / block.size  # Calculate the average of positive values

  return network_matrix

def get_global_heatmap_range(control_only):
  """
  Calculate global color range across all sex conditions (all/male/female)
  to ensure consistent colormaps.
  """
  all_values_global = []

  for male, female in [(False, False), (True, False), (False, True)]:
    SC_unordered, FC_unordered, FCgsr_unordered, SC_ordered, FC_ordered, FCgsr_ordered = get_ordered_coef_matrices(
      control_only=control_only, male=male, female=female
    )

    SC_positive = network_block_means('SC', SC_ordered, sign='positive')
    SC_negative = network_block_means('SC', SC_ordered, sign='negative')
    FC_positive = network_block_means('FC', FC_ordered, sign='positive')
    FC_negative = network_block_means('FC', FC_ordered, sign='negative')

    all_values_global.extend([
      SC_positive.flatten(), SC_negative.flatten(),
      FC_positive.flatten(), FC_negative.flatten()
    ])

  all_values = np.concatenate(all_values_global)
  vmin = np.min(all_values)
  vmax = np.max(all_values)
  # Make symmetric around 0
  vmax_abs = max(abs(vmin), abs(vmax))
  return -vmax_abs, vmax_abs

def create_square_format_heatmap(control_only, male, female, save_fig, vmin=None, vmax=None):
  """
  Creates a 2x2 square format heatmap:
  - Top left: SC positive coefficients
  - Top right: SC negative coefficients
  - Bottom left: FC positive coefficients
  - Bottom right: FC negative coefficients

  Network order: Visual, SomMot, DorsAttn, VentAttn, Limbic, Control, Default, Subcortex, Cerebellum

  Parameters:
  - vmin, vmax: Optional color scale limits. If not provided, calculated from this dataset only.
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

  # Get ordered coefficient matrices
  SC_unordered, FC_unordered, FCgsr_unordered, SC_ordered, FC_ordered, FCgsr_ordered = get_ordered_coef_matrices(
    control_only=control_only, male=male, female=female
  )

  # Calculate network means for positive and negative coefficients
  SC_positive = network_block_means('SC', SC_ordered, sign='positive')
  SC_negative = network_block_means('SC', SC_ordered, sign='negative')
  FC_positive = network_block_means('FC', FC_ordered, sign='positive')
  FC_negative = network_block_means('FC', FC_ordered, sign='negative')

  # Use provided color scale or calculate from this dataset
  if vmin is None or vmax is None:
    all_values = np.concatenate([
      SC_positive.flatten(), SC_negative.flatten(),
      FC_positive.flatten(), FC_negative.flatten()
    ])
    vmin = np.min(all_values)
    vmax = np.max(all_values)
    # Make symmetric around 0
    vmax_abs = max(abs(vmin), abs(vmax))
    vmin = -vmax_abs
    vmax = vmax_abs

  # Create figure with 2x2 subplots
  fig, axes = plt.subplots(2, 2, figsize=(20, 18))

  # Set tick labels for SC and FC (new order: Visual, SomMot, DorsAttn, VentAttn, Limbic, Control, Default, Subcortex, Cerebellum)
  SC_tick_labels = ['Visual', 'SomMot', 'DorsAttn', 'VentAttn', 'Limbic', 'Control', 'Default', 'Subcortex']
  FC_tick_labels = ['Visual', 'SomMot', 'DorsAttn', 'VentAttn', 'Limbic', 'Control', 'Default', 'Subcortex', 'Cerebellum']

  # Plot SC positive (top left)
  ax1 = axes[0, 0]
  sns.heatmap(SC_positive, vmin=vmin, vmax=vmax, cmap='coolwarm', linewidths=0.1,
              square=True, annot=True, fmt=".3f", ax=ax1, cbar=False,
              annot_kws={'fontsize': 10})
  ax1.set_xticks(np.arange(0.5, len(SC_positive), 1))
  ax1.set_xticklabels(SC_tick_labels, rotation=30, ha='center', fontsize=22)
  ax1.set_yticks(np.arange(0.5, len(SC_positive), 1))
  ax1.set_yticklabels(SC_tick_labels, rotation=0, fontsize=22)
  ax1.set_xlabel('')
  ax1.set_ylabel('')
  ax1.set_title('SC (Positive Coefficients)', fontsize=30, pad=15)

  # Plot SC negative (top right)
  ax2 = axes[0, 1]
  sns.heatmap(SC_negative, vmin=vmin, vmax=vmax, cmap='coolwarm', linewidths=0.1,
              square=True, annot=True, fmt=".3f", ax=ax2, cbar=False,
              annot_kws={'fontsize': 10})
  ax2.set_xticks(np.arange(0.5, len(SC_negative), 1))
  ax2.set_xticklabels(SC_tick_labels, rotation=30, ha='center', fontsize=22)
  ax2.set_yticks(np.arange(0.5, len(SC_negative), 1))
  ax2.set_yticklabels(SC_tick_labels, rotation=0, fontsize=22)
  ax2.yaxis.tick_left()
  ax2.set_xlabel('')
  ax2.set_ylabel('')
  ax2.set_title('SC (Negative Coefficients)', fontsize=30, pad=15)

  # Plot FC positive (bottom left)
  ax3 = axes[1, 0]
  sns.heatmap(FC_positive, vmin=vmin, vmax=vmax, cmap='coolwarm', linewidths=0.1,
              square=True, annot=True, fmt=".3f", ax=ax3, cbar=False,
              annot_kws={'fontsize': 10})
  ax3.set_xticks(np.arange(0.5, len(FC_positive), 1))
  ax3.set_xticklabels(FC_tick_labels, rotation=30, ha='center', fontsize=22)
  ax3.set_yticks(np.arange(0.5, len(FC_positive), 1))
  ax3.set_yticklabels(FC_tick_labels, rotation=0, fontsize=22)
  ax3.set_xlabel('')
  ax3.set_ylabel('')
  ax3.set_title('FC (Positive Coefficients)', fontsize=30, pad=15)

  # Plot FC negative (bottom right) - with colorbar
  ax4 = axes[1, 1]
  im = sns.heatmap(FC_negative, vmin=vmin, vmax=vmax, cmap='coolwarm', linewidths=0.1,
                   square=True, annot=True, fmt=".3f", ax=ax4, cbar=False,
                   annot_kws={'fontsize': 10})
  ax4.set_xticks(np.arange(0.5, len(FC_negative), 1))
  ax4.set_xticklabels(FC_tick_labels, rotation=30, ha='center', fontsize=22)
  ax4.set_yticks(np.arange(0.5, len(FC_negative), 1))
  ax4.set_yticklabels(FC_tick_labels, rotation=0, fontsize=22)
  ax4.yaxis.tick_left()
  ax4.set_xlabel('')
  ax4.set_ylabel('')
  ax4.set_title('FC (Negative Coefficients)', fontsize=30, pad=15)

  # Adjust layout first to ensure proper spacing
  # Add extra vertical spacing between rows (hspace controls height spacing)
  plt.tight_layout(h_pad=3.0)

  # Add colorbar after tight_layout to prevent it from affecting plot spacing
  # Place it to the right of the entire figure
  # hspace controls vertical spacing between rows, wspace controls horizontal spacing between columns
  fig.subplots_adjust(right=0.92, hspace=0.3, wspace=0.15)
  cbar_ax = fig.add_axes([0.94, 0.11, 0.02, 0.77])  # [left, bottom, width, height]
  cbar = fig.colorbar(ax4.collections[0], cax=cbar_ax)
  cbar.ax.tick_params(labelsize=18)

  if save_fig:
    # Create appropriate directory path
    if sex:
      dir_path = f'figures/edges_heatmap{sex}'
      file_path = f'{dir_path}/square_format_network_means_{file_name}{sex}.png'
    else:
      dir_path = 'figures/edges_heatmap'
      file_path = f'{dir_path}/square_format_network_means_{file_name}.png'

    import os
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {file_path}")
  else:
    plt.show()

  plt.close()

def visualize_matrix_heatmap(matrix, matrix_type, control_only, male, female, reorder, network_mean, sign, save_fig):
  """
  [visualize_matrix_heatmap] visualizes the matrix as a heatmap.
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

  if reorder:
    order = '_ordered'
    # print('matrix_type for ordered: ', matrix_type)
    # print('max value: ', np.max(matrix))
    # print('min value: ', np.min(matrix))
    if matrix_type == 'SC':
      # New order: Visual, SomMot, DorsAttn, VentAttn, Limbic, Control, Default, Subcortex
      # Each network starts at the indices listed. (e.g., Visual starts at 0, SomMot starts at 14, etc.)
      tick_indices = [0, 14, 28, 32, 38, 50, 57, 76]
      tick_labels = ['Visual', 'SomMot', 'DorsAttn', 'VentAttn', 'Limbic', 'Control', 'Default', 'Subcortex']

    elif matrix_type == 'FC' or matrix_type == 'FCgsr':
      # New order: Visual, SomMot, DorsAttn, VentAttn, Limbic, Control, Default, Subcortex, Cerebellum
      # Each network starts at the indices listed. (e.g., Visual starts at 0, SomMot starts at 14, etc.)
      tick_indices = [0, 14, 28, 32, 38, 50, 57, 76, 88]
      tick_labels = ['Visual', 'SomMot', 'DorsAttn', 'VentAttn', 'Limbic', 'Control', 'Default', 'Subcortex', 'Cerebellum']

    if network_mean:
      matrix = network_block_means(matrix_type, matrix, sign)
      order = '_network_means'
    
  else:
    order = '_unordered'

  # Calculate dynamic color scale
  vmin = np.min(matrix)
  vmax = np.max(matrix)
  # Make symmetric around 0
  vmax_abs = max(abs(vmin), abs(vmax))
  vmin = -vmax_abs
  vmax = vmax_abs

  # Create heatmap
  plt.figure(figsize=(10, 8))
  if network_mean:
    ax = sns.heatmap(matrix, vmin=vmin, vmax=vmax, cmap='coolwarm', linewidths=0.1, square=True, annot=True, fmt=".3f")
  else:
    ax = sns.heatmap(matrix, vmin=vmin, vmax=vmax, cmap='coolwarm', linewidths=0.1, square=True)
  if sign == 'positive':
    sign_name = '_positive'
    plt.title(f'Network Means for {matrix_type} - Positive Haufe Coefficients')
  elif sign == 'negative':
    sign_name = '_negative'
    plt.title(f'Network Means for {matrix_type} - Negative Haufe Coefficients')
  else:
    sign_name = ''
    plt.title(f'Edges Heatmap for {matrix_type} Haufe Coefficients ({file_name}{sex}{order})')

  plt.xlabel('Region')
  plt.ylabel('Region')

  if reorder:
    if network_mean:
      ax.set_xticks(np.arange(0.5, len(matrix), 1))
      ax.set_xticklabels(tick_labels, rotation=90)
      ax.set_yticks(np.arange(0.5, len(matrix), 1))
      ax.set_yticklabels(tick_labels, rotation=0)
    else:
      ax.set_xticks(tick_indices)
      ax.set_xticklabels(tick_labels, rotation=90)
      ax.set_yticks(tick_indices)
      ax.set_yticklabels(tick_labels, rotation=0)

    plt.xlabel('Network')
    plt.ylabel('Network')
    plt.tight_layout()

  if save_fig:
    plt.savefig(f'figures/edges_heatmap/{sex}/{matrix_type}_edges_heatmap_{file_name}{order}{sex}{sign_name}.png')
  else:
    plt.show()

if __name__ == '__main__':
  CONTROL_ONLY = False
  SAVE_FIG = True


  print("Generating square format heatmaps...")
  print("=" * 80)

  # Calculate global color range across all sex conditions
  print("\nCalculating global color range across all/male/female...")
  vmin, vmax = get_global_heatmap_range(control_only=CONTROL_ONLY)
  print(f"Global color range: [{vmin:.4f}, {vmax:.4f}]")

  # Generate square format heatmaps for all subjects
  print("\n1. Generating square format heatmap for all subjects...")
  create_square_format_heatmap(control_only=CONTROL_ONLY, male=False, female=False, save_fig=SAVE_FIG, vmin=vmin, vmax=vmax)

  # Generate square format heatmaps for male only
  print("\n2. Generating square format heatmap for male subjects only...")
  create_square_format_heatmap(control_only=CONTROL_ONLY, male=True, female=False, save_fig=SAVE_FIG, vmin=vmin, vmax=vmax)

  # Generate square format heatmaps for female only
  print("\n3. Generating square format heatmap for female subjects only...")
  create_square_format_heatmap(control_only=CONTROL_ONLY, male=False, female=True, save_fig=SAVE_FIG, vmin=vmin, vmax=vmax)

  print("\n" + "=" * 80)
  print("All square format heatmaps generated successfully!")
  print("=" * 80)

  # OLD CODE - Commented out but kept for reference
  # SC_unordered_coefs_matrix, FC_unordered_coefs_matrix, FCgsr_unordered_coefs_matrix, SC_ordered_coefs_matrix, FC_ordered_coefs_matrix, FCgsr_ordered_coefs_matrix = get_ordered_coef_matrices(control_only=CONTROL_ONLY, male=MALE, female=FEMALE)

  # # Visualize the unordered matrices (no reordering --> just the raw data)
  # visualize_matrix_heatmap(matrix=SC_unordered_coefs_matrix, matrix_type='SC', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, reorder=False, network_mean=False, save_fig=SAVE_FIG)
  # visualize_matrix_heatmap(matrix=FC_unordered_coefs_matrix, matrix_type='FC', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, reorder=False, network_mean=False, save_fig=SAVE_FIG)
  # visualize_matrix_heatmap(matrix=FCgsr_unordered_coefs_matrix, matrix_type='FCgsr', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, reorder=False, network_mean=False, save_fig=SAVE_FIG)

  # # Visualize the ordered matrices (ordered by network and hierarchy)
  # visualize_matrix_heatmap(matrix=SC_ordered_coefs_matrix, matrix_type='SC', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, reorder=True, network_mean=False, sign=None, save_fig=SAVE_FIG)
  # visualize_matrix_heatmap(matrix=FC_ordered_coefs_matrix, matrix_type='FC', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, reorder=True, network_mean=False, sign=None, save_fig=SAVE_FIG)
  # visualize_matrix_heatmap(matrix=FCgsr_ordered_coefs_matrix, matrix_type='FCgsr', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, reorder=True, network_mean=False, sign=None, save_fig=SAVE_FIG)

  # # Visualize network means (average of each network block) (8x8 for SC, 9x9 for FC and FCgsr)
  # visualize_matrix_heatmap(matrix=SC_ordered_coefs_matrix, matrix_type='SC', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, reorder=True, network_mean=True, sign=None, save_fig=SAVE_FIG)
  # visualize_matrix_heatmap(matrix=FC_ordered_coefs_matrix, matrix_type='FC', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, reorder=True, network_mean=True, sign=None, save_fig=SAVE_FIG)
  # visualize_matrix_heatmap(matrix=FCgsr_ordered_coefs_matrix, matrix_type='FCgsr', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, reorder=True, network_mean=True, sign=None, save_fig=SAVE_FIG)

  # # Visualize network means with positive values only
  # visualize_matrix_heatmap(matrix=SC_ordered_coefs_matrix, matrix_type='SC', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, reorder=True, network_mean=True, sign='positive', save_fig=SAVE_FIG)
  # visualize_matrix_heatmap(matrix=FC_ordered_coefs_matrix, matrix_type='FC', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, reorder=True, network_mean=True, sign='positive', save_fig=SAVE_FIG)
  # visualize_matrix_heatmap(matrix=FCgsr_ordered_coefs_matrix, matrix_type='FCgsr', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, reorder=True, network_mean=True, sign='positive', save_fig=SAVE_FIG)

  # # Visualize network means with negative values only
  # visualize_matrix_heatmap(matrix=SC_ordered_coefs_matrix, matrix_type='SC', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, reorder=True, network_mean=True, sign='negative', save_fig=SAVE_FIG)
  # visualize_matrix_heatmap(matrix=FC_ordered_coefs_matrix, matrix_type='FC', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, reorder=True, network_mean=True, sign='negative', save_fig=SAVE_FIG)
  # visualize_matrix_heatmap(matrix=FCgsr_ordered_coefs_matrix, matrix_type='FCgsr', control_only=CONTROL_ONLY, male=MALE, female=FEMALE, reorder=True, network_mean=True, sign='negative', save_fig=SAVE_FIG)

