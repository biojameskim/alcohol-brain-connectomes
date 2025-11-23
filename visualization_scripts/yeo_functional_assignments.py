import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from edges_heatmap import upper_tri_to_matrix
from sig_coefs import get_sig_indices
from get_haufe_coefs import get_haufe_coefs

def get_network_indices(yeo_assignment_file):
  """
  [get_network_indices] reads the Yeo7 assignment file and returns the indices where the regions in the SC and FC matrices belong to each network
  """
  df = pd.read_csv(yeo_assignment_file)
  df.loc[df['Yeo7_Index'] > 7, 'Yeo7_Index'] = df.loc[df['Yeo7_Index'] > 7, 'Yeo7_Index'] - 7 # Adjust indices for 9-network parcellation

  df_SC = df[df['Used for tractography subcortical'] == 'Y']
  df_SC.reset_index(drop=True, inplace=True)

  df_FC = df[df['Used for rs-fMRI gm signal'] == 'Y']
  df_FC.reset_index(drop=True, inplace=True)

  # Dictionaries to store the results
  SC_network_indices = {}
  FC_network_indices = {}

  for network in [1,2,3,4,5,6,7,8]:
    # Find indices in df_SC and df_FC where 'Yeo7_Index' matches the network value
    SC_network = np.where(df_SC['Yeo7_Index'] == network)
    FC_network = np.where(df_FC['Yeo7_Index'] == network)
    
    # Save the results into dictionaries
    SC_network_indices[network] = SC_network
    FC_network_indices[network] = FC_network
  
  FC_network_indices[9] = np.where(df_FC['Yeo7_Index'] == 9) # Only FC/FCgsr has Cerebellum regions

  return SC_network_indices, FC_network_indices

def get_network_influences(SC_region_vector, FC_region_vector, FCgsr_region_vector, SC_pos_region_vector, SC_neg_region_vector, FC_pos_region_vector, FC_neg_region_vector, FCgsr_pos_region_vector, FCgsr_neg_region_vector, SC_network_indices, FC_network_indices):
  """
  [get_network_influences] calculates the mean influence of each network in the SC and FC matrices for positive and negative values separately.
  """
  SC_network_influences = {}
  FC_network_influences = {}
  FCgsr_network_influences = {}

  SC_pos_network_influences = {}
  SC_neg_network_influences = {}
  FC_pos_network_influences = {}

  FC_neg_network_influences = {}
  FCgsr_pos_network_influences = {}
  FCgsr_neg_network_influences = {}

  for network in [1,2,3,4,5,6,7,8]:
    # Get the indices for the network
    SC_indices = SC_network_indices[network][0]
    FC_indices = FC_network_indices[network][0]

    # Calculate the mean influence for the network
    SC_network_influences[network] = np.round(np.mean(SC_region_vector[SC_indices]), 3)
    FC_network_influences[network] = np.round(np.mean(FC_region_vector[FC_indices]), 3)
    FCgsr_network_influences[network] = np.round(np.mean(FCgsr_region_vector[FC_indices]), 3)

    SC_pos_network_influences[network] = np.round(np.mean(SC_pos_region_vector[SC_indices]), 3)
    SC_neg_network_influences[network] = np.round(np.mean(SC_neg_region_vector[SC_indices]), 3)
    FC_pos_network_influences[network] = np.round(np.mean(FC_pos_region_vector[FC_indices]), 3)

    FC_neg_network_influences[network] = np.round(np.mean(FC_neg_region_vector[FC_indices]), 3)
    FCgsr_pos_network_influences[network] = np.round(np.mean(FCgsr_pos_region_vector[FC_indices]), 3)
    FCgsr_neg_network_influences[network] = np.round(np.mean(FCgsr_neg_region_vector[FC_indices]), 3)
  
  # Only FC/FCgsr has Cerebellum regions
  FC_network_influences[9] = np.round(np.mean(FC_region_vector[FC_network_indices[9][0]]), 3)
  FCgsr_network_influences[9] = np.round(np.mean(FCgsr_region_vector[FC_network_indices[9][0]]), 3)

  FC_pos_network_influences[9] = np.round(np.mean(FC_pos_region_vector[FC_network_indices[9][0]]), 3)
  FC_neg_network_influences[9] = np.round(np.mean(FC_neg_region_vector[FC_network_indices[9][0]]), 3)

  FCgsr_pos_network_influences[9] = np.round(np.mean(FCgsr_pos_region_vector[FC_network_indices[9][0]]), 3)
  FCgsr_neg_network_influences[9] = np.round(np.mean(FCgsr_neg_region_vector[FC_network_indices[9][0]]), 3)

  network_influences = {
    "SC_network_influences": SC_network_influences,
    "FC_network_influences": FC_network_influences,
    "FCgsr_network_influences": FCgsr_network_influences,
    "SC_pos_network_influences": SC_pos_network_influences,
    "SC_neg_network_influences": SC_neg_network_influences,
    "FC_pos_network_influences": FC_pos_network_influences,
    "FC_neg_network_influences": FC_neg_network_influences,
    "FCgsr_pos_network_influences": FCgsr_pos_network_influences,
    "FCgsr_neg_network_influences": FCgsr_neg_network_influences
  }

  return network_influences

def yeo_network_barplot(network_influences, file_name, sex):
  networks = ['Visual', 'SomMot', 'DorsAttn', 'VentAttn', 'Limbic', 'Control', 'Default', 'Subcortex', 'Cerebellum']
  positive_coeffs_SC = [network_influences['SC_pos_network_influences'][1], network_influences['SC_pos_network_influences'][2], network_influences['SC_pos_network_influences'][3], network_influences['SC_pos_network_influences'][4], network_influences['SC_pos_network_influences'][5], network_influences['SC_pos_network_influences'][6], network_influences['SC_pos_network_influences'][7], network_influences['SC_pos_network_influences'][8], 0]
  negative_coeffs_SC = [network_influences['SC_neg_network_influences'][1], network_influences['SC_neg_network_influences'][2], network_influences['SC_neg_network_influences'][3], network_influences['SC_neg_network_influences'][4], network_influences['SC_neg_network_influences'][5], network_influences['SC_neg_network_influences'][6], network_influences['SC_neg_network_influences'][7], network_influences['SC_neg_network_influences'][8], 0]
  positive_coeffs_FC = [network_influences['FC_pos_network_influences'][1], network_influences['FC_pos_network_influences'][2], network_influences['FC_pos_network_influences'][3], network_influences['FC_pos_network_influences'][4], network_influences['FC_pos_network_influences'][5], network_influences['FC_pos_network_influences'][6], network_influences['FC_pos_network_influences'][7], network_influences['FC_pos_network_influences'][8], network_influences['FC_pos_network_influences'][9]]
  negative_coeffs_FC = [network_influences['FC_neg_network_influences'][1], network_influences['FC_neg_network_influences'][2], network_influences['FC_neg_network_influences'][3], network_influences['FC_neg_network_influences'][4], network_influences['FC_neg_network_influences'][5], network_influences['FC_neg_network_influences'][6], network_influences['FC_neg_network_influences'][7], network_influences['FC_neg_network_influences'][8], network_influences['FC_neg_network_influences'][9]]

  print(positive_coeffs_SC)
  print(negative_coeffs_SC)
  print(positive_coeffs_FC)
  print(negative_coeffs_FC)

  # Number of networks
  n = len(networks)
  # Position of the bars on the x-axis
  ind = np.arange(n)
  # Width of a bar
  width = 0.35

  fig, ax = plt.subplots(figsize=(10, 7))

  # Plotting the bars
  bars1 = ax.bar(ind - width/2, positive_coeffs_SC, width, label='Positive SC', color='b')
  bars2 = ax.bar(ind + width/2, positive_coeffs_FC, width, label='Positive FC', color='g')
  bars3 = ax.bar(ind - width/2, negative_coeffs_SC, width, label='Negative SC', color='r')
  bars4 = ax.bar(ind + width/2, negative_coeffs_FC, width, label='Negative FC', color='orange')

  # Adding a horizontal line at y=0
  ax.axhline(0, color='black', linewidth=0.8)

  # Adding labels
  ax.set_xlabel('Networks')
  ax.set_ylabel('Mean Coefficients')
  ax.set_title('Yeo Network Mean Influences')
  ax.set_xticks(ind)
  ax.set_xticklabels(networks)
  ax.legend()

  # Calculate dynamic y-axis limits
  all_values = positive_coeffs_SC + negative_coeffs_SC + positive_coeffs_FC + negative_coeffs_FC
  y_min = min(all_values)
  y_max = max(all_values)
  y_range = y_max - y_min
  y_min = y_min - 0.05 * y_range
  y_max = y_max + 0.05 * y_range
  ax.set_ylim(y_min, y_max)

  # plt.show()
  plt.savefig(f'figures/yeo_network_barplots/yeo_network_influences_barplot_{file_name}{sex}.png')

def create_combined_yeo_barplot(file_name):
  """
  Creates a combined figure with three subplots showing Yeo network influences for:
  1. Combined (all subjects)
  2. Male-only
  3. Female-only
  """
  # Get network influences for all three conditions
  conditions = [
    ('', 'Combined (All Subjects)'),
    ('_male', 'Male-only'),
    ('_female', 'Female-only')
  ]

  all_network_influences = []
  all_bar_values = []  # Collect all values for dynamic y-axis
  for sex, _ in conditions:
    if sex == '_male':
      MALE, FEMALE = True, False
    elif sex == '_female':
      MALE, FEMALE = False, True
    else:
      MALE, FEMALE = False, False

    CONTROL_ONLY = (file_name == 'control')

    SC_reject, FC_reject, FCgsr_reject = get_sig_indices(control_only=CONTROL_ONLY, male=MALE, female=FEMALE, p_value_threshold=0.05)
    avg_SC_haufe_coefs = get_haufe_coefs(matrix_type='SC', file_name=file_name, sex=sex)
    avg_FC_haufe_coefs = get_haufe_coefs(matrix_type='FC', file_name=file_name, sex=sex)
    avg_FCgsr_haufe_coefs = get_haufe_coefs(matrix_type='FCgsr', file_name=file_name, sex=sex)

    SC_false_indices = np.where(SC_reject == False)
    FC_false_indices = np.where(FC_reject == False)
    FCgsr_false_indices = np.where(FCgsr_reject == False)

    avg_SC_haufe_coefs[SC_false_indices] = 0
    avg_FC_haufe_coefs[FC_false_indices] = 0
    avg_FCgsr_haufe_coefs[FCgsr_false_indices] = 0

    SC_coef_matrix = upper_tri_to_matrix(avg_SC_haufe_coefs, 90)
    FC_coef_matrix = upper_tri_to_matrix(avg_FC_haufe_coefs, 109)
    FCgsr_coef_matrix = upper_tri_to_matrix(avg_FCgsr_haufe_coefs, 109)

    SC_region_vector = np.sum(SC_coef_matrix, axis=1)
    FC_region_vector = np.sum(FC_coef_matrix, axis=1)
    FCgsr_region_vector = np.sum(FCgsr_coef_matrix, axis=1)

    SC_pos_region_vector = np.sum(np.where(SC_coef_matrix > 0, SC_coef_matrix, 0), axis=1)
    SC_neg_region_vector = np.sum(np.where(SC_coef_matrix < 0, SC_coef_matrix, 0), axis=1)
    FC_pos_region_vector = np.sum(np.where(FC_coef_matrix > 0, FC_coef_matrix, 0), axis=1)
    FC_neg_region_vector = np.sum(np.where(FC_coef_matrix < 0, FC_coef_matrix, 0), axis=1)
    FCgsr_pos_region_vector = np.sum(np.where(FCgsr_coef_matrix > 0, FCgsr_coef_matrix, 0), axis=1)
    FCgsr_neg_region_vector = np.sum(np.where(FCgsr_coef_matrix < 0, FCgsr_coef_matrix, 0), axis=1)

    SC_network_indices, FC_network_indices = get_network_indices('data/csv/tzo116plus_yeo7xhemi.csv')

    network_influences = get_network_influences(
      SC_region_vector=SC_region_vector,
      FC_region_vector=FC_region_vector,
      FCgsr_region_vector=FCgsr_region_vector,
      SC_pos_region_vector=SC_pos_region_vector,
      SC_neg_region_vector=SC_neg_region_vector,
      FC_pos_region_vector=FC_pos_region_vector,
      FC_neg_region_vector=FC_neg_region_vector,
      FCgsr_pos_region_vector=FCgsr_pos_region_vector,
      FCgsr_neg_region_vector=FCgsr_neg_region_vector,
      SC_network_indices=SC_network_indices,
      FC_network_indices=FC_network_indices
    )

    all_network_influences.append(network_influences)

    # Collect all bar values for this condition
    positive_coeffs_SC = [network_influences['SC_pos_network_influences'][i] if i <= 8 else 0 for i in range(1, 9)] + [0]
    negative_coeffs_SC = [network_influences['SC_neg_network_influences'][i] if i <= 8 else 0 for i in range(1, 9)] + [0]
    positive_coeffs_FC = [network_influences['FC_pos_network_influences'][i] for i in range(1, 9)] + [network_influences['FC_pos_network_influences'][9]]
    negative_coeffs_FC = [network_influences['FC_neg_network_influences'][i] for i in range(1, 9)] + [network_influences['FC_neg_network_influences'][9]]
    all_bar_values.extend(positive_coeffs_SC + negative_coeffs_SC + positive_coeffs_FC + negative_coeffs_FC)

  # Calculate dynamic y-axis limits based on all values
  y_min = min(all_bar_values)
  y_max = max(all_bar_values)
  y_range = y_max - y_min
  y_min = y_min - 0.05 * y_range
  y_max = y_max + 0.15 * y_range

  # Create figure with 3 subplots
  fig, axes = plt.subplots(1, 3, figsize=(30, 8))

  networks = ['Visual', 'SomMot', 'DorsAttn', 'VentAttn', 'Limbic', 'Control', 'Default', 'Subcortex', 'Cerebellum']
  n = len(networks)
  ind = np.arange(n)
  width = 0.35

  for idx, (ax, (sex, title), network_influences) in enumerate(zip(axes, conditions, all_network_influences)):
    positive_coeffs_SC = [network_influences['SC_pos_network_influences'][i] if i <= 8 else 0 for i in range(1, 9)] + [0]
    negative_coeffs_SC = [network_influences['SC_neg_network_influences'][i] if i <= 8 else 0 for i in range(1, 9)] + [0]
    positive_coeffs_FC = [network_influences['FC_pos_network_influences'][i] for i in range(1, 9)] + [network_influences['FC_pos_network_influences'][9]]
    negative_coeffs_FC = [network_influences['FC_neg_network_influences'][i] for i in range(1, 9)] + [network_influences['FC_neg_network_influences'][9]]

    # Plotting the bars
    ax.bar(ind - width/2, positive_coeffs_SC, width, label='Positive SC', color='b')
    ax.bar(ind + width/2, positive_coeffs_FC, width, label='Positive FC', color='g')
    ax.bar(ind - width/2, negative_coeffs_SC, width, label='Negative SC', color='r')
    ax.bar(ind + width/2, negative_coeffs_FC, width, label='Negative FC', color='orange')

    # Adding a horizontal line at y=0
    ax.axhline(0, color='black', linewidth=0.8)

    # Set x-ticks and labels with diagonal rotation
    ax.set_xticks(ind)
    ax.set_xticklabels(networks, fontsize=20, rotation=30, ha='center')

    # Add title for each subplot
    ax.set_title(title, fontsize=24)

    # Only show y-axis tick labels on leftmost plot
    if idx == 0:
      ax.tick_params(axis='y', labelsize=20)
    else:
      ax.tick_params(axis='y', labelleft=False)

    # Only show legend on leftmost plot
    if idx == 0:
      ax.legend(fontsize=16)

    # Set dynamic y-axis limits
    ax.set_ylim(y_min, y_max)

  # Adjust layout with padding at the top for the suptitle
  plt.tight_layout(rect=[0, 0, 1, 0.96])
  plt.savefig(f'figures/yeo_network_barplots/yeo_network_influences_combined_{file_name}.png', dpi=300, bbox_inches='tight')
  print(f"\nSaved: figures/yeo_network_barplots/yeo_network_influences_combined_{file_name}.png")

if __name__ == "__main__":
  # 9 networks in the Yeo7 parcellation where Subcortex=8 and Cerebellum=9
  # Note THAT SC DOES NOT HAVE CEREBELLAR REGIONS (NETWORK 9)

  CONTROL_ONLY = False

  if CONTROL_ONLY:
    file_name = 'control'
  else:
    file_name = 'control_moderate'

  # Create the combined plot with all three conditions
  print(f"\nGenerating combined Yeo network barplot for {file_name}...")
  create_combined_yeo_barplot(file_name)
  print(f"Done!")