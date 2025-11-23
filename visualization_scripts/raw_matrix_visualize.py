from matplotlib import pyplot as plt
import numpy as np

from edges_heatmap import upper_tri_to_matrix

SC_data = np.mean(np.load('data/training_data/aligned/X_SC_control_moderate.npy'), axis=0)
FC_data = np.mean(np.load('data/training_data/aligned/X_FC_control_moderate.npy'), axis=0)
FCgsr_data = np.mean(np.load('data/training_data/aligned/X_FCgsr_control_moderate.npy'), axis=0)

def plot_matrix(matrix_data, matrix_type):
    print("Matrix shape:", matrix_data.shape)
    matrix_data = upper_tri_to_matrix(matrix_data, 90) if matrix_type == 'SC' else upper_tri_to_matrix(matrix_data, 109)

    plt.figure()
    plt.imshow(matrix_data)
    plt.title(f'{matrix_type} matrix')

    # Remove axis ticks and labels
    plt.xticks([])
    plt.yticks([])

    # plt.show()
    plt.savefig(f'figures/raw_matrices/{matrix_type}_matrix.png')

plot_matrix(SC_data, 'SC')
plot_matrix(FC_data, 'FC')
plot_matrix(FCgsr_data, 'FCgsr')