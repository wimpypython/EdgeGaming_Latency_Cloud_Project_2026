import numpy as np

def calculate_spatial_divergence(true_coords, predicted_coords):
    # Calculates the Euclidean distance error between server state and AI prediction
    return np.linalg.norm(true_coords - predicted_coords)
