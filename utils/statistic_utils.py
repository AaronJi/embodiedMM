
import numpy as np

# vector_array: shape = [dim1, dim2],
# dim1: time index
# dim2: vector feature dims
def cal_stat(vetor_array):
    v_mean, v_std = np.mean(vetor_array, axis=0), np.std(vetor_array, axis=0) + 1e-6
    v_bounds = [np.min(vetor_array, axis=0), np.max(vetor_array, axis=0)]

    vector_array_normalized = (vetor_array - v_mean)/v_std
    v_normalized_bounds = [np.min(vector_array_normalized, axis=0), np.max(vector_array_normalized, axis=0)]
    return v_mean, v_std, v_bounds, v_normalized_bounds
