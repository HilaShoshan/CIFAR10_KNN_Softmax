import numpy as np


def predict_labels(dists, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        dist_arr = dists[i, :]
        print("dist_arr: ", dist_arr)
        k_min_idx = np.argpartition(dist_arr, k)[:k]  # k indices of the smallest distances from i
        print("k_min_idx: ", k_min_idx)
    return y_pred


dists = np.array([[0, 3, 6, 9],
                 [1, 4, 6, 8],
                 [6, 7, 8, 9]])

predict_labels(dists)

