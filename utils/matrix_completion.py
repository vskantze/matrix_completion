import numpy as np

class MatrixHandler:
    def __init__(self,
                 matrix):
        self.matrix = matrix
        self.sparse_matrix = None

    def sparsify_matrix(self, sparsity = 0.5):
        num_nans = int( self.matrix.size*sparsity )
        idx_flat = np.random.choice(self.matrix.size, num_nans, replace=False)
        self.sparse_matrix = self.matrix[idx_flat] = np.nan

    def impute_matrix(self):
        imputed_matrix = self.sparse_matrix.copy()
        col_means = np.nanmean(imputed_matrix, axis=0)
        nan_mask = np.isnan(imputed_matrix)
        imputed_matrix[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        return imputed_matrix

    def matrix_completion(self, k = 4, iterations = 10):
        imputed_matrix = self.impute_matrix()
        reconstructed_matrix = imputed_matrix.copy()
        complete_matrix = self.sparse_matrix.copy()
        for i in range(iterations):
            U, S, Vt = np.linalg.svd(reconstructed_matrix)
            reconstructed_matrix = U[:,:k] @ np.diag(S[:k]) @ Vt[:k,:]
            nan_mask = np.isnan(self.sparse_matrix)
            complete_matrix[nan_mask] = reconstructed_matrix[nan_mask]









