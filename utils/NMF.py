"""
-------------------------------------------------
# @Project  :state_space_disaggregation-main
# @File     :Non-negative_Matrix_Factorization
# @Date     :2024/12/30 14:23
# @Author   : ZS
-------------------------------------------------
"""
import numpy as np


def nmf_multiplicative_update(A, B, C, max_iter=500, tol=0.1):
    """
    Solve A = BC using multiplicative update rules with non-negative constraints.

    Parameters:
        A (ndarray): m x k matrix.
        B (ndarray): m x n matrix.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        C (ndarray): n x k matrix satisfying non-negativity constraints.
    """
    for iteration in range(max_iter):
        # Compute numerator and denominator for update
        numerator = B.T @ A
        denominator = (B.T @ B) @ C
        denominator = np.maximum(denominator, 1e-10)  # Avoid division by zero

        # Update C using multiplicative update rule
        C *= numerator / denominator
        C = C / np.sum(C, axis=1, keepdims=True)

        # Check convergence
        # error = np.linalg.norm(A - B @ C, 'fro')
        # a = np.abs(A - B @ C).sum(-1).mean()
        # if error < tol:
        #     print(f"Converged at iteration {iteration} with error {error:.6f}")
        #     break

    return C
