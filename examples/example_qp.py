"""
Example: Using dQP

This example demonstrates how to use dQP to solve a quadratic programming problem
and differentiate through the solution.

The QP problem solved is:
    minimize    1/2 x^T P x + q^T x
    subject to  Cx <= d
                Ax = b

"""

import torch
import numpy as np
from scipy.sparse import csc_matrix

# Import dQP components
from dqp import dQP
from dqp.sparse_helper import csc_scipy_to_torch

def qp_2d(qp_solver="osqp"):
    """
    minimize    x1^2 + x2^2
    subject to  x1 + x2 >= 1  (i.e., -x1 - x2 <= -1)
                x1 >= 0       (i.e., -x1 <= 0)
                x2 >= 0       (i.e., -x2 <= 0)
    
    The optimal solution is x* = [0.5, 0.5]
    """

    print("\n")
    print("=" * 60)
    print("Using qp solver: " + qp_solver)
    print("=" * 60)
    
    # --- Define the QP ---
    # Objective: 1/2 x^T P x + q^T x where P = 2*I
    P = csc_matrix(np.array([[2.0, 0.0], [0.0, 2.0]]))
    q = np.array([0.0, 0.0])
    # Inequality constraints: Cx <= d
    C = csc_matrix(np.array([
        [-1.0, -1.0],  # -x1 - x2 <= -1 (i.e., x1 + x2 >= 1)
        [-1.0, 0.0],   # -x1 <= 0 (i.e., x1 >= 0)
        [0.0, -1.0]    # -x2 <= 0 (i.e., x2 >= 0)
    ]))
    d = np.array([-1.0, 0.0, 0.0])
    
    # --- Convert to PyTorch tensors ---
    P_torch = csc_scipy_to_torch(P)
    q_torch = torch.tensor(q, dtype=torch.float64, requires_grad=True)
    C_torch = csc_scipy_to_torch(C)
    d_torch = torch.tensor(d, dtype=torch.float64, requires_grad=True)
    
    # --- Build dQP settings ---
    settings = dQP.build_settings(
        solve_type="sparse",     # Use sparse matrices
        qp_solver=qp_solver,     # QP solver for forward pass
        lin_solver="scipy SPLU", # Linear solver for backward pass
        verbose=False,
        empty_batch=False,       # Remove batch dimension
    )
    
    # --- Create the differentiable QP layer ---
    layer = dQP.dQP_layer(settings=settings)
    
    # --- Solve the QP ---
    # Returns: x* (primal), lambda* (eq dual), mu* (ineq dual), and profiling/timing
    x_star, lambda_star, mu_star, solve_time, total_time = layer(
        P_torch, q_torch, C_torch, d_torch, A=None, b=None
    )
    
    print(f"\nOptimal solution x* = {x_star.detach().numpy()}")
    print(f"Analytical solution x^* = [0.5, 0.5]")
    print(f"Inequality dual mu* = {mu_star.detach().numpy()}")
    
    # --- Form an example scalar loss L = sum_i x_i and backpropogate (differentiate) ---
    loss = x_star.sum()
    loss.backward()
    
    print(f"\n grad_d sum(x^*)  = {d_torch.grad.numpy()}")
    print(f" grad_q sum(x^*) = {q_torch.grad.numpy()}")
    
if __name__ == "__main__":
    qp_2d("osqp")
    qp_2d("piqp")
    qp_2d("scs")
    # qp_2d("gurobi") # requires license
