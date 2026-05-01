import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class ShortestPathLayer(nn.Module):
    def __init__(
        self,
        grid_size: int = 5,
        tau: float = 1e-3,
        solver_args: dict | None = None
    ):
        super().__init__()

        self.grid_size = grid_size
        self.tau = tau
        self.solver_args = solver_args or {"eps": 1e-6, "max_iters": 100000}

        self.edges, self.A, self.b = self._build_grid_lp(grid_size)
        self.n_edges = len(self.edges)

        flow = cp.Variable(self.n_edges)
        cost = cp.Parameter(self.n_edges)

        objective = cp.Minimize(
            cost @ flow + self.tau * cp.sum_squares(flow)
        )

        constraints = [
            self.A @ flow == self.b,
            flow >= 0,
            flow <= 1
        ]

        problem = cp.Problem(objective, constraints)

        if not problem.is_dpp():
            raise ValueError("Shortest path CVXPY problem is not DPP.")

        self.layer = CvxpyLayer(
            problem,
            parameters=[cost],
            variables=[flow]
        )

    def _node_id(self, i: int, j: int) -> int:
        return i * self.grid_size + j

    def _build_grid_lp(self, grid_size: int):
        edges = []

        for i in range(grid_size):
            for j in range(grid_size):
                current = self._node_id(i, j)

                if i + 1 < grid_size:
                    down = self._node_id(i + 1, j)
                    edges.append((current, down))

                if j + 1 < grid_size:
                    right = self._node_id(i, j + 1)
                    edges.append((current, right))

        n_nodes = grid_size * grid_size
        n_edges = len(edges)

        A = np.zeros((n_nodes, n_edges), dtype=np.float64)

        for edge_idx, (u, v) in enumerate(edges):
            A[u, edge_idx] = 1.0
            A[v, edge_idx] = -1.0

        b = np.zeros(n_nodes, dtype=np.float64)
        b[0] = 1.0
        b[n_nodes - 1] = -1.0

        return edges, A, b

    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        if cost.dim() != 2:
            raise ValueError("cost must have shape [batch, n_edges].")

        if cost.shape[1] != self.n_edges:
            raise ValueError("cost dimension does not match number of edges.")

        flow, = self.layer(cost, solver_args=self.solver_args)

        return flow