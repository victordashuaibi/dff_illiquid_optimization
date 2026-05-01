"""
Bias correction layer F_theta from DFF (Yang et al., AAAI 2025).
Implements Eq. 9-11: c_tilde = phi(x) * c_hat, with phi constrained
to [1-eps, 1+eps] via offset-scaled sigmoid.
"""
import torch
import torch.nn as nn


class BiasCorrectionLayer(nn.Module):
    """
    F_theta(x) = phi(x) * c_hat, where phi in [1-eps, 1+eps]
    Trust region: |c_tilde - c_hat| / |c_hat| <= eps  (Eq. 10)
    """

    def __init__(self, input_dim: int, output_dim: int, epsilon: float = 0.3,
                 hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.epsilon = epsilon

        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, output_dim)]
        self.h = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, c_hat: torch.Tensor) -> torch.Tensor:
        # phi(x) = (1 - eps) + 2*eps * sigmoid(h(x))   per Eq. 11
        phi = (1.0 - self.epsilon) + 2.0 * self.epsilon * torch.sigmoid(self.h(x))
        return phi * c_hat
