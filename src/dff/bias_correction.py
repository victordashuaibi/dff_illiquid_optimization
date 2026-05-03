"""
Bias correction layer F_theta from DFF (Yang et al., AAAI 2025).
Implements Eq. 9-11: c_tilde = phi(x) * c_hat, with phi constrained
to [1-eps, 1+eps] via offset-scaled sigmoid.

Two callable layers live here:

* :class:`BiasCorrectionLayer` — paper-faithful flat layer. Input
  ``x`` is a vector of features per instance; output ``c_tilde`` has
  the same shape as ``c_hat``. Used for the Wilder-2019 toy LP
  replication and as the building block for the per-asset wrapper.

* :class:`PerAssetBiasCorrectionLayer` — multi-asset cross-section
  wrapper used by the portfolio task (see ``docs/exp02_design.md``,
  decision D1b). Applies a *single* :class:`BiasCorrectionLayer` with
  ``output_dim=1`` row-by-row across the asset dimension, with
  per-asset NN input augmented by the scalar ``c_hat_i`` (D1b's
  per-asset feature vector + scalar c_hat encoding). Permutation-
  equivariant by construction.
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


class PerAssetBiasCorrectionLayer(nn.Module):
    """Per-asset shared-weight wrapper around :class:`BiasCorrectionLayer`.

    For a multi-asset cross-section, applies the *same* small NN to every
    asset row independently with shared weights, producing a per-asset
    multiplicative correction. The per-asset NN input is the concatenation
    of the asset's feature row with its scalar backbone prediction
    ``c_hat_i`` — see ``docs/exp02_design.md`` decision D1b.

    Shapes
    ------
    forward(X, c_hat):
        X     : ``(B, n_assets, n_features_per_asset)``
        c_hat : ``(B, n_assets)``
        c_tilde -> ``(B, n_assets)``

    Properties
    ----------
    * **Trust region**: elementwise ``|c_tilde - c_hat| / |c_hat| <= eps``
      because each asset row is one independent ``BiasCorrectionLayer``
      call.
    * **Permutation equivariance**: shuffling the ``n_assets`` axis of
      ``(X, c_hat)`` shuffles the output identically.
    * **Scales to any ``n_assets`` at eval time** (the inner NN is a
      fixed mapping over per-asset features; nothing depends on N).
    """

    def __init__(
        self,
        n_features_per_asset: int,
        epsilon: float = 0.5,
        hidden_dim: int = 32,
        n_layers: int = 3,
    ):
        super().__init__()
        # input_dim = features + 1 (the per-asset scalar c_hat).
        self._inner = BiasCorrectionLayer(
            input_dim=n_features_per_asset + 1,
            output_dim=1,
            epsilon=epsilon,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
        self.n_features_per_asset = int(n_features_per_asset)
        self.epsilon = float(epsilon)

    def forward(self, X: torch.Tensor, c_hat: torch.Tensor) -> torch.Tensor:
        if X.dim() != 3:
            raise ValueError(
                f"X must have shape (B, n_assets, n_features), got {tuple(X.shape)}"
            )
        if c_hat.dim() != 2:
            raise ValueError(
                f"c_hat must have shape (B, n_assets), got {tuple(c_hat.shape)}"
            )
        B, N, F = X.shape
        if c_hat.shape != (B, N):
            raise ValueError(
                f"c_hat shape {tuple(c_hat.shape)} != (B={B}, n_assets={N})"
            )
        if F != self.n_features_per_asset:
            raise ValueError(
                f"X feature dim {F} != configured n_features_per_asset "
                f"{self.n_features_per_asset}"
            )

        flat_X = X.reshape(B * N, F)
        flat_c_hat = c_hat.reshape(B * N, 1)
        # Augment per-asset NN input with the per-asset scalar c_hat (D1b).
        aug_input = torch.cat([flat_X, flat_c_hat], dim=-1)  # (B*N, F+1)
        flat_c_tilde = self._inner(aug_input, flat_c_hat)    # (B*N, 1)
        return flat_c_tilde.reshape(B, N)
