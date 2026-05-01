import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.optimizer.toy_lp import ShortestPathLayer


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


class CostPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softplus()
        )

    def forward(self, x):
        return self.net(x) + 0.05


def generate_synthetic_data(
    n_samples: int,
    input_dim: int,
    n_edges: int,
    seed: int = 42
):
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(n_samples, input_dim))
    W = rng.normal(scale=0.5, size=(input_dim, n_edges))

    raw_cost = X @ W
    raw_cost += 0.3 * np.sin(X @ W)
    raw_cost += rng.normal(scale=0.1, size=raw_cost.shape)

    cost = np.log1p(np.exp(raw_cost)) + 0.1

    X = torch.tensor(X, dtype=torch.float64)
    cost = torch.tensor(cost, dtype=torch.float64)

    return X, cost


def decision_regret(cost_true, flow_pred, flow_true):
    pred_obj = torch.sum(cost_true * flow_pred, dim=1)
    true_obj = torch.sum(cost_true * flow_true, dim=1)
    regret = pred_obj - true_obj
    return regret


def normalized_decision_regret(cost_true, flow_pred, flow_true):
    regret = decision_regret(cost_true, flow_pred, flow_true)
    true_obj = torch.sum(cost_true * flow_true, dim=1)
    ndr = regret.sum() / torch.abs(true_obj).sum()
    return ndr


def train_two_stage(model, X_train, c_train, epochs=100, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        pred = model(X_train)
        loss = loss_fn(pred, c_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == 1:
            print(f"Two-stage epoch {epoch}: MSE={loss.item():.6f}")

    return model


def train_dfl(
    model,
    sp_layer,
    X_train,
    c_train,
    flow_true_train,
    epochs=250,
    lr=5e-4,
    lambda_mse=0.02
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        c_pred = model(X_train)
        flow_pred = sp_layer(c_pred)

        regret = decision_regret(c_train, flow_pred, flow_true_train).mean()
        mse = mse_fn(c_pred, c_train)
        loss = regret + lambda_mse * mse

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            ndr = normalized_decision_regret(
                c_train,
                flow_pred.detach(),
                flow_true_train
            )
            print(
                f"DFL epoch {epoch}: "
                f"loss={loss.item():.6f}, "
                f"regret={regret.item():.6f}, "
                f"NDR={ndr.item():.6f}"
            )

    return model


def evaluate_model(model, sp_layer, X, c_true, flow_true, name):
    with torch.no_grad():
        c_pred = model(X)
        flow_pred = sp_layer(c_pred)

        mse = torch.mean((c_pred - c_true) ** 2).item()
        ndr = normalized_decision_regret(
            c_true,
            flow_pred,
            flow_true
        ).item()

    print(f"{name} MSE: {mse:.6f}")
    print(f"{name} NDR: {ndr:.6f}")

    return {
        "Model": name,
        "MSE": mse,
        "NDR": ndr
    }


def main():
    set_seed(42)

    grid_size = 5
    input_dim = 20
    n_samples = 1000
    train_ratio = 0.8

    sp_layer = ShortestPathLayer(grid_size=grid_size).double()
    n_edges = sp_layer.n_edges

    X, c_true = generate_synthetic_data(
        n_samples=n_samples,
        input_dim=input_dim,
        n_edges=n_edges,
        seed=42
    )

    n_train = int(n_samples * train_ratio)

    X_train = X[:n_train]
    c_train = c_true[:n_train]

    X_test = X[n_train:]
    c_test = c_true[n_train:]

    with torch.no_grad():
        flow_true_train = sp_layer(c_train)
        flow_true_test = sp_layer(c_test)

    two_stage_model = CostPredictor(
        input_dim=input_dim,
        output_dim=n_edges
    ).double()

    two_stage_model = train_two_stage(
        model=two_stage_model,
        X_train=X_train,
        c_train=c_train,
        epochs=100,
        lr=1e-3
    )

    dfl_model = copy.deepcopy(two_stage_model)

    dfl_model = train_dfl(
        model=dfl_model,
        sp_layer=sp_layer,
        X_train=X_train,
        c_train=c_train,
        flow_true_train=flow_true_train,
        epochs=250,
        lr=5e-4,
        lambda_mse=0.02
    )

    print("\nFinal evaluation on test set")
    print("-" * 60)

    result_two_stage = evaluate_model(
        model=two_stage_model,
        sp_layer=sp_layer,
        X=X_test,
        c_true=c_test,
        flow_true=flow_true_test,
        name="Two-stage"
    )

    result_dfl = evaluate_model(
        model=dfl_model,
        sp_layer=sp_layer,
        X=X_test,
        c_true=c_test,
        flow_true=flow_true_test,
        name="DFL"
    )

    print("-" * 60)

    if result_dfl["NDR"] < result_two_stage["NDR"]:
        print("Result: DFL improves NDR over Two-stage.")
    else:
        print("Result: DFL did not improve NDR in this run.")

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "toy_dfl_results.csv"

    with open(result_path, "w") as f:
        f.write("Model,MSE,NDR\n")
        f.write(
            f"{result_two_stage['Model']},"
            f"{result_two_stage['MSE']},"
            f"{result_two_stage['NDR']}\n"
        )
        f.write(
            f"{result_dfl['Model']},"
            f"{result_dfl['MSE']},"
            f"{result_dfl['NDR']}\n"
        )

    print(f"Saved results to: {result_path}")


if __name__ == "__main__":
    main()