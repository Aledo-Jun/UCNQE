import argparse
import matplotlib.pyplot as plt
import torch
from typing import Optional

from nqe.data import (
    build_train_loader,
    build_validation_loader,
    load_fashion_mnist_pca,
    load_mnist_pca,
)
from nqe.embedding import QuantumEmbeddingLayer, ZZFeatureMap
from nqe.models import NQE, NQE_BIG, UCNQE
from nqe.training import train_with_early_stopping


def parse_args():
    parser = argparse.ArgumentParser(description="Train an NQE model and plot results")
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashion"],
        default="mnist",
        help="Dataset to use (mnist or fashion)",
    )
    parser.add_argument(
        "--model",
        choices=["nqe", "nqe_big", "ucnqe"],
        default="nqe",
        help="Model variant to train",
    )
    parser.add_argument(
        "--pca-dim", type=int, default=8, help="Number of PCA components"
    )
    parser.add_argument(
        "--uc-mode",
        choices=["single", "block"],
        default="single",
        help="UCNQE up-convolution mode",
    )
    parser.add_argument(
        "--n-qubits", type=int, default=4, help="Number of qubits in the quantum layer"
    )
    parser.add_argument(
        "--n-layers", type=int, default=2, help="Number of embedding layers"
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[16, 16],
        help="Dimensions of hidden classical layers",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="Maximum training steps"
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=25,
        help="Validate every N training steps",
    )
    parser.add_argument(
        "--patience", type=int, default=300, help="Early stopping patience"
    )
    parser.add_argument(
        "--warm-up", type=int, default=100, help="Early stopping warmup"
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Disable learning rate scheduler",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device",
    )
    parser.add_argument(
        "--output",
        default="training.png",
        help="Where to save the training plot",
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to save final model weights (.pt)",
    )
    return parser.parse_args()


def load_dataset(name: str, pca_dim: int):
    if name == "mnist":
        return load_mnist_pca(pca_dim)
    else:
        return load_fashion_mnist_pca(pca_dim)


def train_model(
    *,
    dataset: str = "mnist",
    model: str = "nqe",
    pca_dim: int = 8,
    uc_mode: str = "single",
    n_qubits: int = 4,
    n_layers: int = 2,
    hidden_dims=None,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_steps: int = 1000,
    validate_every: int = 25,
    patience: int = 300,
    warm_up: int = 100,
    use_scheduler: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output: str = "training.png",
    checkpoint: Optional[str] = None,
):
    """Train a Neural Quantum Embedding model and save loss plots.

    Parameters mirror the command line options exposed by ``train_nqe.py``.
    ``hidden_dims`` defaults to ``[16, 16]`` when not provided.  The training
    history dictionary returned by ``train_with_early_stopping`` is returned
    along with the trained model instance.  If ``checkpoint`` is supplied, the
    model's ``state_dict`` is saved there in PyTorch's ``.pt`` format.
    """

    if hidden_dims is None:
        hidden_dims = [16, 16]

    X_train, Y_train, X_test, Y_test = load_dataset(dataset, pca_dim)

    q_layer = QuantumEmbeddingLayer(ZZFeatureMap, n_qubits=n_qubits)
    model_kwargs = {
        "in_dims": pca_dim,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "hidden_dims": hidden_dims,
        "q_embedding": q_layer,
    }
    if model == "nqe":
        model_cls = NQE
    elif model == "nqe_big":
        model_cls = NQE_BIG
    else:
        model_cls = UCNQE
        model_kwargs["mode"] = uc_mode
    model_inst = model_cls(**model_kwargs)

    optimizer = torch.optim.Adam(model_inst.parameters(), lr=lr)
    train_loader = build_train_loader(
        X_train, Y_train, batch_size, n=max_steps + 1
    )
    val_loader = build_validation_loader(X_test, Y_test, batch_size)

    hist = train_with_early_stopping(
        model_inst,
        train_loader,
        val_loader,
        optimizer,
        max_steps=max_steps,
        device=device,
        validate_every=validate_every,
        patience=patience,
        warm_up=warm_up,
        scheduler=use_scheduler,
    )

    steps = range(len(hist["train_loss"]))
    val_steps = list(range(0, len(hist["train_loss"]), validate_every))[: len(hist["val_loss"])]

    plt.figure()
    plt.plot(steps, hist["train_loss"], label="train loss")
    plt.plot(val_steps, hist["val_loss"], label="val loss")
    plt.xlabel("training step")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print(f"Training plot saved to {output}")

    if checkpoint:
        torch.save(model_inst.state_dict(), checkpoint)
        print(f"Model checkpoint saved to {checkpoint}")

    return hist, model_inst


def main():
    args = parse_args()
    train_model(
        dataset=args.dataset,
        model=args.model,
        pca_dim=args.pca_dim,
        uc_mode=args.uc_mode,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        hidden_dims=args.hidden_dims,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        validate_every=args.validate_every,
        patience=args.patience,
        warm_up=args.warm_up,
        use_scheduler=not args.no_scheduler,
        device=args.device,
        output=args.output,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
