import argparse
import matplotlib.pyplot as plt
import torch

from nqe.data import (
    load_mnist_pca,
    load_fashion_mnist_pca,
    build_train_loader,
    build_validation_loader,
)
from nqe.embedding import QuantumEmbeddingLayer, ZZFeatureMap
from nqe.models import NQE
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
        "--pca-dim", type=int, default=8, help="Number of PCA components"
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
    return parser.parse_args()


def load_dataset(name: str, pca_dim: int):
    if name == "mnist":
        return load_mnist_pca(pca_dim)
    else:
        return load_fashion_mnist_pca(pca_dim)


def main():
    args = parse_args()

    X_train, Y_train, X_test, Y_test = load_dataset(args.dataset, args.pca_dim)

    q_layer = QuantumEmbeddingLayer(ZZFeatureMap, n_qubits=args.n_qubits)
    model = NQE(
        in_dims=args.pca_dim,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        hidden_dims=args.hidden_dims,
        q_embedding=q_layer,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = build_train_loader(
        X_train, Y_train, args.batch_size, n=args.max_steps + 1
    )
    val_loader = build_validation_loader(X_test, Y_test, args.batch_size)

    hist = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        optimizer,
        max_steps=args.max_steps,
        device=args.device,
        validate_every=args.validate_every,
        patience=args.patience,
        warm_up=args.warm_up,
        scheduler=not args.no_scheduler,
    )

    steps = range(len(hist["train_loss"]))
    val_steps = list(range(0, len(hist["train_loss"]), args.validate_every))[: len(hist["val_loss"])]

    plt.plot(steps, hist["train_loss"], label="train loss")
    plt.plot(val_steps, hist["val_loss"], label="val loss")
    plt.xlabel("training step")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Training plot saved to {args.output}")


if __name__ == "__main__":
    main()
