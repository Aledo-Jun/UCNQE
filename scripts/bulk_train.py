import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from train_nqe import train_model, load_dataset
from nqe.utils import Metric


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multiple NQE variants across a range of layer counts"
    )
    parser.add_argument(
        "--dataset", choices=["mnist", "fashion"], default="mnist",
        help="Dataset to use for all runs",
    )
    parser.add_argument("--pca-dim", type=int, default=8, help="Number of PCA components")
    parser.add_argument("--n-qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=[16, 16],
        help="Dimensions of hidden classical layers",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument(
        "--validate-every", type=int, default=25,
        help="Validate every N training steps",
    )
    parser.add_argument("--patience", type=int, default=300, help="Early stopping patience")
    parser.add_argument("--warm-up", type=int, default=100, help="Early stopping warmup")
    parser.add_argument(
        "--no-scheduler", action="store_true", help="Disable learning rate scheduler"
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[1, 2, 3, 4],
        help="List of layer counts to train",
    )
    parser.add_argument(
        "--models", nargs="+", default=["nqe", "nqe_big", "ucnqe"],
        choices=["nqe", "nqe_big", "ucnqe"],
        help="Model variants to train",
    )
    parser.add_argument(
        "--uc-modes", nargs="+", default=["single", "block"],
        choices=["single", "block"],
        help="UCNQE up-convolution modes to sweep",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device",
    )
    parser.add_argument(
        "--output-dir", default="bulk_training", help="Directory to store plots",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory to store model checkpoints",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Preload test data for metric evaluation
    _, _, X_test, Y_test = load_dataset(args.dataset, args.pca_dim)
    device = torch.device(args.device)
    X_test_t = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    Y_test_t = torch.as_tensor(Y_test, dtype=torch.float32, device=device)
    idx0 = np.where(Y_test == 0)[0]
    idx1 = np.where(Y_test == 1)[0]
    X0_t = torch.as_tensor(X_test[idx0], dtype=torch.float32, device=device)
    X1_t = torch.as_tensor(X_test[idx1], dtype=torch.float32, device=device)

    loss_results = {}
    tr_results = {}
    hs_results = {}
    qka_results = {}

    for model in args.models:
        modes = args.uc_modes if model == "ucnqe" else ["single"]
        for uc_mode, n_layers in product(modes, args.layers):
            suffix = (
                f"{model}_{uc_mode}_{n_layers}layers"
                if model == "ucnqe"
                else f"{model}_{n_layers}layers"
            )
            output = out_dir / f"{suffix}.png"
            checkpoint = str(ckpt_dir / f"{suffix}.pt") if ckpt_dir else None
            hist, mdl = train_model(
                dataset=args.dataset,
                model=model,
                pca_dim=args.pca_dim,
                uc_mode=uc_mode,
                n_qubits=args.n_qubits,
                n_layers=n_layers,
                hidden_dims=args.hidden_dims,
                batch_size=args.batch_size,
                lr=args.lr,
                max_steps=args.max_steps,
                validate_every=args.validate_every,
                patience=args.patience,
                warm_up=args.warm_up,
                use_scheduler=not args.no_scheduler,
                device=args.device,
                output=str(output),
                checkpoint=checkpoint,
            )
            label = model.upper() if model != "ucnqe" else f"UCNQE-{uc_mode}"
            metric = Metric(mdl)
            tr = float(metric.separability(X0_t, X1_t, "Tr", trained=True).item())
            hs = float(metric.separability(X0_t, X1_t, "HS", trained=True).item())
            qka = metric.centered_kernel_alignment(X_test_t, Y_test_t, trained=True)
            loss_results.setdefault(label, []).append((n_layers, min(hist["val_loss"])))
            tr_results.setdefault(label, []).append((n_layers, tr))
            hs_results.setdefault(label, []).append((n_layers, hs))
            qka_results.setdefault(label, []).append((n_layers, qka))
            print(
                f"Finished {suffix} (Tr={tr:.6f}, HS={hs:.6f}, QKA={qka:.6f})"
            )

    def plot_metric(res_map, ylabel, filename):
        if not res_map:
            return
        plt.figure()
        for label, data in res_map.items():
            data.sort(key=lambda x: x[0])
            layers = [n for n, _ in data]
            vals = [v for _, v in data]
            plt.plot(layers, vals, marker="o", label=label)
        plt.xlabel("embedding layers")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        path = out_dir / filename
        plt.savefig(path)
        plt.close()
        print(f"{ylabel} plot saved to {path}")

    plot_metric(loss_results, "best validation loss", "comparison.png")
    plot_metric(tr_results, "trace distance", "trace_distance.png")
    plot_metric(hs_results, "HS distance", "hs_distance.png")
    plot_metric(qka_results, "quantum kernel alignment", "qka.png")


if __name__ == "__main__":
    main()
