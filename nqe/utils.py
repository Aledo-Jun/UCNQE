import numpy
from pennylane import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import mpld3
from mpld3 import plugins
from typing import Literal
import pennylane as qml

# Disable automatic JS injection on import.
mpld3.disable_notebook()


def interactive_legend_for_fig(fig, *, pair_fill=True, alpha_unsel=0.2,
                               alpha_over=1.5, start_visible=True):
    """Attach an interactive legend to each axis in ``fig``."""
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        lines = [ln for ln in ax.get_lines() if ln.get_label() != "_nolegend_"]
        labels = [ln.get_label() or f"line-{i}" for i, ln in enumerate(lines)]
        groups = []
        if pair_fill:
            fills = [c for c in ax.collections if isinstance(c, PolyCollection)]
            for i, ln in enumerate(lines):
                f = fills[i] if i < len(fills) else None
                groups.append([a for a in (ln, f) if a is not None])
        else:
            groups = [[ln] for ln in lines]
        if groups:
            plugins.connect(
                fig,
                plugins.InteractiveLegendPlugin(
                    groups, labels,
                    alpha_unsel=alpha_unsel,
                    alpha_over=alpha_over,
                    start_visible=start_visible,
                    ax=ax,
                ),
            )
    return mpld3.display(fig)


def _rolling_mean(x, k):
    if len(x) == 0:
        return x
    k = max(1, min(k, len(x)))
    c = numpy.convolve(x, np.ones(k) / k, mode="valid")
    pad = np.full(k - 1, c[0])
    return np.concatenate([pad, c])


def _robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad)


def detect_anomalies(train_hist, val_hist, val_steps, *, rel_div_factor=1.8,
                     spike_z=8.0, plateau_windows=10, overfit_tail=6):
    out = {"notes": [], "diverge_idx": [], "spike_idx": [],
           "nan_inf": False, "plateau": False, "overfitting": False,
           "best_idx": None}
    v = np.asarray(val_hist, dtype=float)
    t = np.asarray(train_hist, dtype=float)
    if len(v) == 0:
        out["notes"].append("No validation points recorded.")
        return out
    out["nan_inf"] = (not np.all(np.isfinite(v))) or (not np.all(np.isfinite(t)))
    if out["nan_inf"]:
        out["notes"].append("NaN/Inf detected in loss history.")
    prev_best = numpy.minimum.accumulate(v[:-1]) if len(v) > 1 else numpy.array([v[0]])
    ratios = np.empty_like(v); ratios[:] = np.nan
    if len(v) > 1:
        ratios[1:] = v[1:] / numpy.maximum(prev_best, 1e-12)
        rising = numpy.r_[False, np.diff(v) > 0]
        div_idx = np.where((ratios > rel_div_factor) & rising)[0].tolist()
        out["diverge_idx"] = div_idx
        if div_idx:
            out["notes"].append(
                f"Divergence spikes at val iters {div_idx} (x>{rel_div_factor:.1f} of prior best)."
            )
    if len(v) > 2:
        dz = _robust_z(np.diff(v))
        spike_idx = np.where(np.abs(dz) > spike_z)[0] + 1
        out["spike_idx"] = spike_idx.tolist()
        if len(spike_idx):
            out["notes"].append(
                f"Instability spikes at val iters {out['spike_idx']} (|z|>{spike_z})."
            )
    best_idx = int(numpy.argmin(v))
    out["best_idx"] = best_idx
    since_best = (len(v) - 1) - best_idx
    if since_best >= plateau_windows:
        out["plateau"] = True
        out["notes"].append(f"Plateau: no improvement for {since_best} validations.")
    tail = min(overfit_tail, len(v))
    if tail >= 3:
        seg_means = []
        for k in range(len(v)):
            left = 0 if k == 0 else val_steps[k-1]
            right = val_steps[k] if k < len(val_steps) else len(t)-1
            seg = t[left:right+1] if right >= left else t[max(0, right-5):right+1]
            seg_means.append(np.mean(seg) if len(seg) else np.nan)
        seg_means = np.array(seg_means, dtype=float)
        xv = np.arange(len(v))
        sv = numpy.polyfit(xv[-tail:], v[-tail:], 1)[0]
        st = numpy.polyfit(xv[-tail:], seg_means[-tail:], 1)[0] if np.all(np.isfinite(seg_means[-tail:])) else 0.0
        if sv > 0 and st < 0:
            out["overfitting"] = True
            out["notes"].append(
                f"Overfitting signature on last {tail} validations (val↑, train↓)."
            )
    if not out["notes"]:
        out["notes"].append("No anomalies detected.")
    return out


@torch.no_grad()
def _dm_batch_from_inputs(model, Xb: torch.Tensor) -> np.ndarray:
    model.eval()
    qfeat = model.c_layer(Xb)
    rho = model.q_embedding._dm_qnode(qfeat)
    return rho.detach().cpu().numpy()


def _bloch_from_dm(dm_batch: np.ndarray):
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    xs = np.real(np.trace(dm_batch @ sx, axis1=1, axis2=2))
    ys = np.real(np.trace(dm_batch @ sy, axis1=1, axis2=2))
    zs = np.real(np.trace(dm_batch @ sz, axis1=1, axis2=2))
    return xs, ys, zs


def _plot_bloch_points(x0, y0, z0, x1, y1, z1, *, title="Bloch sphere: class clusters"):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, rstride=2, cstride=2, alpha=0.08, linewidth=0)
    ax.plot([-1,1],[0,0],[0,0], alpha=0.3, lw=1)
    ax.plot([0,0],[-1,1],[0,0], alpha=0.3, lw=1)
    ax.plot([0,0],[0,0],[-1,1], alpha=0.3, lw=1)
    ax.scatter(x0,y0,z0, s=20, label="class 0", depthshade=False, marker='o')
    ax.scatter(x1,y1,z1, s=20, label="class 1", depthshade=False, marker='s')
    m0 = np.array([x0.mean(), y0.mean(), z0.mean()])
    m1 = np.array([x1.mean(), y1.mean(), z1.mean()])
    ax.scatter(m0[0], m0[1], m0[2], label='mean class 0', depthshade=False, marker='o', c='b')
    ax.scatter(m1[0], m1[1], m1[2], label='mean class 1', depthshade=False, marker='s', c='r')
    ax.quiver(0,0,0, m0[0],m0[1],m0[2], normalize=False, colors='b')
    ax.quiver(0,0,0, m1[0],m1[1],m1[2], normalize=False, colors='r')
    ax.set_box_aspect((1,1,1))
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc='upper left')
    plt.show()


class Metric(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.c_layer = model.c_layer
        self.q_layer = model.q_embedding._dm_qnode

    def separability(self, x0, x1, mode: Literal['Tr','HS'], trained: bool):
        if trained:
            self.c_layer.eval()
            x0 = self.c_layer(x0)
            x1 = self.c_layer(x1)
        rhos0 = self.q_layer(x0)
        rhos1 = self.q_layer(x1)
        rho0 = torch.mean(rhos0, dim=0)
        rho1 = torch.mean(rhos1, dim=0)
        diff = rho1 - rho0
        if mode == 'Tr':
            eigvals = torch.linalg.eigvalsh(diff)
            return 0.5 * torch.real(torch.sum(torch.abs(eigvals)))
        if mode == 'HS':
            return 0.5 * torch.real(torch.trace(diff @ diff))
        raise ValueError(f"Invalid mode: {mode}")

    def centered_kernel_alignment(self, x, y, trained: bool):
        if trained:
            self.c_layer.eval()
            x = self.c_layer(x)
        rhos = self.q_layer(x)
        B = rhos.size(0)
        K = torch.real(torch.einsum('bij,cji->bc', rhos, rhos)).to(torch.float32)
        H = torch.eye(B, device=K.device) - torch.ones((B,B), device=K.device) / B
        Kc = H @ K @ H
        Ky = torch.outer(y, y)
        Ky_c = H @ Ky @ H
        num = torch.sum(Kc*Ky_c)
        den = torch.linalg.norm(Kc, 'fro') * torch.linalg.norm(Ky_c, 'fro')
        return float((num/den).item())


class QFIMTracker:
    """Compute the quantum Fisher information matrix for QCNN parameters."""
    def __init__(self, nqe, ansatz, dev):
        self.nqe = nqe
        self.ansatz = ansatz
        self.dev = dev

        @qml.qnode(device=dev, interface='torch')
        def _state_qnode(inputs, params):
            nqe.q_embedding._embedding(inputs)
            ansatz.apply(params)
            return qml.state()
        self._state_qnode = _state_qnode
        self._mt_fn = qml.metric_tensor(self._state_qnode)

    @torch.no_grad()
    def qfim(self, Xb: torch.Tensor, params: torch.Tensor, max_samples: int | None = 8) -> torch.Tensor:
        if max_samples is None:
            max_samples = Xb.shape[0]
        m = min(int(max_samples), int(Xb.shape[0]))
        Fs, ok = [], True
        for i in range(m):
            xi = Xb[i:i+1]
            try:
                g = self._mt_fn(xi.cpu(), params.detach().cpu())
                g = torch.as_tensor(g, dtype=torch.get_default_dtype())
                g = 0.5 * (g + g.T)
                Fs.append(4.0 * g)
            except Exception:
                ok = False
                break
        if ok and Fs:
            return torch.stack(Fs, dim=0).mean(dim=0)
        raise RuntimeError("QFIM not available (likely noisy/mixed/shots device).")

    @staticmethod
    @torch.no_grad()
    def trainability_from_qfim(F: torch.Tensor) -> dict:
        F = 0.5 * (F + F.T)
        eps = 1e-12
        evals = torch.linalg.eigvalsh(F + eps*torch.eye(F.shape[0], dtype=F.dtype))
        trace = torch.clamp(evals, min=0).sum().item()
        lam_min = float(evals.min().item())
        lam_max = float(evals.max().item())
        cond = float(lam_max / max(lam_min, eps))
        return {"trace": trace,
                "lambda_min": lam_min,
                "lambda_max": lam_max,
                "condition_number": cond}
