from typing import List, Optional, Literal
import torch
import torch.nn as nn
from itertools import pairwise
import pennylane as qml
from .embedding import QuantumEmbeddingLayer
from .utils import QFIMTracker


class Stacking(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.n_layers = n_layers

    def forward(self, x):
        return torch.stack([x for _ in range(self.n_layers)], dim=-1).reshape(x.size(0), -1)


class NQE(nn.Module):
    def __init__(self, in_dims: int, n_qubits: int, n_layers: int,
                 hidden_dims: int | List[int], q_embedding: QuantumEmbeddingLayer):
        super().__init__()
        self.in_dims = in_dims
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        self.q_embedding = q_embedding
        layers: List[nn.Module] = []
        layers.append(nn.Linear(in_dims, self.hidden_dims[0]))
        layers.append(nn.ReLU())
        for h1, h2 in pairwise(self.hidden_dims):
            layers.append(nn.Linear(h1, h2))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dims[-1], n_qubits))
        self.linear = nn.Sequential(*layers)
        self.stacking = Stacking(n_layers)
        self.c_layer = nn.Sequential(self.linear, self.stacking)
        self.q_layer = qml.qnn.TorchLayer(self.q_embedding._train_qnode, weight_shapes={})

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.linear(x1)
        x2 = self.linear(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.stacking(x)
        x = self.q_layer(x)
        return x[:, 0]


NQE_repeat = NQE


class NQE_BIG(nn.Module):
    def __init__(self, in_dims: int, n_qubits: int, n_layers: int,
                 hidden_dims: Optional[int | List[int]], q_embedding: QuantumEmbeddingLayer):
        super().__init__()
        self.in_dims = in_dims
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        self.q_embedding = q_embedding
        layers: List[nn.Module] = []
        layers.append(nn.Linear(in_dims, self.hidden_dims[0]))
        layers.append(nn.ReLU())
        for h1, h2 in pairwise(self.hidden_dims):
            layers.append(nn.Linear(h1, h2))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dims[-1], n_qubits * n_layers))
        self.linear = nn.Sequential(*layers)
        self.c_layer = nn.Sequential(self.linear)
        self.q_layer = qml.qnn.TorchLayer(self.q_embedding._train_qnode, weight_shapes={})

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.linear(x1).reshape(-1, self.n_qubits, self.n_layers)
        x2 = self.linear(x2).reshape(-1, self.n_qubits, self.n_layers)
        x = torch.cat([x1, x2], dim=1).reshape(x1.size(0), -1)
        x = self.q_layer(x)
        return x[:, 0]


class UpConvolution1(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, m))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(2)
        w = self.weight.unsqueeze(0)
        return (x * w).reshape(x.size(0), -1)


class UpConvolution2(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, m))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(2)
        w = self.weight.unsqueeze(0)
        return (x * w).reshape(x.size(0), -1)


class UCNQE(nn.Module):
    def __init__(self, in_dims: int, n_qubits: int, n_layers: int,
                 hidden_dims: Optional[int | List[int]], q_embedding: QuantumEmbeddingLayer,
                 mode: Literal['single', 'block']):
        super().__init__()
        self.in_dims = in_dims
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        self.q_embedding = q_embedding
        layers: List[nn.Module] = []
        layers.append(nn.Linear(in_dims, self.hidden_dims[0]))
        layers.append(nn.ReLU())
        for h1, h2 in pairwise(self.hidden_dims):
            layers.append(nn.Linear(h1, h2))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dims[-1], n_qubits))
        layers.append(nn.ReLU())
        self.linear = nn.Sequential(*layers)
        if mode == 'single':
            self.up_conv = UpConvolution1(n_layers)
        else:
            self.up_conv = UpConvolution2(n_qubits, n_layers)
        self.c_layer = nn.Sequential(self.linear, self.up_conv)
        self.q_layer = qml.qnn.TorchLayer(self.q_embedding._train_qnode, weight_shapes={})

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up_conv(self.linear(x1))
        x2 = self.up_conv(self.linear(x2))
        x = torch.cat([x1, x2], dim=1)
        x = self.q_layer(x)
        return x[:, 0]


def SU2(params, wires):
    qml.RZ(params[0], wires=wires)
    qml.RY(params[1], wires=wires)
    qml.RZ(params[2], wires=wires)


def SU4(params, wires, *, cartan_sign='ising'):
    if len(wires) != 2:
        raise ValueError("SU4 requires exactly two wires.")
    if qml.math.size(params) != 15:
        raise ValueError("SU4 expects 15 parameters.")
    p = qml.math.asarray(params)
    L0, L1 = p[0:3], p[3:6]
    C = p[6:9]
    R0, R1 = p[9:12], p[12:15]
    SU2(L0, wires=wires[0])
    SU2(L1, wires=wires[1])
    if cartan_sign == 'canonical':
        cx, cy, cz = (-2*C[0], -2*C[1], -2*C[2])
    elif cartan_sign == 'ising':
        cx, cy, cz = (C[0], C[1], C[2])
    else:
        raise ValueError("cartan_sign must be 'ising' or 'canonical'.")
    qml.IsingXX(cx, wires=wires)
    qml.IsingYY(cy, wires=wires)
    qml.IsingZZ(cz, wires=wires)
    SU2(R0, wires=wires[0])
    SU2(R1, wires=wires[1])


class Ansatz:
    def __init__(self, n_qubits, mode):
        self.n_qubits = n_qubits
        self.mode = mode
        if mode == 'SU4':
            g1 = n_qubits // 2
            g2 = (n_qubits - 1) // 2
            self._layout = ([(0 + 2*i, 1 + 2*i) for i in range(g1)] +
                            [(1 + 2*i, 2 + 2*i) for i in range(g2)])
            self.num_ansatz_params = 15 * (g1 + g2)
        elif mode == 'TTN':
            self.num_ansatz_params = max(0, n_qubits - 1) * 15 + 3
        else:
            raise ValueError(f"Unknown ansatz mode: {mode}")

    def apply(self, params):
        if self.mode == 'SU4':
            p = 0
            for (a, b) in self._layout:
                SU4(params[p:p+15], wires=[a, b])
                p += 15
            SU2(params[p:p+3], wires=[0])
        else:
            wires = list(range(self.n_qubits))
            p = 0
            while len(wires) > 1:
                new_wires = []
                for i in range(0, len(wires) - 1, 2):
                    SU4(params[p:p+15], wires=[wires[i], wires[i+1]])
                    p += 15
                    new_wires.append(wires[i])
                if len(wires) % 2 != 0:
                    new_wires.append(wires[-1])
                wires = new_wires
            SU2(params[p:p+3], wires=[0])


@torch.no_grad()
def _to_layers_from_nqe(nqe, x):
    if hasattr(nqe, 'up_conv'):
        nqe.c_layer.eval()
        return nqe.c_layer(x)
    elif hasattr(nqe, 'stacking'):
        nqe.c_layer.eval()
        return nqe.c_layer(x)
    else:
        nqe.linear.eval()
        return nqe.linear(x)


class QCNN(nn.Module):
    def __init__(self, n_qubits, nqe, ansatz_mode):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = nqe.n_layers
        self.nqe = nqe
        self.embedding = nqe.q_embedding._embedding
        self.ansatz = Ansatz(n_qubits, ansatz_mode)
        self.eps = 1e-6
        self.params = nn.Parameter(torch.zeros(self.ansatz.num_ansatz_params))

        @qml.qnode(device=nqe.q_embedding.dev, interface='torch')
        def qc_expval(inputs, params):
            self.embedding(inputs)
            self.ansatz.apply(params)
            return qml.expval(qml.PauliZ(0))
        self.qc = qc_expval

        @qml.qnode(device=nqe.q_embedding.dev, interface='torch')
        def qc_dm(inputs, params):
            self.embedding(inputs)
            self.ansatz.apply(params)
            return qml.density_matrix(0)
        self.qc_dm = qc_dm

    def forward(self, x, for_visual=False):
        qfeat = _to_layers_from_nqe(self.nqe, x)
        if for_visual:
            return self.qc_dm(qfeat, self.params)
        y = self.qc(qfeat, self.params)
        y = (y + 1.0) * 0.5
        return torch.clamp(y, self.eps, 1.0 - self.eps)
