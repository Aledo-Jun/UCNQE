import pennylane as qml
from pennylane import numpy as np


def ZZFeatureMap(features, *, wires, rot_factor=2.0, ent_factor=2.0):
    """PennyLane implementation of Qiskit's ZZFeatureMap."""
    wires = qml.wires.Wires(wires)
    n_qubits = len(wires)
    f = qml.math.reshape(features, (-1, n_qubits))
    for w in wires:
        qml.H(w)
    for k, w in enumerate(wires):
        qml.RZ(rot_factor * f[..., k], wires=w)
    if n_qubits == 1:
        return
    for i in range(n_qubits):
        j = (i + 1) % n_qubits
        angle = ent_factor * (np.pi - f[..., i]) * (np.pi - f[..., j])
        qml.CNOT(wires=[wires[i], wires[j]])
        qml.RZ(angle, wires=wires[j])
        qml.CNOT(wires=[wires[i], wires[j]])


def make_qml_device(n_wires, shots=None):
    """Create a PennyLane device with a sensible fallback.

    The function prefers the fast ``lightning.qubit`` backend when
    available and falls back to the standard ``default.qubit`` device if
    the former cannot be instantiated.

    Parameters
    ----------
    n_wires:
        Number of quantum wires the device should support.
    shots:
        Optional number of measurement shots. ``None`` denotes analytic
        mode.

    Returns
    -------
    pennylane.Device
        Instantiated PennyLane device ready for use.
    """
    try:
        return qml.device("lightning.qubit", wires=n_wires, shots=shots)
    except Exception:
        return qml.device("default.qubit", wires=n_wires, shots=shots)


class QuantumEmbeddingLayer:
    """Wrap a feature embedding circuit in PennyLane QNodes.

    The layer exposes QNodes for training with probability outputs and for
    generating density matrices.  A helper ``_embedding`` method applies the
    embedding circuit without returning a value, which is useful when composing
    with additional circuits.
    """

    def __init__(self, embedding, *, n_qubits, dev=None):
        self.embedding = embedding
        self.n_qubits = n_qubits
        self.dev = dev if dev is not None else make_qml_device(n_qubits)

        def _train(inputs):
            x = inputs.view(inputs.size(0), 2 * n_qubits, -1)
            B, _, n_layers = x.shape
            x1, x2 = x[:, :n_qubits, :], x[:, n_qubits:, :]
            for i in range(n_layers):
                self.embedding(
                    x1[:, :, i], wires=range(n_qubits),
                    rot_factor=2.0 / n_layers, ent_factor=2.0 / n_layers,
                )
            for i in reversed(range(n_layers)):
                qml.adjoint(self.embedding)(
                    x2[:, :, i], wires=range(n_qubits),
                    rot_factor=2.0 / n_layers, ent_factor=2.0 / n_layers,
                )
            return qml.probs(wires=range(n_qubits))

        def _dm(inputs):
            x = inputs.view(inputs.size(0), n_qubits, -1)
            B, _, n_layers = x.shape
            for i in range(n_layers):
                self.embedding(
                    x[:, :, i], wires=range(n_qubits),
                    rot_factor=2.0 / n_layers, ent_factor=2.0 / n_layers,
                )
            return qml.density_matrix(wires=range(n_qubits))

        def _embedding(inputs):
            x = inputs.view(inputs.size(0), n_qubits, -1)
            B, _, n_layers = x.shape
            for i in range(n_layers):
                self.embedding(
                    x[:, :, i], wires=range(n_qubits),
                    rot_factor=2.0 / n_layers, ent_factor=2.0 / n_layers,
                )
            # No return value: this is used for side-effect state preparation.

        self._train_qnode = qml.QNode(_train, self.dev, interface="torch")
        self._dm_qnode = qml.QNode(_dm, self.dev, interface="torch")
        self._embedding = _embedding

    def train_layer(self, x):
        """Evaluate the embedding circuit and return probabilities.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor containing encoded parameters.

        Returns
        -------
        torch.Tensor
            Measurement probabilities for the prepared quantum state.
        """
        return self._train_qnode(x)

    def dm_layer(self, x):
        """Evaluate the embedding circuit and return density matrices.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor containing encoded parameters.

        Returns
        -------
        torch.Tensor
            Density matrices of the prepared quantum states.
        """
        return self._dm_qnode(x)


# Backward compatibility
QLayer = QuantumEmbeddingLayer
