# UCNQE

This repository reconstructs the original project from the `NQE.ipynb` notebook.
It provides utilities for building and training neural quantum embedding (NQE)
models and quantum convolutional neural networks (QCNNs) with multi‑layer
embeddings.

## Package layout

```
nqe/
  data.py         # dataset loading and pair generation utilities
  embedding.py    # PennyLane embedding layer and QNode wrapper
  models.py       # PyTorch modules for NQE, UCNQE and QCNN
  training.py     # training loops with early stopping
  utils.py        # plotting helpers and analysis utilities
```

The modules can be imported as a package:

```python
from nqe import NQE, QuantumEmbeddingLayer, ZZFeatureMap, train_with_early_stopping
```

The code requires PyTorch, PennyLane and the usual scientific Python stack.
