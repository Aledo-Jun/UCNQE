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

## Training script

A convenient CLI is available for training NQE, NQE_BIG or UCNQE models and
visualising the loss curves.  For example, the following trains a UCNQE model
with block up-convolution:

```bash
python scripts/train_nqe.py --model ucnqe --uc-mode block --dataset mnist \
    --pca-dim 8 --n-qubits 4 --n-layers 2 --hidden-dims 16 16 --max-steps 1000
```

The `--model` flag selects the architecture (`nqe`, `nqe_big` or `ucnqe`); the
`--uc-mode` option chooses the up-convolution strategy used by UCNQE models.
All major hyper-parameters (dataset choice, model size, optimisation settings,
number of steps, etc.) can be configured through command line flags.  The script
saves a plot of the training and validation losses.
