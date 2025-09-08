import numpy
import torch
from torchvision.datasets import MNIST, FashionMNIST
from sklearn.decomposition import PCA


def load_mnist_pca(reduction_dim: int):
    """Load the binary MNIST dataset and apply PCA.

    The function keeps only the digits ``0`` and ``1`` from the MNIST
    dataset, flattens the images, scales them to ``[0, 1]`` and then
    performs a principal component analysis to reduce the feature
    dimensionality.

    Parameters
    ----------
    reduction_dim:
        Number of PCA components to keep.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Training features, training labels, test features and test labels
        after PCA transformation.
    """
    mnist_trainset = MNIST(root='./data', train=True, download=True)
    mnist_testset = MNIST(root='./data', train=False, download=True)
    x_train, y_train = mnist_trainset.data.numpy(), mnist_trainset.targets.numpy()
    x_test, y_test = mnist_testset.data.numpy(), mnist_testset.targets.numpy()
    x_train = x_train[numpy.where(y_train < 2)].reshape(-1, 28*28) / 255.0
    y_train = y_train[numpy.where(y_train < 2)]
    x_test = x_test[numpy.where(y_test < 2)].reshape(-1, 28*28) / 255.0
    y_test = y_test[numpy.where(y_test < 2)]
    pca = PCA(reduction_dim)
    X_train = pca.fit_transform(x_train)
    X_test = pca.transform(x_test)
    x_train = X_train / 2
    x_test = X_test / 2
    return x_train, y_train, x_test, y_test


def load_fashion_mnist_pca(reduction_dim: int):
    """Load the binary Fashion-MNIST dataset and apply PCA.

    This utility mirrors :func:`load_mnist_pca` but uses the
    ``FashionMNIST`` dataset instead.  Only classes ``0`` and ``1`` are
    kept and the resulting samples are projected onto a reduced number of
    principal components.

    Parameters
    ----------
    reduction_dim:
        Number of PCA components to keep.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Training features, training labels, test features and test labels
        after PCA transformation.
    """
    f_train = FashionMNIST(root='./data', train=True, download=True)
    f_test = FashionMNIST(root='./data', train=False, download=True)
    x_train, y_train = f_train.data.numpy(), f_train.targets.numpy()
    x_test, y_test = f_test.data.numpy(), f_test.targets.numpy()
    x_train = x_train[numpy.where(y_train < 2)].reshape(-1, 28*28) / 255.0
    y_train = y_train[numpy.where(y_train < 2)]
    x_test = x_test[numpy.where(y_test < 2)].reshape(-1, 28*28) / 255.0
    y_test = y_test[numpy.where(y_test < 2)]
    pca = PCA(reduction_dim)
    X_train = pca.fit_transform(x_train)
    X_test = pca.transform(x_test)
    x_train = X_train / 2
    x_test = X_test / 2
    return x_train, y_train, x_test, y_test


def get_random_data(batch_size, X, Y):
    """Generate random paired samples for siamese training.

    Two independent samples are drawn from ``X``.  The returned label is
    ``1`` when the corresponding targets in ``Y`` match and ``0``
    otherwise.  The function is used to build training batches for the
    ``NQE`` model family.

    Parameters
    ----------
    batch_size:
        Number of pairs to generate.
    X, Y:
        Source feature and label arrays.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Two batches of features and the binary similarity label.
    """
    X1, X2, Y_new = [], [], []
    for _ in range(batch_size):
        n, m = numpy.random.randint(len(X)), numpy.random.randint(len(X))
        X1.append(X[n])
        X2.append(X[m])
        Y_new.append(1 if Y[n] == Y[m] else 0)
    X1 = torch.as_tensor(numpy.array(X1), dtype=torch.float32)
    X2 = torch.as_tensor(numpy.array(X2), dtype=torch.float32)
    Y_new = torch.as_tensor(numpy.array(Y_new), dtype=torch.float32)
    return X1, X2, Y_new


def get_random_data_qcnn(batch_size, X, Y):
    """Randomly sample a batch for QCNN training.

    Samples are drawn with replacement from ``X`` and ``Y`` to create a
    batch of feature vectors and labels for quantum convolutional neural
    networks.

    Parameters
    ----------
    batch_size:
        Number of samples to draw.
    X, Y:
        Source feature and label arrays.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A batch of features and their corresponding labels as tensors.
    """
    X_new, Y_new = [], []
    for _ in range(batch_size):
        n = numpy.random.randint(len(X))
        X_new.append(X[n])
        Y_new.append(Y[n])
    X_new = torch.as_tensor(numpy.array(X_new), dtype=torch.float32)
    Y_new = torch.as_tensor(numpy.array(Y_new), dtype=torch.float32)
    return X_new, Y_new
