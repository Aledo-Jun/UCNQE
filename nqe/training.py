import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import trange, tqdm
from typing import List
from .data import get_random_data, get_random_data_qcnn


def train(model, X_train, Y_train, optimizer, epoch, batch_size, device):
    """Train ``model`` using randomly generated siamese pairs."""
    model = model.to(device)
    loss_fn = nn.MSELoss()
    train_loss = []
    model.train()
    for i in trange(epoch, desc=f'{model.__class__.__name__}'):
        X1, X2, Y = get_random_data(batch_size, X_train, Y_train)
        X1, X2, Y = X1.to(device), X2.to(device), Y.to(device)
        output = model(X1, X2)
        loss = loss_fn(output, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if i % 200 == 0:
            print(f'Epoch [{i:>4}] Loss: {loss.item():.6f}')
    return train_loss


def build_train_loader(X_train, Y_train, batch_size, *, n=5000):
    """Pre-generate ``n`` batches for training."""
    return [get_random_data(batch_size, X_train, Y_train) for _ in range(n)]


def build_validation_loader(X_val, Y_val, batch_size, *, n=32):
    """Pre-generate ``n`` batches for validation."""
    return [get_random_data(batch_size, X_val, Y_val) for _ in range(n)]


@torch.no_grad()
def compute_val_loss(model, val_loader, loss_fn, device):
    """Compute the mean validation loss over ``val_loader``."""
    model.eval()
    acc_loss = 0.0
    for X1, X2, Y in val_loader:
        X1, X2, Y = X1.to(device), X2.to(device), Y.to(device)
        out = model(X1, X2)
        acc_loss += float(loss_fn(out, Y).item())
    return acc_loss / len(val_loader)


class EarlyStopper:
    """Utility to monitor validation loss and stop training early."""

    def __init__(self, patience=300, min_delta=1e-4, warmup=100):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.best = float("inf")
        self.best_state = None
        self.best_step = -1

    def step(self, val_loss, step, model):
        """Update stopper state and decide whether to stop."""
        improved = (self.best - val_loss) > self.min_delta
        if improved:
            self.best = val_loss
            self.best_step = step
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False, True
        should_stop = (step >= self.warmup) and (step - self.best_step >= self.patience)
        return should_stop, False


def train_with_early_stopping(model, train_loader, val_loader, optimizer,
                              *, max_steps=5000, device=None,
                              validate_every=25, patience=300, warm_up=100,
                              min_delta=1e-4, scheduler=True):
    """Train ``model`` with early stopping based on validation loss."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_fn = nn.MSELoss()
    stopper = EarlyStopper(patience=patience, min_delta=min_delta, warmup=warm_up)
    sched = None
    if scheduler:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=max(validate_every//2, 20),
            threshold=min_delta, min_lr=1e-5
        )
    train_loss_hist, val_loss_hist = [], []
    grad_norm_hist, grad_var_hist = [], []
    for step in trange(max_steps, desc=f'{model.__class__.__name__}'):
        X1, X2, Y = train_loader[step]
        X1, X2, Y = X1.to(device), X2.to(device), Y.to(device)
        model.train()
        out = model(X1, X2)
        loss = loss_fn(out, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_hist.append(float(loss.item()))
        g2 = []
        for p in model.parameters():
            if p.grad is not None:
                g2.append(p.grad.detach().flatten()**2)
        g2 = torch.cat(g2)
        grad_norm_hist.append(g2.norm().item())
        grad_var_hist.append(g2.var().item())
        if (step % validate_every) == 0:
            vloss = compute_val_loss(model, val_loader, loss_fn, device)
            val_loss_hist.append(vloss)
            if sched is not None:
                sched.step(vloss)
            should_stop, improved = stopper.step(vloss, step, model)
            if step % (validate_every*4) == 0:
                print(f"[{step:>4}] train={train_loss_hist[-1]:.4f} val={vloss:.4f}"
                      f" best@{stopper.best_step}={stopper.best:.4f}")
            if should_stop:
                print(f"Early stop at step {step} (best val {stopper.best:.4f} @ {stopper.best_step})")
                break
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    return {
        "train_loss": train_loss_hist,
        "val_loss": val_loss_hist,
        "grad_norm": grad_norm_hist,
        "grad_var": grad_var_hist,
        "best_val": stopper.best,
        "best_step": stopper.best_step,
        "final_step": step,
    }


# QCNN-specific utilities

def build_train_loader_qcnn(X_train, Y_train, batch_size, *, n=5000):
    """Pre-generate ``n`` training batches for a QCNN."""
    return [get_random_data_qcnn(batch_size, X_train, Y_train) for _ in range(n)]


def build_validation_loader_qcnn(X_val, Y_val, batch_size, *, n=32):
    """Pre-generate ``n`` validation batches for a QCNN."""
    return [get_random_data_qcnn(batch_size, X_val, Y_val) for _ in range(n)]


@torch.no_grad()
def compute_val_loss_qcnn(model, val_loader, loss_fn, device):
    """Compute the mean validation loss for a QCNN model."""
    model.eval()
    acc_loss = 0.0
    for X, Y in val_loader:
        X, Y = X.to(device), Y.to(device).to(torch.float32)
        out = model(X)
        acc_loss += float(loss_fn(out, Y).item())
    return acc_loss / len(val_loader)


def train_with_early_stopping_qcnn(model, train_loader, val_loader, test_loader, optimizer,
                                   *, max_steps=5000, device=None,
                                   validate_every=25, patience=300, warm_up=100,
                                   min_delta=1e-4, scheduler=True):
    """Train a QCNN model with early stopping and evaluation metrics."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_fn = nn.MSELoss()
    stopper = EarlyStopper(patience=patience, min_delta=min_delta, warmup=warm_up)
    sched = None
    if scheduler:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=max(validate_every//2, 20),
            threshold=min_delta, min_lr=1e-5
        )
    train_loss_hist, val_loss_hist = [], []
    acc_hist, auc_hist = [], []
    for step, (X, Y) in tqdm(enumerate(train_loader), desc=f'{model.__class__.__name__}-{model.nqe.__class__.__name__}'):
        X, Y = X.to(device), Y.to(device).to(torch.float32)
        model.train()
        out = model(X)
        loss = loss_fn(out, Y)
        optimizer.zero_grad()
        c_layer_before = [m.weight.detach().clone() for m in model.nqe.linear if hasattr(m, 'weight')]
        loss.backward()
        optimizer.step()
        c_layer_after = [m.weight for m in model.nqe.linear if hasattr(m, 'weight')]
        train_loss_hist.append(float(loss.item()))
        for wb, wa in zip(c_layer_before, c_layer_after):
            assert torch.allclose(wb, wa, atol=1e-4)
        if (step % validate_every) == 0:
            correct = 0
            model.eval()
            with torch.no_grad():
                test_score, test_label = [], []
                for Xt, Yt in test_loader:
                    Xt, Yt = Xt.to(device), Yt.to(device).to(torch.float32)
                    out = model(Xt)
                    correct += (torch.round(out) == Yt).sum().item()
                    test_score.append(out.cpu().numpy())
                    test_label.append(Yt.cpu().numpy())
                test_score = np.concatenate(test_score)
                test_label = np.concatenate(test_label)
            acc = correct / len(test_loader.dataset)
            acc_hist.append(acc)
            from sklearn.metrics import roc_auc_score
            auc_hist.append(roc_auc_score(test_label, test_score))
            vloss = compute_val_loss_qcnn(model, val_loader, loss_fn, device)
            val_loss_hist.append(vloss)
            if sched is not None:
                sched.step(vloss)
            should_stop, improved = stopper.step(vloss, step, model)
            if step % (validate_every*4) == 0:
                print(f"[{step:>4}] train={train_loss_hist[-1]:.4f} val={vloss:.4f} acc={acc:.4f}"
                      f" best@{stopper.best_step}={stopper.best:.4f}")
            if should_stop:
                print(f"Early stop at step {step} (best val {stopper.best:.4f} @ {stopper.best_step})")
                break
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    return {
        "train_loss": torch.tensor(train_loss_hist),
        "val_loss": torch.tensor(val_loss_hist),
        "acc": torch.tensor(acc_hist),
        "auc": torch.tensor(auc_hist),
        "best_val": stopper.best,
        "best_step": stopper.best_step,
        "final_step": step,
    }
