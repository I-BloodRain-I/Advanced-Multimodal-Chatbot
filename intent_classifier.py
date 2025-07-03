"""
Core utilities for training and evaluation a three-class intent classifier on
sentence embeddings.

Classes (default): 0 = websearch, 1 = ImgGen, 2 = TextGen
"""

from typing import Tuple, Dict, List, Optional, Any
import argparse
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from copy import deepcopy


# ---------------------------- Dataset ---------------------------- #

class EmbeddingDataset(Dataset):
    """
    Dataset wrapper for sentence embeddings and integer labels.
    """

    def __init__(self, x: Any, y: Any) -> None:
        assert len(x) == len(y), "x and y must have equal length."

        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:                           
        """Dataset size."""
        return self.y.size(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.x[idx], self.y[idx]


# ----------------------------- Model ----------------------------- #

class IntentClassifier(nn.Module):
    """
    The MLP for classifying fixed embeddings.

    Architecture: embed_dim → 512 → 128 → n_classes
    BatchNorm + Dropout keep it stable on small data.
    """

    def __init__(self,
                 embed_dim: int,
                 n_classes: int,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),

            nn.Linear(128, n_classes)
        )

    def forward(self, x: Tensor) -> Tensor:             # noqa: D401
        return self.net(x)

# ------------------------- Data Pipeline ------------------------- #

def _make_loaders(x: torch.Tensor,
                  y: torch.Tensor,
                  val_ratio: float,
                  batch_size: int,
                  seed: int) -> Tuple[DataLoader, DataLoader]:
    """
    Stratified train/val split and DataLoader construction.
    """
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=val_ratio, 
                                                random_state=seed, stratify=y)
    train_loader = DataLoader(EmbeddingDataset(x_tr, y_tr),
                              batch_size=batch_size, 
                              shuffle=True, 
                              drop_last=True
    )
    val_loader = DataLoader(EmbeddingDataset(x_val, y_val),
                            batch_size=batch_size, 
                            shuffle=False, 
                            drop_last=False
    )
    return train_loader, val_loader


def evaluate_intent_classifier(
    model: nn.Module,
    loader: DataLoader,
    n_classes: int,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Compute accuracy and macro-F1 on a DataLoader using torchmetrics.
    Runs entirely on the given device (CPU or GPU).
    """
    acc_metric = MulticlassAccuracy(num_classes=n_classes).to(device)
    f1_metric  = MulticlassF1Score(num_classes=n_classes, average="macro").to(device)

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            acc_metric.update(preds, yb)
            f1_metric.update(preds, yb)

    return acc_metric.compute().item(), f1_metric.compute().item()

# -------------------------- Training Loop -------------------------- #

def train_intent_classifier(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    embed_dim: int = 768,
    n_classes: int = 3,
    seed: int = 42,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    max_epochs: int = 100,
    patience: int = 8,
    dropout: float = 0.2,
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, List[float]], int]:
    """
    Train an `IntentClassifier` with early stopping on validation macro-F1.

    Parameters
    ----------
    x, y           : numpy arrays of shape (n_samples, embed_dim) and (n_samples,)
    embed_dim      : dimensionality of input embeddings
    n_classes      : number of intent labels
    seed           : random seed for reproducibility
    device         : torch device.
    val_ratio      : fraction of data reserved for validation
    batch_size     : mini-batch size
    lr             : AdamW learning rate
    weight_decay   : AdamW weight decay
    max_epochs     : training epochs (upper bound, may stop early)
    patience       : #epochs without val-F1 improvement before early stop
    dropout        : first layer dropout probability
    verbose        : print epoch metrics if True

    Returns
    -------
    model      : trained network (weights from best val-F1 epoch)
    history    : dict with lists of train/val accuracy & macro-F1 per epoch
    best_epoch : epoch achieved the best result by macro-F1
    """

    # ---------- data ---------- #
    train_loader, val_loader = _make_loaders(
        x, y, val_ratio=val_ratio, batch_size=batch_size, seed=seed
    )

    # ---------- model & optimiser ---------- #
    model = IntentClassifier(embed_dim, n_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                     T_0=10, 
                                                                     T_mult=2)

    best_state, best_f1, best_epoch, no_improve = None, 0.0, 0, 0
    hist: Dict[str, List[float]] = {"train_acc": [], "val_acc": [],
                                    "train_f1": [],  "val_f1": []}

    # ------------------------------ training loop ---------------------- #
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss   = criterion(logits, batch_y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step(epoch - 1)

        # ------------- metrics ------------- #
        train_acc, train_f1 = evaluate_intent_classifier(model, train_loader, 
                                                   n_classes, device)
        val_acc, val_f1 = evaluate_intent_classifier(model, val_loader, 
                                                   n_classes, device)

        hist["train_acc"].append(train_acc) 
        hist["train_f1"].append(train_f1)
        hist["val_acc"].append(val_acc)   
        hist["val_f1"].append(val_f1)

        if verbose:
            print(f"Epoch {epoch:3d} | "
                  f"train_acc={train_acc:.3f}  train_f1={train_f1:.3f} | "
                  f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}")

        # ------------- early stopping ------------- #
        if val_f1 > best_f1:
            best_f1, no_improve = val_f1, 0
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print("Early stopping.")
                break

    # ---------- load best weights ---------- #
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, hist, best_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train intent-classifier on the data."
    )
    parser.add_argument("--embed-dim", type=int, default=768,
                        help="Size of the input embeddings.")
    parser.add_argument("--n-classes", type=int, default=3,
                        help="Number of intent categories.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed.")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu | cuda | cuda:0 … (auto if omitted)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mini-batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
                        help="Weight decay for optimizer.")
    parser.add_argument("--max-epochs", type=int, default=100,
                        help="Maximum number of training epochs.")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience (epochs without val-F1 improvement).")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate for MLP layers.")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Validation set ratio.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable training progress output.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ) 

    seed = args.seed if args.seed is not None else np.random.randint(0, 10000) 
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Dummy data to verify training
    X = torch.randn((10_000, 768), dtype=torch.float)
    y = torch.randint(0, 3, (10_000,))

    model, history, best_epoch = train_intent_classifier(
        X, y,
        embed_dim=args.embed_dim,
        n_classes=args.n_classes,
        device=device,
        seed=seed,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        dropout=args.dropout,
        verbose=args.verbose
    )

    best_index = best_epoch - 1
    print(f"Finished | "
          f"train_acc={history['train_acc'][best_index]:.3f}  train_f1={history['train_f1'][best_index]:.3f} | "
          f"val_acc={history['val_acc'][best_index]:.3f}  val_f1={history['val_f1'][best_index]:.3f}")
    
    # ------------------ save the best model on the disk ------------------ 
    torch.save({
        "weights": model.state_dict(), 
        "train_acc": history['train_acc'][best_index],
        "train_f1":  history['train_f1'][best_index],
        "val_acc":   history['val_acc'][best_index],
        "val_f1":    history['val_f1'][best_index],
    }, './models/intent_classifier.pth')    
