import torch
import torch.nn as nn

class TaskClassifier(nn.Module):
    """
    A simple multi-layer perceptron (MLP) for classifying fixed-size embeddings into task categories.

    The model architecture follows:
        embed_dim -> 256 -> 128 -> n_classes,
    with BatchNorm and Dropout layers to help regularize training, especially on smaller datasets.

    Args:
        embed_dim (int): Dimension of the input embedding vectors.
        n_classes (int): Number of output classes to predict.
        dropout (float): Dropout rate applied after each ReLU activation (default is 0.2).
    """

    def __init__(self,
                 embed_dim: int,
                 n_classes: int,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),

            nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)