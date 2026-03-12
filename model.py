"""
model.py
SimCLR model: ResNet-18 encoder + MLP projection head.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head.
    Maps encoder output h -> latent z where NT-Xent loss is applied.
    The projection head is discarded after pre-training; only the
    encoder (h) is used for downstream tasks.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256,
                 output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    """
    Full SimCLR model.
        f(.)  : ResNet-18 encoder  -> h  (512-d representation)
        g(.)  : Projection head    -> z  (128-d, used only during pre-training)
    """

    def __init__(self, projection_dim: int = 128):
        super().__init__()

        # Encoder: ResNet-18 with the final FC layer replaced by identity
        backbone = resnet18(weights=None)
        self.encoder_dim = backbone.fc.in_features          # 512
        backbone.fc = nn.Identity()
        self.encoder = backbone

        # Projection head
        self.projector = ProjectionHead(
            input_dim=self.encoder_dim,
            hidden_dim=256,
            output_dim=projection_dim
        )

    def encode(self, x):
        """Return the representation h (used for downstream evaluation)."""
        return self.encoder(x)

    def forward(self, x):
        """Return the projected representation z (used during pre-training)."""
        h = self.encoder(x)
        z = self.projector(h)
        return z


class LinearClassifier(nn.Module):
    """
    Single linear layer placed on top of a frozen pre-trained encoder.
    Used for the linear-probing evaluation protocol.
    """

    def __init__(self, input_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
