"""
loss.py
NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss.
Implementation follows Chen et al., ICML 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent loss for a batch of N images.

    Each image produces two augmented views, giving 2N representations.
    For each sample i, the positive pair is its other augmented view (j),
    and the 2(N-1) other samples in the batch are negatives.

    Args:
        temperature: scaling factor tau (default 0.5, as in SimCLR paper)
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i : projected embeddings for view 1  (N, dim)
            z_j : projected embeddings for view 2  (N, dim)
        Returns:
            scalar loss
        """
        N = z_i.size(0)
        device = z_i.device

        # L2-normalise both sets of embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate: [z_i ; z_j]  shape (2N, dim)
        z = torch.cat([z_i, z_j], dim=0)

        # Compute full (2N x 2N) cosine similarity matrix, scaled by temperature
        sim = torch.mm(z, z.T) / self.temperature  # (2N, 2N)

        # Mask out self-similarities on the diagonal
        mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, float('-inf'))

        # Positive pair indices:
        # for row i in [0, N),   positive is at index i + N
        # for row i in [N, 2N),  positive is at index i - N
        pos_indices = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N, device=device)
        ])  # (2N,)

        # Cross-entropy over the 2N-1 non-self similarities
        loss = self.criterion(sim, pos_indices)
        return loss
