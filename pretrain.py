"""
pretrain.py
Self-supervised SimCLR pre-training on EuroSAT (unlabeled).
Saves the encoder weights after training for downstream evaluation.

Usage:
    python pretrain.py --epochs 50 --batch_size 256 --lr 1e-3
"""

import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_eurosat_pretrain
from model import SimCLR
from loss import NTXentLoss


def parse_args():
    parser = argparse.ArgumentParser(description="SimCLR Pre-training on EuroSAT")
    parser.add_argument("--data_root",   type=str,   default="./data")
    parser.add_argument("--save_dir",    type=str,   default="./checkpoints")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--proj_dim",    type=int,   default=128)
    parser.add_argument("--num_workers", type=int,   default=4)
    return parser.parse_args()


def pretrain(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    loader = get_eurosat_pretrain(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Model, loss, optimiser
    model = SimCLR(projection_dim=args.proj_dim).to(device)
    criterion = NTXentLoss(temperature=args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nStarting pre-training for {args.epochs} epochs "
          f"| batch={args.batch_size} | lr={args.lr} | tau={args.temperature}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for (x_i, x_j), _ in loader:
            x_i, x_j = x_i.to(device), x_j.to(device)

            z_i = model(x_i)
            z_j = model(x_j)

            loss = criterion(z_i, z_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch:>3}/{args.epochs}]  Loss: {avg_loss:.4f}  "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save encoder weights only (projection head is discarded)
    ckpt_path = os.path.join(args.save_dir, "simclr_encoder.pth")
    torch.save(model.encoder.state_dict(), ckpt_path)
    print(f"\nEncoder saved to {ckpt_path}")


if __name__ == "__main__":
    args = parse_args()
    pretrain(args)
