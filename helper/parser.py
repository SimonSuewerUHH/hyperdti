import argparse
from dataclasses import dataclass

import torch


@dataclass
class Config:
    lr: float = 0.005
    weight_decay: float = 1e-5
    epochs: int = 100
    step_size: int = 50
    gamma: float = 0.5
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    hidden_dim: int = 64
    heads: int = 4
    dropout: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description='HeteroHyperModel Training')
    parser.add_argument('--lr', type=float, default=Config.lr)
    parser.add_argument('--weight-decay', type=float, default=Config.weight_decay)
    parser.add_argument('--epochs', type=int, default=Config.epochs)
    parser.add_argument('--step-size', type=int, default=Config.step_size)
    parser.add_argument('--gamma', type=float, default=Config.gamma)
    parser.add_argument('--hidden-dim', type=int, default=Config.hidden_dim)
    parser.add_argument('--heads', type=int, default=Config.heads)
    parser.add_argument('--dropout', type=float, default=Config.dropout)
    parser.add_argument('--seed', type=int, default=Config.seed)
    args = parser.parse_args()
    return Config(
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        step_size=args.step_size,
        gamma=args.gamma,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
    )
