"""
Training script for Latent Program Network
Run this to train the model on the list operations dataset
"""

import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.lpn_model import (
    LatentProgramNetwork,
    ListOpsDataset,
    train_model
)


def main():
    parser = argparse.ArgumentParser(description='Train Latent Program Network')
    parser.add_argument('--data_dir', type=str, default='./data/list_ops_data',
                      help='Path to data directory')
    parser.add_argument('--latent_dim', type=int, default=64,
                      help='Dimension of latent space')
    parser.add_argument('--hidden_dim', type=int, default=128,
                      help='Hidden dimension for networks')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1,
                      help='KL divergence weight')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to train on')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training Latent Program Network")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Beta (KL weight): {args.beta}")
    print("=" * 60)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = ListOpsDataset(f"{args.data_dir}/train.json")
    val_dataset = ListOpsDataset(f"{args.data_dir}/val.json")
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\nInitializing model...")
    model = LatentProgramNetwork(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        max_length=20
    ).to(args.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Train
    print("\nStarting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        num_epochs=args.num_epochs,
        lr=args.lr,
        beta=args.beta
    )
    
    # Save training history
    history_path = Path('./results/metrics/training_history.json')
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Saved training history to {history_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best model saved to: results/models/best_lpn_model.pt")
    print(f"Training history saved to: {history_path}")
    print("\nNext step: Run 'python src/testing/test_lpn.py' to evaluate with test-time search")


if __name__ == "__main__":
    main()
