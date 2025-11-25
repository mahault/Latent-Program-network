"""
Training script for Spatial LPN on grid tasks
Tests on synthetic grid transformations before ARC
"""

import torch
from torch.utils.data import DataLoader
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from spatial_model import SpatialLPN, compute_spatial_loss
from generate_grid_data import SyntheticGridDataset


def train_epoch_spatial(model: SpatialLPN, dataloader: DataLoader,
                       optimizer: torch.optim.Optimizer, device: str, beta: float = 0.01) -> dict:
    """Train one epoch"""
    model.train()
    
    total_loss = 0
    total_ce = 0
    total_kl = 0
    total_acc = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        train_inputs = batch['train_inputs'].to(device)
        train_outputs = batch['train_outputs'].to(device)
        test_input = batch['test_input'].to(device)
        test_output = batch['test_output'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output_logits, mu, logvar = model(train_inputs, train_outputs, test_input)
        
        # Compute loss
        losses = compute_spatial_loss(output_logits, test_output, mu, logvar, beta)
        
        # Backward
        losses['total'].backward()
        optimizer.step()
        
        # Compute accuracy
        predictions = output_logits.argmax(dim=1)
        acc = (predictions == test_output).float().mean().item()
        
        total_loss += losses['total'].item()
        total_ce += losses['ce'].item()
        total_kl += losses['kl'].item()
        total_acc += acc
    
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'ce': total_ce / num_batches,
        'kl': total_kl / num_batches,
        'accuracy': total_acc / num_batches
    }


def evaluate_spatial(model: SpatialLPN, dataloader: DataLoader, device: str,
                     use_search: bool = False) -> dict:
    """Evaluate spatial model"""
    model.eval()
    
    total_acc = 0
    total_pixel_acc = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval ({'with' if use_search else 'no'} search)"):
            train_inputs = batch['train_inputs'].to(device)
            train_outputs = batch['train_outputs'].to(device)
            test_input = batch['test_input'].to(device)
            test_output = batch['test_output'].to(device)
            
            if use_search:
                output_logits = model.predict_with_search(
                    train_inputs, train_outputs, test_input,
                    num_steps=30, lr=0.05
                )
            else:
                output_logits, _, _ = model(train_inputs, train_outputs, test_input)
            
            # Predictions
            predictions = output_logits.argmax(dim=1)
            
            # Exact match (whole grid correct)
            exact = (predictions == test_output).all(dim=[1, 2]).float().sum().item()
            
            # Pixel accuracy
            pixel_acc = (predictions == test_output).float().mean().item()
            
            total_acc += exact
            total_pixel_acc += pixel_acc * test_input.shape[0]
            total_samples += test_input.shape[0]
    
    return {
        'accuracy': total_acc / total_samples,
        'pixel_accuracy': total_pixel_acc / total_samples
    }


def main():
    parser = argparse.ArgumentParser(description='Train Spatial LPN')
    parser.add_argument('--data_dir', type=str, default='./synthetic_grid_data')
    parser.add_argument('--num_colors', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training Spatial LPN on Grid Tasks")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print("=" * 60)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = SyntheticGridDataset(f"{args.data_dir}/train.json")
    val_dataset = SyntheticGridDataset(f"{args.data_dir}/val.json")
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print("\nInitializing Spatial LPN...")
    model = SpatialLPN(
        num_colors=args.num_colors,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    ).to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    # Training loop
    print("\nStarting training...")
    history = {
        'train_loss': [],
        'train_ce': [],
        'train_kl': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'val_pixel_accuracy': []
    }
    
    best_val_acc = 0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_metrics = train_epoch_spatial(model, train_loader, optimizer, args.device, args.beta)
        history['train_loss'].append(train_metrics['loss'])
        history['train_ce'].append(train_metrics['ce'])
        history['train_kl'].append(train_metrics['kl'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        
        # Validate
        val_metrics = evaluate_spatial(model, val_loader, args.device, use_search=False)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_pixel_accuracy'].append(val_metrics['pixel_accuracy'])
        
        scheduler.step(val_metrics['accuracy'])
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val Pixel Acc: {val_metrics['pixel_accuracy']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), 'best_spatial_model.pt')
            print(f"✓ Saved best model (acc: {best_val_acc:.4f})")
    
    # Save history
    with open('spatial_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("\nNext: Run 'python test_spatial.py' to evaluate with test-time search")


if __name__ == "__main__":
    main()
