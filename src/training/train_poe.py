"""
Training script for Product of Experts LPN
Compares PoE vs baseline amortized inference
"""

import torch
from torch.utils.data import DataLoader
import argparse
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from tqdm import tqdm

from src.models.poe_model import ProductOfExpertsLPN, compute_poe_loss
from src.models.lpn_model import ListOpsDataset


def train_epoch_poe(model: ProductOfExpertsLPN, dataloader: DataLoader, 
                   optimizer: torch.optim.Optimizer, device: str, beta: float = 0.1) -> dict:
    """Train one epoch with PoE-LPN"""
    model.train()
    
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_agreement = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        train_inputs = batch['train_inputs'].to(device)
        train_outputs = batch['train_outputs'].to(device)
        train_masks = batch['train_masks'].to(device)
        test_inputs = batch['test_inputs'][:, 0, :].to(device)
        test_outputs = batch['test_outputs'][:, 0, :].to(device)
        test_masks = batch['test_masks'][:, 0, :].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions, mu, logvar, agreement = model(train_inputs, train_outputs, test_inputs)
        
        # Compute loss
        losses = compute_poe_loss(
            predictions, test_outputs, test_masks,
            mu, logvar,
            model.poe_combiner.prior_mu, model.poe_combiner.prior_logvar,
            beta
        )
        
        # Backward
        losses['total'].backward()
        optimizer.step()
        
        total_loss += losses['total'].item()
        total_recon += losses['recon'].item()
        total_kl += losses['kl'].item()
        total_agreement += agreement.mean().item()
    
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'kl': total_kl / num_batches,
        'agreement': total_agreement / num_batches
    }


def evaluate_poe(model: ProductOfExpertsLPN, dataloader: DataLoader, 
                device: str, use_search: bool = False) -> dict:
    """Evaluate PoE-LPN"""
    model.eval()
    
    total_mse = 0
    total_exact = 0
    total_samples = 0
    total_agreement = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval ({'with' if use_search else 'no'} search)"):
            train_inputs = batch['train_inputs'].to(device)
            train_outputs = batch['train_outputs'].to(device)
            train_masks = batch['train_masks'].to(device)
            test_inputs = batch['test_inputs'][:, 0, :].to(device)
            test_outputs = batch['test_outputs'][:, 0, :].to(device)
            test_masks = batch['test_masks'][:, 0, :].to(device)
            
            if use_search:
                predictions = model.predict_with_search(
                    train_inputs, train_outputs, train_masks, test_inputs,
                    num_steps=50, lr=0.1
                )
                agreement = torch.ones(test_inputs.shape[0])  # Dummy for search
            else:
                predictions, _, _, agreement = model(train_inputs, train_outputs, test_inputs)
            
            # Compute metrics
            masked_pred = predictions * test_masks
            masked_target = test_outputs * test_masks
            
            mse = torch.nn.functional.mse_loss(masked_pred, masked_target, reduction='sum').item()
            exact = (torch.abs(masked_pred - masked_target) < 0.5).all(dim=1).sum().item()
            
            total_mse += mse
            total_exact += exact
            total_samples += test_inputs.shape[0]
            total_agreement += agreement.mean().item()
    
    return {
        'mse': total_mse / total_samples,
        'accuracy': total_exact / total_samples,
        'agreement': total_agreement / len(dataloader)
    }


def main():
    parser = argparse.ArgumentParser(description='Train Product of Experts LPN')
    parser.add_argument('--data_dir', type=str, default='./data/list_ops_data')
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training Product of Experts LPN")
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
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print("\nInitializing PoE-LPN...")
    model = ProductOfExpertsLPN(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        max_length=20
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
        'train_recon': [],
        'train_kl': [],
        'train_agreement': [],
        'val_mse': [],
        'val_accuracy': [],
        'val_agreement': []
    }
    
    best_val_acc = 0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_metrics = train_epoch_poe(model, train_loader, optimizer, args.device, args.beta)
        history['train_loss'].append(train_metrics['loss'])
        history['train_recon'].append(train_metrics['recon'])
        history['train_kl'].append(train_metrics['kl'])
        history['train_agreement'].append(train_metrics['agreement'])
        
        # Validate (without search for speed)
        val_metrics = evaluate_poe(model, val_loader, args.device, use_search=False)
        history['val_mse'].append(val_metrics['mse'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_agreement'].append(val_metrics['agreement'])
        
        scheduler.step(val_metrics['accuracy'])
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Agreement: {train_metrics['agreement']:.3f} | "
              f"Val MSE: {val_metrics['mse']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val Agreement: {val_metrics['agreement']:.3f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), 'results/models/best_poe_model.pt')
            print(f"✓ Saved best model (acc: {best_val_acc:.4f})")
    
    # Save history
    with open('results/metrics/poe_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Saved training history to poe_training_history.json")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: best_poe_model.pt")
    print("\nNext: Run 'python test_poe.py' to evaluate with test-time search")


if __name__ == "__main__":
    main()
