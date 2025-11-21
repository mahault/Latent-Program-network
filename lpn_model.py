"""
Latent Program Network (LPN) Implementation
Based on "Searching Latent Program Spaces" (Bonnet et al., 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm


class ListOpsDataset(Dataset):
    """Dataset for list operation tasks"""
    
    def __init__(self, data_path: str, max_length: int = 20):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def pad_sequence(self, seq: List[int]) -> torch.Tensor:
        """Pad sequence to max_length"""
        padded = seq + [0] * (self.max_length - len(seq))
        return torch.tensor(padded[:self.max_length], dtype=torch.float32)
    
    def get_length_mask(self, seq: List[int]) -> torch.Tensor:
        """Create mask for actual sequence length"""
        length = min(len(seq), self.max_length)
        mask = torch.zeros(self.max_length)
        mask[:length] = 1
        return mask
    
    def __getitem__(self, idx):
        task = self.data[idx]
        
        # Prepare training examples (for encoder)
        train_inputs = []
        train_outputs = []
        train_masks = []
        
        for ex in task['train_examples']:
            train_inputs.append(self.pad_sequence(ex['input']))
            train_outputs.append(self.pad_sequence(ex['output']))
            train_masks.append(self.get_length_mask(ex['output']))
        
        # Prepare test examples
        test_inputs = []
        test_outputs = []
        test_masks = []
        
        for ex in task['test_examples']:
            test_inputs.append(self.pad_sequence(ex['input']))
            test_outputs.append(self.pad_sequence(ex['output']))
            test_masks.append(self.get_length_mask(ex['output']))
        
        return {
            'train_inputs': torch.stack(train_inputs),     # [num_examples, max_length]
            'train_outputs': torch.stack(train_outputs),   # [num_examples, max_length]
            'train_masks': torch.stack(train_masks),       # [num_examples, max_length]
            'test_inputs': torch.stack(test_inputs),
            'test_outputs': torch.stack(test_outputs),
            'test_masks': torch.stack(test_masks),
            'program_type': task['program_type']
        }


class Encoder(nn.Module):
    """
    Encoder: Takes example pairs and outputs latent program distribution
    """
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Process each input-output pair
        self.pair_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Aggregate across all pairs
        self.aggregator = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Output mean and log_variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: [batch, num_examples, seq_len]
            outputs: [batch, num_examples, seq_len]
        Returns:
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
        """
        batch_size, num_examples, seq_len = inputs.shape
        
        # Combine input and output for each pair
        pairs = torch.cat([inputs, outputs], dim=-1)  # [batch, num_examples, seq_len*2]
        
        # Reshape for processing
        pairs = pairs.view(batch_size * num_examples, -1)
        pair_encodings = self.pair_encoder(pairs)  # [batch*num_examples, hidden_dim]
        pair_encodings = pair_encodings.view(batch_size, num_examples, -1)
        
        # Aggregate across examples
        _, (hidden, _) = self.aggregator(pair_encodings)
        aggregated = hidden[-1]  # [batch, hidden_dim]
        
        # Output distribution parameters
        mu = self.fc_mu(aggregated)
        logvar = self.fc_logvar(aggregated)
        
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder: Takes latent program + input and generates output
    """
    
    def __init__(self, input_dim: int = 1, latent_dim: int = 64, hidden_dim: int = 128, max_length: int = 20):
        super().__init__()
        self.max_length = max_length
        
        # Combine latent program with input
        self.input_processor = nn.Linear(input_dim + latent_dim, hidden_dim)
        
        # Autoregressive decoder
        self.decoder_rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, z: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch, latent_dim]
            inputs: [batch, seq_len]
        Returns:
            outputs: [batch, seq_len]
        """
        batch_size, seq_len = inputs.shape
        
        # Expand latent to match sequence
        z_expanded = z.unsqueeze(1).expand(batch_size, seq_len, -1)  # [batch, seq_len, latent_dim]
        inputs_expanded = inputs.unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Combine
        combined = torch.cat([inputs_expanded, z_expanded], dim=-1)  # [batch, seq_len, 1+latent_dim]
        processed = self.input_processor(combined)
        
        # Decode
        decoded, _ = self.decoder_rnn(processed)
        outputs = self.output_layer(decoded).squeeze(-1)  # [batch, seq_len]
        
        return outputs


class LatentProgramNetwork(nn.Module):
    """
    Complete LPN: Encoder + Decoder with test-time optimization
    """
    
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128, max_length: int = 20):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim=1, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = Decoder(input_dim=1, latent_dim=latent_dim, hidden_dim=hidden_dim, max_length=max_length)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, train_inputs: torch.Tensor, train_outputs: torch.Tensor, 
                test_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass
        """
        # Encode examples to latent
        mu, logvar = self.encoder(train_inputs, train_outputs)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode test inputs
        predictions = self.decoder(z, test_inputs)
        
        return predictions, mu, logvar
    
    def predict_with_search(self, train_inputs: torch.Tensor, train_outputs: torch.Tensor,
                           train_masks: torch.Tensor, test_inputs: torch.Tensor,
                           num_steps: int = 50, lr: float = 0.1) -> torch.Tensor:
        """
        Test-time optimization: refine latent to best explain training examples
        """
        batch_size = train_inputs.shape[0]
        
        # Initialize with encoder
        with torch.no_grad():
            mu, logvar = self.encoder(train_inputs, train_outputs)
            z = mu.clone()  # Start from mean
        
        # Make z require gradients
        z = z.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)
        
        # Optimize z to fit training examples
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Decode all training examples with current z
            num_train = train_inputs.shape[1]
            z_expanded = z.unsqueeze(1).expand(batch_size, num_train, -1)
            
            loss = 0
            for i in range(num_train):
                pred = self.decoder(z_expanded[:, i, :], train_inputs[:, i, :])
                mask = train_masks[:, i, :]
                loss += F.mse_loss(pred * mask, train_outputs[:, i, :] * mask)
            
            loss = loss / num_train
            loss.backward()
            optimizer.step()
        
        # Use optimized z for test prediction
        with torch.no_grad():
            predictions = self.decoder(z, test_inputs)
        
        return predictions


def compute_loss(predictions: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor,
                mu: torch.Tensor, logvar: torch.Tensor, beta: float = 0.1) -> Dict[str, torch.Tensor]:
    """
    Compute ELBO loss
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(predictions * masks, targets * masks, reduction='mean')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return {
        'total': total_loss,
        'recon': recon_loss,
        'kl': kl_loss
    }


def train_epoch(model: LatentProgramNetwork, dataloader: DataLoader, 
               optimizer: torch.optim.Optimizer, device: str, beta: float = 0.1) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        train_inputs = batch['train_inputs'].to(device)
        train_outputs = batch['train_outputs'].to(device)
        train_masks = batch['train_masks'].to(device)
        test_inputs = batch['test_inputs'][:, 0, :].to(device)  # Use first test example
        test_outputs = batch['test_outputs'][:, 0, :].to(device)
        test_masks = batch['test_masks'][:, 0, :].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions, mu, logvar = model(train_inputs, train_outputs, test_inputs)
        
        # Compute loss
        losses = compute_loss(predictions, test_outputs, test_masks, mu, logvar, beta)
        
        # Backward pass
        losses['total'].backward()
        optimizer.step()
        
        total_loss += losses['total'].item()
        total_recon += losses['recon'].item()
        total_kl += losses['kl'].item()
    
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'kl': total_kl / num_batches
    }


def evaluate(model: LatentProgramNetwork, dataloader: DataLoader, 
            device: str, use_search: bool = False) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()
    total_mse = 0
    total_exact = 0
    total_samples = 0
    
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
            else:
                mu, logvar = model.encoder(train_inputs, train_outputs)
                predictions = model.decoder(mu, test_inputs)
            
            # Compute metrics
            masked_pred = predictions * test_masks
            masked_target = test_outputs * test_masks
            
            mse = F.mse_loss(masked_pred, masked_target, reduction='sum').item()
            
            # Exact match (allowing small tolerance)
            exact = (torch.abs(masked_pred - masked_target) < 0.5).all(dim=1).sum().item()
            
            total_mse += mse
            total_exact += exact
            total_samples += test_inputs.shape[0]
    
    return {
        'mse': total_mse / total_samples,
        'accuracy': total_exact / total_samples
    }


def train_model(model: LatentProgramNetwork, train_loader: DataLoader, val_loader: DataLoader,
               device: str, num_epochs: int = 50, lr: float = 1e-3, beta: float = 0.1) -> Dict:
    """Full training loop"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    history = {
        'train_loss': [],
        'train_recon': [],
        'train_kl': [],
        'val_mse': [],
        'val_accuracy': []
    }
    
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, beta)
        history['train_loss'].append(train_metrics['loss'])
        history['train_recon'].append(train_metrics['recon'])
        history['train_kl'].append(train_metrics['kl'])
        
        # Validate (without search for speed)
        val_metrics = evaluate(model, val_loader, device, use_search=False)
        history['val_mse'].append(val_metrics['mse'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        
        scheduler.step(val_metrics['mse'])
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val MSE: {val_metrics['mse']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), 'best_lpn_model.pt')
            print(f"✓ Saved best model (acc: {best_val_acc:.4f})")
    
    return history


if __name__ == "__main__":
    print("LPN Model Module")
    print("Import this module to use the model")
