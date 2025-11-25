"""
Product of Experts Latent Program Network
Implements Bayesian inference by combining evidence from multiple examples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class SingleExampleEncoder(nn.Module):
    """
    Encodes ONE (input, output) example pair to a distribution over θ
    This is the building block for Product of Experts
    """
    
    def __init__(self, input_dim: int = 1, latent_dim: int = 64, hidden_dim: int = 128, max_length: int = 20):
        super().__init__()
        
        # Process input sequence
        self.input_lstm = nn.LSTM(input_dim, hidden_dim // 2, batch_first=True)
        
        # Process output sequence  
        self.output_lstm = nn.LSTM(input_dim, hidden_dim // 2, batch_first=True)
        
        # Combine and output distribution parameters
        self.fc_combine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, input_seq: torch.Tensor, output_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_seq: [batch, seq_len]
            output_seq: [batch, seq_len]
        Returns:
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
        """
        # Add channel dimension for LSTM
        input_seq = input_seq.unsqueeze(-1)  # [batch, seq_len, 1]
        output_seq = output_seq.unsqueeze(-1)
        
        # Encode sequences
        _, (h_in, _) = self.input_lstm(input_seq)
        _, (h_out, _) = self.output_lstm(output_seq)
        
        # Combine: [batch, hidden_dim]
        combined = torch.cat([h_in[-1], h_out[-1]], dim=-1)
        combined = self.fc_combine(combined)
        
        # Output distribution
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        
        return mu, logvar


class ProductOfExpertsCombiner(nn.Module):
    """
    Combines multiple Gaussian distributions q(θ|example_i) via product
    
    For Gaussians: q(θ) ∝ ∏ᵢ 𝒩(μᵢ, σᵢ²)
    Result: 𝒩(μ*, σ*²) where:
        1/σ*² = Σᵢ 1/σᵢ²
        μ* = σ*² * Σᵢ (μᵢ/σᵢ²)
    """
    
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        
        # Learnable prior: p(θ)
        self.register_parameter('prior_mu', nn.Parameter(torch.zeros(latent_dim)))
        self.register_parameter('prior_logvar', nn.Parameter(torch.zeros(latent_dim)))
    
    def combine(self, mus: torch.Tensor, logvars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine via Product of Experts
        
        Args:
            mus: [batch, num_examples, latent_dim]
            logvars: [batch, num_examples, latent_dim]
        Returns:
            mu_combined: [batch, latent_dim]
            logvar_combined: [batch, latent_dim]
        """
        # Convert log variance to precision (1/variance)
        precisions = torch.exp(-logvars)  # [batch, num_ex, latent]
        
        # Prior precision
        prior_precision = torch.exp(-self.prior_logvar)
        
        # Total precision = prior + sum of example precisions
        total_precision = prior_precision + precisions.sum(dim=1)
        
        # Weighted sum of means (weighted by precision)
        weighted_mus = (mus * precisions).sum(dim=1)
        prior_weighted = self.prior_mu * prior_precision
        
        # Combined distribution
        mu_combined = (prior_weighted + weighted_mus) / total_precision
        logvar_combined = -torch.log(total_precision)
        
        return mu_combined, logvar_combined
    
    def compute_agreement(self, mus: torch.Tensor, logvars: torch.Tensor) -> torch.Tensor:
        """
        Measure consistency between different examples
        Higher = examples agree more
        
        Returns:
            agreement: [batch] - scalar per batch element
        """
        num_examples = mus.shape[1]
        
        if num_examples < 2:
            # Can't measure agreement with < 2 examples
            return torch.ones(mus.shape[0], device=mus.device)
        
        # Compute pairwise KL divergences
        kl_sum = 0
        count = 0
        
        for i in range(num_examples):
            for j in range(i + 1, num_examples):
                # KL[q_i || q_j]
                mu_i, logvar_i = mus[:, i], logvars[:, i]
                mu_j, logvar_j = mus[:, j], logvars[:, j]
                
                var_ratio = torch.exp(logvar_i - logvar_j)
                mean_diff_sq = (mu_i - mu_j) ** 2 / torch.exp(logvar_j)
                kl = 0.5 * (var_ratio + mean_diff_sq - 1 - (logvar_i - logvar_j))
                kl_sum += kl.sum(dim=-1)
                count += 1
        
        # Average KL (lower = more agreement)
        avg_kl = kl_sum / count
        
        # Convert to agreement score (higher = better)
        agreement = torch.exp(-avg_kl)
        
        return agreement


class Decoder(nn.Module):
    """Decoder: Takes latent θ + input → generates output"""
    
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128, max_length: int = 20):
        super().__init__()
        
        # Encode input
        self.input_encoder = nn.LSTM(1, hidden_dim // 2, batch_first=True)
        
        # Combine latent with encoded input
        self.latent_processor = nn.Linear(latent_dim + hidden_dim // 2, hidden_dim)
        
        # Decode to output
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
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
        
        # Encode input
        inputs_expanded = inputs.unsqueeze(-1)
        _, (h_in, _) = self.input_encoder(inputs_expanded)
        input_enc = h_in[-1]  # [batch, hidden//2]
        
        # Combine with latent
        combined = torch.cat([z, input_enc], dim=-1)
        processed = self.latent_processor(combined)
        
        # Expand to sequence
        processed = processed.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # Decode
        decoded, _ = self.decoder_lstm(processed)
        outputs = self.output_layer(decoded).squeeze(-1)
        
        return outputs


class ProductOfExpertsLPN(nn.Module):
    """
    Complete Product of Experts Latent Program Network
    """
    
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128, max_length: int = 20):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Components
        self.single_encoder = SingleExampleEncoder(
            input_dim=1,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_length=max_length
        )
        
        self.poe_combiner = ProductOfExpertsCombiner(latent_dim=latent_dim)
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_length=max_length
        )
    
    def encode_examples(self, train_inputs: torch.Tensor, train_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode multiple examples with Product of Experts
        
        Args:
            train_inputs: [batch, num_examples, seq_len]
            train_outputs: [batch, num_examples, seq_len]
        Returns:
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
            agreement: [batch] - consistency score
        """
        batch_size, num_examples = train_inputs.shape[:2]
        
        # Encode each example independently
        mus = []
        logvars = []
        
        for i in range(num_examples):
            mu_i, logvar_i = self.single_encoder(
                train_inputs[:, i],
                train_outputs[:, i]
            )
            mus.append(mu_i)
            logvars.append(logvar_i)
        
        # Stack: [batch, num_examples, latent_dim]
        mus = torch.stack(mus, dim=1)
        logvars = torch.stack(logvars, dim=1)
        
        # Combine via PoE
        mu_combined, logvar_combined = self.poe_combiner.combine(mus, logvars)
        
        # Measure agreement
        agreement = self.poe_combiner.compute_agreement(mus, logvars)
        
        return mu_combined, logvar_combined, agreement
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, train_inputs: torch.Tensor, train_outputs: torch.Tensor, 
                test_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            predictions: [batch, seq_len]
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
            agreement: [batch]
        """
        # Encode with PoE
        mu, logvar, agreement = self.encode_examples(train_inputs, train_outputs)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        predictions = self.decoder(z, test_inputs)
        
        return predictions, mu, logvar, agreement
    
    def predict_with_search(self, train_inputs: torch.Tensor, train_outputs: torch.Tensor,
                           train_masks: torch.Tensor, test_inputs: torch.Tensor,
                           num_steps: int = 50, lr: float = 0.1) -> torch.Tensor:
        """
        Test-time optimization in latent space
        Same as baseline LPN but starts from PoE-combined θ
        """
        batch_size = train_inputs.shape[0]
        
        # Initialize from PoE encoding
        with torch.no_grad():
            mu, logvar, _ = self.encode_examples(train_inputs, train_outputs)
            z_init = mu.clone()
        
        # Make z optimizable
        z = torch.nn.Parameter(z_init.clone().detach().requires_grad_(True))
        optimizer = torch.optim.Adam([z], lr=lr)
        
        # Optimize z to fit training examples
        with torch.enable_grad():
            for step in range(num_steps):
                optimizer.zero_grad()
                
                num_train = train_inputs.shape[1]
                loss = 0
                
                for i in range(num_train):
                    pred = self.decoder(z, train_inputs[:, i, :])
                    mask = train_masks[:, i, :]
                    loss += F.mse_loss(pred * mask, train_outputs[:, i, :] * mask)
                
                loss = loss / num_train
                loss.backward()
                optimizer.step()
        
        # Use optimized z for prediction
        with torch.no_grad():
            predictions = self.decoder(z, test_inputs)
        
        return predictions


def compute_poe_loss(predictions: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor,
                    mu: torch.Tensor, logvar: torch.Tensor,
                    prior_mu: torch.Tensor, prior_logvar: torch.Tensor,
                    beta: float = 0.1) -> Dict[str, torch.Tensor]:
    """
    Compute ELBO loss for PoE-LPN
    
    ELBO = E[log p(y|x,θ)] - KL[q(θ|data) || p(θ)]
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(predictions * masks, targets * masks)
    
    # KL divergence: KL[q(θ|data) || p(θ)]
    var_ratio = torch.exp(logvar - prior_logvar)
    mean_diff_sq = (mu - prior_mu) ** 2 / torch.exp(prior_logvar)
    kl_loss = 0.5 * (var_ratio + mean_diff_sq - 1 - (logvar - prior_logvar))
    kl_loss = kl_loss.sum(dim=-1).mean()
    
    # Total loss (negative ELBO to minimize)
    total_loss = recon_loss + beta * kl_loss
    
    return {
        'total': total_loss,
        'recon': recon_loss,
        'kl': kl_loss
    }


if __name__ == "__main__":
    print("Product of Experts LPN")
    print("=" * 60)
    
    # Test model
    model = ProductOfExpertsLPN(latent_dim=64, hidden_dim=128)
    
    # Dummy data
    batch_size = 4
    num_examples = 3
    seq_len = 10
    
    train_inputs = torch.randn(batch_size, num_examples, seq_len)
    train_outputs = torch.randn(batch_size, num_examples, seq_len)
    test_inputs = torch.randn(batch_size, seq_len)
    
    # Forward pass
    predictions, mu, logvar, agreement = model(train_inputs, train_outputs, test_inputs)
    
    print(f"✓ Model initialized")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Latent dim: {model.latent_dim}")
    print(f"\n✓ Forward pass successful")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Latent mu shape: {mu.shape}")
    print(f"  Agreement scores: {agreement}")
    print(f"\n✓ Ready for training!")
