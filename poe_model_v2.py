"""
IMPROVED Product of Experts LPN with Enhanced Decoder
Fixes: Better latent integration, stronger decoder capacity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from poe_model import SingleExampleEncoder, ProductOfExpertsCombiner
from typing import Tuple, Dict


class ImprovedDecoder(nn.Module):
    """
    Improved Decoder with better latent integration
    """
    
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128, max_length: int = 20):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Process latent to hidden state
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)  # For both h and c
        )
        
        # Encode input with latent conditioning
        self.input_processor = nn.Linear(1 + latent_dim, hidden_dim)
        
        # Decoder LSTM with latent-conditioned initial state
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch, latent_dim]
            inputs: [batch, seq_len]
        Returns:
            outputs: [batch, seq_len]
        """
        batch_size, seq_len = inputs.shape
        
        # Process latent to LSTM initial state
        latent_hidden = self.latent_to_hidden(z)  # [batch, hidden*2]
        h0 = latent_hidden[:, :self.hidden_dim].unsqueeze(0).repeat(2, 1, 1)  # [2, batch, hidden]
        c0 = latent_hidden[:, self.hidden_dim:].unsqueeze(0).repeat(2, 1, 1)
        
        # Expand latent to sequence length
        z_expanded = z.unsqueeze(1).expand(batch_size, seq_len, self.latent_dim)
        
        # Concatenate input with latent at each timestep
        inputs_expanded = inputs.unsqueeze(-1)  # [batch, seq_len, 1]
        combined_input = torch.cat([inputs_expanded, z_expanded], dim=-1)  # [batch, seq_len, 1+latent_dim]
        
        # Process combined input
        processed = self.input_processor(combined_input)  # [batch, seq_len, hidden]
        
        # Decode with latent-initialized LSTM
        decoded, _ = self.decoder_lstm(processed, (h0, c0))  # [batch, seq_len, hidden]
        
        # Combine decoded with latent for output
        decoded_with_latent = torch.cat([decoded, z_expanded], dim=-1)
        outputs = self.output_layer(decoded_with_latent).squeeze(-1)
        
        return outputs


class ImprovedPoELPN(nn.Module):
    """
    Improved Product of Experts LPN with enhanced decoder
    """
    
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128, max_length: int = 20):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Use existing PoE components
        self.single_encoder = SingleExampleEncoder(
            input_dim=1,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_length=max_length
        )
        
        self.poe_combiner = ProductOfExpertsCombiner(latent_dim=latent_dim)
        
        # Use improved decoder
        self.decoder = ImprovedDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_length=max_length
        )
    
    def encode_examples(self, train_inputs: torch.Tensor, train_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same as original PoE"""
        batch_size, num_examples = train_inputs.shape[:2]
        
        mus = []
        logvars = []
        
        for i in range(num_examples):
            mu_i, logvar_i = self.single_encoder(
                train_inputs[:, i],
                train_outputs[:, i]
            )
            mus.append(mu_i)
            logvars.append(logvar_i)
        
        mus = torch.stack(mus, dim=1)
        logvars = torch.stack(logvars, dim=1)
        
        mu_combined, logvar_combined = self.poe_combiner.combine(mus, logvars)
        agreement = self.poe_combiner.compute_agreement(mus, logvars)
        
        return mu_combined, logvar_combined, agreement
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, train_inputs: torch.Tensor, train_outputs: torch.Tensor, 
                test_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, logvar, agreement = self.encode_examples(train_inputs, train_outputs)
        z = self.reparameterize(mu, logvar)
        predictions = self.decoder(z, test_inputs)
        
        return predictions, mu, logvar, agreement
    
    def predict_with_search(self, train_inputs: torch.Tensor, train_outputs: torch.Tensor,
                           train_masks: torch.Tensor, test_inputs: torch.Tensor,
                           num_steps: int = 50, lr: float = 0.1) -> torch.Tensor:
        """Test-time optimization"""
        with torch.no_grad():
            mu, logvar, _ = self.encode_examples(train_inputs, train_outputs)
            z_init = mu.clone()
        
        z = torch.nn.Parameter(z_init.clone().detach().requires_grad_(True))
        optimizer = torch.optim.Adam([z], lr=lr)
        
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
        
        with torch.no_grad():
            predictions = self.decoder(z, test_inputs)
        
        return predictions


# Alias for compatibility
ProductOfExpertsLPN_v2 = ImprovedPoELPN


if __name__ == "__main__":
    print("Improved Product of Experts LPN")
    print("=" * 60)
    
    # Test model
    model = ImprovedPoELPN(latent_dim=64, hidden_dim=128)
    
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
    print(f"\n✓ Enhanced decoder with better latent integration!")
