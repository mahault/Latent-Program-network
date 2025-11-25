# Product of Experts for LPN - Implementation Guide

## 🎯 Core Idea

**Problem:** Given N examples of a task, find the latent program θ that is **consistent with ALL examples**.

**Solution:** Each example provides a distribution over θ. Combine via Product of Experts to find the most consistent θ.

## 📐 Mathematical Foundation

### Single Example Inference

For one example (x, y):

```
p(θ | x, y) ∝ p(y | x, θ) · p(θ)
```

Approximate with Gaussian:
```
q(θ | x, y) = 𝒩(μ, σ²)
```

### Multiple Examples

For N examples {(x₁, y₁), ..., (xₙ, yₙ)}:

```
p(θ | X, Y) ∝ p(θ) · ∏ᵢ p(yᵢ | xᵢ, θ)
```

**Product of Experts assumption:**
```
q(θ | X, Y) ∝ ∏ᵢ q(θ | xᵢ, yᵢ)
```

For Gaussian factors, the product is also Gaussian:

```
q(θ | X, Y) = 𝒩(μ*, σ*²)

where:
  Precision: τ*ᵢ = 1/σ*ᵢ² = Σⱼ 1/σⱼ²
  Mean: μ*ᵢ = σ*ᵢ² · Σⱼ (μⱼ/σⱼ²)
```

## 🔨 Implementation

### Step 1: Single-Example Encoder

First, build an encoder that processes ONE example:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleExampleEncoder(nn.Module):
    """
    Encodes a single (input, output) pair to a distribution over θ
    """
    
    def __init__(self, input_dim=20, latent_dim=64, hidden_dim=128):
        super().__init__()
        
        # Process input and output separately
        self.input_processor = nn.LSTM(1, hidden_dim//2, batch_first=True)
        self.output_processor = nn.LSTM(1, hidden_dim//2, batch_first=True)
        
        # Combine and output distribution
        self.fc_combined = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, input_seq, output_seq):
        """
        Args:
            input_seq: [batch, seq_len]
            output_seq: [batch, seq_len]
        Returns:
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
        """
        batch_size = input_seq.shape[0]
        
        # Add channel dimension
        input_seq = input_seq.unsqueeze(-1)
        output_seq = output_seq.unsqueeze(-1)
        
        # Encode sequences
        _, (h_in, _) = self.input_processor(input_seq)
        _, (h_out, _) = self.output_processor(output_seq)
        
        # Combine
        combined = torch.cat([h_in[-1], h_out[-1]], dim=-1)
        combined = F.relu(self.fc_combined(combined))
        
        # Output distribution
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        
        return mu, logvar
```

### Step 2: Product of Experts Combiner

```python
class ProductOfExpertsCombiner:
    """
    Combines multiple Gaussian distributions via Product of Experts
    """
    
    @staticmethod
    def combine(mus, logvars, prior_mu=None, prior_logvar=None):
        """
        Combine multiple Gaussian q(θ|example_i) via product
        
        Args:
            mus: [batch, num_examples, latent_dim]
            logvars: [batch, num_examples, latent_dim]
            prior_mu: [latent_dim] (optional)
            prior_logvar: [latent_dim] (optional)
            
        Returns:
            combined_mu: [batch, latent_dim]
            combined_logvar: [batch, latent_dim]
        """
        batch_size, num_examples, latent_dim = mus.shape
        
        # Convert to precision (inverse variance)
        # τ = 1/σ² = exp(-log σ²) = exp(-logvar)
        precisions = torch.exp(-logvars)  # [batch, num_ex, latent]
        
        # Sum precisions
        total_precision = precisions.sum(dim=1)  # [batch, latent]
        
        # Add prior precision if provided
        if prior_mu is not None:
            prior_precision = torch.exp(-prior_logvar)
            total_precision = total_precision + prior_precision
        
        # Weighted sum of means
        # Each mean weighted by its precision
        weighted_mus = (mus * precisions).sum(dim=1)  # [batch, latent]
        
        # Add prior contribution
        if prior_mu is not None:
            prior_contribution = prior_mu * prior_precision
            weighted_mus = weighted_mus + prior_contribution
        
        # Combined parameters
        combined_mu = weighted_mus / total_precision
        combined_logvar = -torch.log(total_precision)
        
        return combined_mu, combined_logvar
    
    @staticmethod
    def compute_agreement(mus, logvars):
        """
        Measure how much the different examples agree
        Higher = more agreement = more confidence
        
        Returns:
            agreement: [batch] scalar measuring inter-example consistency
        """
        # Compute pairwise KL divergences
        num_examples = mus.shape[1]
        kl_sum = 0
        count = 0
        
        for i in range(num_examples):
            for j in range(i+1, num_examples):
                # KL[q_i || q_j]
                mu_i, logvar_i = mus[:, i], logvars[:, i]
                mu_j, logvar_j = mus[:, j], logvars[:, j]
                
                var_ratio = torch.exp(logvar_i - logvar_j)
                mean_diff_sq = (mu_i - mu_j)**2 / torch.exp(logvar_j)
                kl = 0.5 * (var_ratio + mean_diff_sq - 1 - (logvar_i - logvar_j))
                kl_sum += kl.sum(dim=-1)
                count += 1
        
        # Average KL (lower = more agreement)
        avg_kl = kl_sum / count
        
        # Convert to agreement score (higher = better)
        agreement = torch.exp(-avg_kl)
        
        return agreement
```

### Step 3: Full PoE-LPN Model

```python
class ProductOfExpertsLPN(nn.Module):
    """
    Complete LPN with Product of Experts inference
    """
    
    def __init__(self, latent_dim=64, hidden_dim=128, max_length=20):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Single example encoder
        self.example_encoder = SingleExampleEncoder(
            input_dim=max_length,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        
        # Decoder (same as before)
        self.decoder = Decoder(
            input_dim=1,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_length=max_length
        )
        
        # Learnable prior
        self.register_parameter(
            'prior_mu',
            nn.Parameter(torch.zeros(latent_dim))
        )
        self.register_parameter(
            'prior_logvar',
            nn.Parameter(torch.zeros(latent_dim))
        )
        
        self.poe_combiner = ProductOfExpertsCombiner()
    
    def encode(self, train_inputs, train_outputs):
        """
        Encode with Product of Experts
        
        Args:
            train_inputs: [batch, num_examples, seq_len]
            train_outputs: [batch, num_examples, seq_len]
            
        Returns:
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
            agreement: [batch] - measure of consistency
        """
        batch_size, num_examples = train_inputs.shape[:2]
        
        # Encode each example independently
        mus = []
        logvars = []
        
        for i in range(num_examples):
            mu_i, logvar_i = self.example_encoder(
                train_inputs[:, i],
                train_outputs[:, i]
            )
            mus.append(mu_i)
            logvars.append(logvar_i)
        
        # Stack: [batch, num_examples, latent_dim]
        mus = torch.stack(mus, dim=1)
        logvars = torch.stack(logvars, dim=1)
        
        # Combine via PoE
        mu_combined, logvar_combined = self.poe_combiner.combine(
            mus, logvars,
            self.prior_mu, self.prior_logvar
        )
        
        # Compute agreement (for monitoring)
        agreement = self.poe_combiner.compute_agreement(mus, logvars)
        
        return mu_combined, logvar_combined, agreement
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, train_inputs, train_outputs, test_inputs):
        """
        Forward pass
        
        Returns:
            predictions: [batch, seq_len]
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
            agreement: [batch]
        """
        # Encode with PoE
        mu, logvar, agreement = self.encode(train_inputs, train_outputs)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        predictions = self.decoder(z, test_inputs)
        
        return predictions, mu, logvar, agreement
```

### Step 4: Training Loop

```python
def train_poe_lpn(model, train_loader, optimizer, device, beta=0.1):
    """Training loop with PoE-LPN"""
    
    model.train()
    total_loss = 0
    total_agreement = 0
    
    for batch in train_loader:
        train_inputs = batch['train_inputs'].to(device)
        train_outputs = batch['train_outputs'].to(device)
        train_masks = batch['train_masks'].to(device)
        test_inputs = batch['test_inputs'][:, 0].to(device)
        test_outputs = batch['test_outputs'][:, 0].to(device)
        test_masks = batch['test_masks'][:, 0].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred, mu, logvar, agreement = model(
            train_inputs, train_outputs, test_inputs
        )
        
        # Compute loss
        recon_loss = F.mse_loss(
            pred * test_masks,
            test_outputs * test_masks
        )
        
        # KL with respect to prior
        var_ratio = torch.exp(logvar - model.prior_logvar)
        mean_diff_sq = (mu - model.prior_mu)**2 / torch.exp(model.prior_logvar)
        kl_loss = 0.5 * (var_ratio + mean_diff_sq - 1 - (logvar - model.prior_logvar))
        kl_loss = kl_loss.sum(dim=-1).mean()
        
        # Total loss
        loss = recon_loss + beta * kl_loss
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_agreement += agreement.mean().item()
    
    return {
        'loss': total_loss / len(train_loader),
        'agreement': total_agreement / len(train_loader)
    }
```

## 📊 Visualizing Product of Experts

```python
def visualize_poe(model, task, save_path='poe_visualization.png'):
    """
    Visualize how PoE combines evidence from multiple examples
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    
    with torch.no_grad():
        train_inputs = task['train_inputs'].unsqueeze(0)
        train_outputs = task['train_outputs'].unsqueeze(0)
        
        num_examples = train_inputs.shape[1]
        
        # Encode each example individually
        individual_mus = []
        individual_logvars = []
        
        for i in range(num_examples):
            mu, logvar = model.example_encoder(
                train_inputs[:, i],
                train_outputs[:, i]
            )
            individual_mus.append(mu[0].cpu().numpy())
            individual_logvars.append(logvar[0].cpu().numpy())
        
        # Combine with PoE
        mus_tensor = torch.stack([torch.tensor(m) for m in individual_mus]).unsqueeze(0)
        logvars_tensor = torch.stack([torch.tensor(lv) for lv in individual_logvars]).unsqueeze(0)
        
        combined_mu, combined_logvar = model.poe_combiner.combine(
            mus_tensor, logvars_tensor,
            model.prior_mu, model.prior_logvar
        )
        
        combined_mu = combined_mu[0].cpu().numpy()
        combined_logvar = combined_logvar[0].cpu().numpy()
        
        # Plot first 10 dimensions
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for dim in range(min(10, model.latent_dim)):
            ax = axes[dim]
            
            # Plot individual distributions
            for i in range(num_examples):
                mu = individual_mus[i][dim]
                std = np.exp(0.5 * individual_logvars[i][dim])
                x = np.linspace(mu - 3*std, mu + 3*std, 100)
                y = np.exp(-0.5 * ((x - mu) / std)**2) / (std * np.sqrt(2*np.pi))
                ax.plot(x, y, alpha=0.5, label=f'Ex {i+1}')
            
            # Plot combined distribution
            mu_comb = combined_mu[dim]
            std_comb = np.exp(0.5 * combined_logvar[dim])
            x = np.linspace(mu_comb - 3*std_comb, mu_comb + 3*std_comb, 100)
            y = np.exp(-0.5 * ((x - mu_comb) / std_comb)**2) / (std_comb * np.sqrt(2*np.pi))
            ax.plot(x, y, 'k-', linewidth=2, label='PoE Combined')
            
            ax.set_title(f'Dimension {dim}')
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved PoE visualization to {save_path}")
```

## 🎯 When Product of Experts Helps

### Scenario 1: Ambiguous Single Examples

```
Example 1: [1, 2, 3] → [3, 2, 1]
Could be: reverse, sort_descending, or custom permutation

Example 2: [5, 1, 3] → [5, 3, 1]  
Confirms: sort_descending (not reverse)

PoE finds: θ for sort_descending
```

### Scenario 2: Noisy Examples

```
Example 1: Slightly off (noise in data)
Example 2: Perfect
Example 3: Slightly off

PoE: Downweights low-precision examples
```

### Scenario 3: Compositional Tasks

```
Example 1: Shows "filter positive"
Example 2: Shows "then square"
Example 3: Shows full "filter positive then square"

PoE: Combines evidence for composition
```

## 🔍 Debugging PoE

### Check 1: Individual Example Encoding

```python
# Do individual examples produce reasonable distributions?
mu1, logvar1 = model.example_encoder(input1, output1)
print(f"Example 1 mean: {mu1[0, :5]}")  # First 5 dims
print(f"Example 1 std: {torch.exp(0.5*logvar1[0, :5])}")
```

### Check 2: Agreement Score

```python
# Are examples consistent with each other?
_, _, agreement = model.encode(train_inputs, train_outputs)
print(f"Agreement: {agreement.item():.3f}")  # Should be > 0.5
```

### Check 3: PoE vs. Averaging

```python
# Compare PoE to simple averaging
mu_poe, _ = poe_combine(mus, logvars)
mu_avg = mus.mean(dim=1)

print(f"PoE mean: {mu_poe[0, :5]}")
print(f"Avg mean: {mu_avg[0, :5]}")
# PoE should be different (sharper) when examples agree
```

## 📈 Expected Improvements

With PoE, you should see:

1. **Better test accuracy** (+5-10%) especially with more examples
2. **Reduced uncertainty** (σ² decreases with more examples)
3. **Faster convergence** (fewer epochs needed)
4. **Better few-shot** (works with just 1-2 examples)

## 🚀 Next Steps

1. Implement PoE for current list operations
2. Compare to baseline amortized inference
3. Visualize latent space structure
4. Extend to spatial (grid) data

---

**See also:**
- [BAYESIAN_APPROACH.md](BAYESIAN_APPROACH.md) - Theoretical foundation
- [ROADMAP.md](ROADMAP.md) - Development timeline
