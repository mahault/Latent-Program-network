# Bayesian Approach to Latent Program Networks

## 🎯 Motivation

Traditional LPNs use a **single amortized inference** network to map examples to programs. However, this has limitations:

1. **No uncertainty quantification** - Network outputs point estimates
2. **Inconsistent predictions** - Different examples might suggest different programs
3. **No model selection** - Can't choose between competing explanations
4. **Limited compositionality** - Hard to combine evidence from multiple sources

Our **Bayesian approach** addresses these by treating program inference as **Bayesian inference over a latent variable θ**.

## 📊 Probabilistic Framework

### Generative Model

```
p(y | x, a, θ) = Decoder(x, a; θ)
```

Where:
- **x**: Input (grid, sequence, etc.)
- **a**: Action (optional)
- **y**: Output
- **θ**: Latent program (continuous vector or structured)

### Inference Model

Instead of a single deterministic encoder, we have:

```
q(θ | x₁:ₙ, y₁:ₙ, a₁:ₙ) = ?
```

**Multiple strategies:**

#### 1. **Amortized (Current baseline)**
```python
q(θ | examples) = Encoder(examples)
# Fast but potentially suboptimal
```

#### 2. **Semi-Amortized (LPN paper)**
```python
# Initialize with encoder
θ₀ = Encoder(examples)

# Refine via gradient ascent
for t in 1..T:
    θₜ = θₜ₋₁ + lr * ∇_θ log p(y | x, θ)
    
# Use refined θₜ
```

#### 3. **Product of Experts (Our approach)**
```python
# Each example gives a distribution
q(θ | example_i) = 𝒩(μᵢ, σᵢ²)

# Combine via product
q(θ | all examples) ∝ ∏ᵢ q(θ | example_i)

# For Gaussians:
σ_combined² = 1 / Σᵢ (1/σᵢ²)
μ_combined = σ_combined² * Σᵢ (μᵢ/σᵢ²)
```

## 🔬 Product of Experts Deep Dive

### Why Product of Experts?

Consider a task with 3 examples:
- Example 1: Suggests θ could be "reverse" or "sort"
- Example 2: Rules out "reverse", consistent with "sort"
- Example 3: Confirms "sort"

**Product of Experts finds the θ consistent with ALL examples.**

### Mathematical Foundation

Each example i provides evidence:

```
p(θ | x_i, y_i) ∝ p(y_i | x_i, θ) * p(θ)
```

Combining all examples:

```
p(θ | X, Y) ∝ p(θ) * ∏ᵢ p(y_i | x_i, θ)
           = p(θ) * ∏ᵢ p(θ | x_i, y_i) / p(θ)^(n-1)
```

For Gaussian approximations, this becomes:

```
q(θ | all) = 𝒩(μ*, σ*²)

where:
  1/σ*² = 1/σ_prior² + Σᵢ 1/σᵢ²
  μ* = σ*² * (μ_prior/σ_prior² + Σᵢ μᵢ/σᵢ²)
```

### Implementation

```python
class ProductOfExpertsLPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.single_example_encoder = Encoder()
        self.decoder = Decoder()
        
        # Prior over θ
        self.prior_mu = nn.Parameter(torch.zeros(latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(latent_dim))
    
    def encode_with_poe(self, inputs, outputs):
        """
        Args:
            inputs: [batch, num_examples, ...]
            outputs: [batch, num_examples, ...]
        Returns:
            mu_combined: [batch, latent_dim]
            logvar_combined: [batch, latent_dim]
        """
        batch_size, num_examples = inputs.shape[:2]
        
        # Encode each example independently
        mus = []
        logvars = []
        
        for i in range(num_examples):
            mu_i, logvar_i = self.single_example_encoder(
                inputs[:, i], 
                outputs[:, i]
            )
            mus.append(mu_i)
            logvars.append(logvar_i)
        
        # Stack: [batch, num_examples, latent_dim]
        mus = torch.stack(mus, dim=1)
        logvars = torch.stack(logvars, dim=1)
        
        # Combine via PoE
        # Precision = 1/variance = exp(-logvar)
        precisions = torch.exp(-logvars)  # [batch, num_ex, latent]
        
        # Prior precision
        prior_precision = torch.exp(-self.prior_logvar)
        
        # Total precision
        total_precision = prior_precision + precisions.sum(dim=1)
        
        # Weighted sum of means
        weighted_mus = (mus * precisions).sum(dim=1)
        prior_weighted = self.prior_mu * prior_precision
        
        # Combined parameters
        mu_combined = (prior_weighted + weighted_mus) / total_precision
        logvar_combined = -torch.log(total_precision)
        
        return mu_combined, logvar_combined
    
    def forward(self, train_inputs, train_outputs, test_inputs):
        # Encode with PoE
        mu, logvar = self.encode_with_poe(train_inputs, train_outputs)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        predictions = self.decoder(z, test_inputs)
        
        return predictions, mu, logvar
```

## 🎲 Comparison: Different Inference Strategies

| Strategy | Pros | Cons | When to Use |
|----------|------|------|-------------|
| **Amortized** | Fast, one forward pass | Potentially suboptimal | Quick prototyping, simple tasks |
| **Semi-Amortized** | Better fit to data | Slow (50+ gradient steps) | When test accuracy matters |
| **Product of Experts** | Consistent with all examples | Requires independent encoders | Multiple conflicting examples |
| **Recursive (TRM)** | Iterative refinement | Requires special architecture | Complex compositional tasks |

## 🧮 Training Objective

### ELBO (Evidence Lower Bound)

```
ELBO = 𝔼_q(θ) [log p(y | x, θ)] - KL[q(θ) || p(θ)]
       ↑                            ↑
   Reconstruction                Regularization
```

**For Product of Experts:**

```python
def compute_elbo(predictions, targets, mu, logvar, prior_mu, prior_logvar):
    # Reconstruction loss
    recon_loss = F.mse_loss(predictions, targets)
    
    # KL divergence: KL[q(θ|data) || p(θ)]
    # For Gaussians: KL = 0.5 * (σ₁²/σ₀² + (μ₁-μ₀)²/σ₀² - 1 - log(σ₁²/σ₀²))
    var_ratio = torch.exp(logvar - prior_logvar)
    mean_diff_sq = (mu - prior_mu)**2 / torch.exp(prior_logvar)
    kl_loss = 0.5 * (var_ratio + mean_diff_sq - 1 - (logvar - prior_logvar))
    kl_loss = kl_loss.sum(dim=-1).mean()
    
    # Total ELBO (negative, to minimize)
    elbo = recon_loss + beta * kl_loss
    
    return elbo, recon_loss, kl_loss
```

## 🔍 Model Selection via Bayesian Inference

A key advantage: we can compare different model complexities!

### Marginal Likelihood

```
p(Y | X) = ∫ p(Y | X, θ) p(θ) dθ
```

This is approximated by the ELBO:

```
log p(Y | X) ≥ ELBO
```

**Use this to select:**
- Number of latent dimensions
- Number of object slots
- Complexity of dynamics

### Example: Choosing Latent Dimension

```python
latent_dims = [16, 32, 64, 128]
elbos = []

for dim in latent_dims:
    model = ProductOfExpertsLPN(latent_dim=dim)
    train(model, data)
    elbo = evaluate_elbo(model, validation_data)
    elbos.append(elbo)

# Higher ELBO = better model
best_dim = latent_dims[np.argmax(elbos)]
```

## 🌳 Hierarchical Models

Extend to hierarchical structure:

```
θ = [θ_global, θ_obj1, θ_obj2, ...]

where:
  θ_global: Task-level program (e.g., "move all red objects left")
  θ_obj_i: Object-level state (position, color, etc.)
```

**Benefits:**
- Compositional reasoning
- Better generalization
- Interpretable structure

## 🎯 Active Inference Connection

Our framework naturally extends to **Active Inference**:

### Free Energy Minimization

```
F(θ, a) = -log p(y | x, a, θ) + KL[q(θ) || p(θ)]
```

**Choose actions to minimize expected free energy:**

```
a* = argmin_a 𝔼_q(θ) [F(θ, a)]
```

This naturally handles:
- Exploration (reduce uncertainty about θ)
- Exploitation (achieve desired outcomes)

### Implementation Sketch

```python
def select_action_active_inference(model, current_state, goal_state):
    # For each possible action
    action_scores = []
    
    for action in possible_actions:
        # Predict next state
        predicted_state = model.decoder(theta, current_state, action)
        
        # Pragmatic value: similarity to goal
        pragmatic = -F.mse_loss(predicted_state, goal_state)
        
        # Epistemic value: reduction in θ uncertainty
        current_entropy = 0.5 * torch.log(2 * π * e * torch.exp(logvar))
        # Simulate future entropy after observing outcome
        # epistemic = current_entropy - expected_future_entropy
        epistemic = estimate_information_gain(...)
        
        # Total expected free energy
        efe = -pragmatic - epistemic
        action_scores.append(efe)
    
    # Select action with lowest EFE
    best_action = possible_actions[np.argmin(action_scores)]
    return best_action
```

## 📈 Empirical Validation

### What to Measure

1. **Test-time uncertainty**
   - Does σ² decrease as we see more examples?
   - Are uncertain predictions less accurate?

2. **Consistency**
   - Do different subsets of examples give similar θ?
   - Product of Experts should be more consistent

3. **Sample efficiency**
   - How many examples needed for good performance?
   - PoE should need fewer examples

4. **Compositional generalization**
   - Can model combine learned sub-programs?
   - Hierarchical models should excel here

### Experimental Protocol

```python
# 1. Train with different inference methods
models = {
    'amortized': AmortizedLPN(),
    'semi_amortized': SemiAmortizedLPN(),
    'poe': ProductOfExpertsLPN(),
}

# 2. Evaluate on held-out tasks
for name, model in models.items():
    # Vary number of examples
    for num_examples in [1, 3, 5, 10]:
        accuracy = evaluate(model, test_data, num_examples)
        print(f"{name} with {num_examples} ex: {accuracy:.2f}")
    
    # Measure uncertainty calibration
    uncertainty = compute_predictive_uncertainty(model, test_data)
    plot_uncertainty_vs_accuracy(uncertainty, accuracy)
```

## 🎓 Further Reading

**Bayesian Deep Learning:**
- Blundell et al. (2015) - Weight Uncertainty in Neural Networks
- Gal & Ghahramani (2016) - Dropout as Bayesian Approximation

**Product of Experts:**
- Hinton (2002) - Training Products of Experts
- Wu et al. (2018) - Multimodal Generative Models for PoE

**Active Inference:**
- Friston et al. (2017) - Active Inference: A Process Theory
- Millidge et al. (2021) - Deep Active Inference

**Program Induction:**
- Ellis et al. (2021) - DreamCoder
- Nye et al. (2021) - Improving Coherence and Consistency

---

**Next:** See [PRODUCT_OF_EXPERTS.md](PRODUCT_OF_EXPERTS.md) for implementation details.
