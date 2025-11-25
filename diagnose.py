"""
Diagnostic Script - Check why models aren't learning
"""

import torch
from torch.utils.data import DataLoader
from poe_model import ProductOfExpertsLPN
from lpn_model import ListOpsDataset, LatentProgramNetwork

# Load a small batch of data
dataset = ListOpsDataset('./list_ops_data/train.json')
loader = DataLoader(dataset, batch_size=4, shuffle=False)
batch = next(iter(loader))

print("=" * 80)
print("DIAGNOSTIC CHECK")
print("=" * 80)

# Check data
print("\n1. DATA CHECK:")
print(f"   Train inputs shape: {batch['train_inputs'].shape}")
print(f"   Train outputs shape: {batch['train_outputs'].shape}")
print(f"   Test inputs shape: {batch['test_inputs'].shape}")
print(f"   Test outputs shape: {batch['test_outputs'].shape}")
print(f"   Program types: {batch['program_type']}")

# Example input/output
print(f"\n   Example input: {batch['train_inputs'][0, 0, :10]}")
print(f"   Example output: {batch['train_outputs'][0, 0, :10]}")
print(f"   Example mask: {batch['train_masks'][0, 0, :10]}")

# Check PoE model
print("\n2. PoE MODEL CHECK:")
poe_model = ProductOfExpertsLPN(latent_dim=64, hidden_dim=128)
poe_model.eval()

with torch.no_grad():
    train_inputs = batch['train_inputs']
    train_outputs = batch['train_outputs']
    test_inputs = batch['test_inputs'][:, 0, :]
    
    predictions, mu, logvar, agreement = poe_model(train_inputs, train_outputs, test_inputs)
    
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"   Latent mu mean: {mu.mean():.4f}, std: {mu.std():.4f}")
    print(f"   Latent logvar mean: {logvar.mean():.4f}")
    print(f"   Agreement: {agreement}")
    
    print(f"\n   Example prediction: {predictions[0, :10]}")
    print(f"   Example target: {batch['test_outputs'][0, 0, :10]}")
    
    # Check if predictions are stuck
    if predictions.std() < 0.1:
        print(f"   ⚠️ WARNING: Predictions have very low variance!")
        print(f"      Model might be outputting constant values")

# Check baseline model
print("\n3. BASELINE MODEL CHECK:")
try:
    baseline = LatentProgramNetwork(latent_dim=64, hidden_dim=128)
    baseline.load_state_dict(torch.load('best_lpn_model.pt', map_location='cpu'))
    baseline.eval()
    
    with torch.no_grad():
        predictions, _, _ = baseline(train_inputs, train_outputs, test_inputs)
        print(f"   Baseline predictions shape: {predictions.shape}")
        print(f"   Baseline predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
        print(f"   Example prediction: {predictions[0, :10]}")
        
        if predictions.std() < 0.1:
            print(f"   ⚠️ WARNING: Baseline also has low variance!")
    
    print("   ✓ Baseline model loaded successfully")
except Exception as e:
    print(f"   ✗ Could not load baseline: {e}")

# Check a simple task
print("\n4. SIMPLE TASK CHECK:")
print(f"   Task: {batch['program_type'][0]}")
print(f"   Input: {batch['train_inputs'][0, 0, :10].tolist()}")
print(f"   Expected output: {batch['train_outputs'][0, 0, :10].tolist()}")
print(f"   PoE prediction: {predictions[0, :10].tolist()}")

# Check loss components
print("\n5. LOSS COMPONENTS:")
test_outputs = batch['test_outputs'][:, 0, :]
test_masks = batch['test_masks'][:, 0, :]

recon_loss = torch.nn.functional.mse_loss(
    predictions * test_masks,
    test_outputs * test_masks
)
kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]

print(f"   Reconstruction loss: {recon_loss:.4f}")
print(f"   KL loss: {kl_loss:.4f}")
print(f"   Total (beta=0.1): {recon_loss + 0.1 * kl_loss:.4f}")

if recon_loss > 100:
    print(f"   ⚠️ WARNING: Reconstruction loss is very high!")
    print(f"      Model is not learning the mapping")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)

if predictions.std() < 0.1:
    print("❌ Model outputs are nearly constant - decoder is broken or collapsed")
    print("   → Check decoder architecture")
    print("   → Try reducing beta (KL weight)")
    print("   → Try increasing learning rate")
elif recon_loss > 100:
    print("❌ Reconstruction loss is very high - model not learning")
    print("   → Check if latent is being used properly")
    print("   → Try training longer")
    print("   → Check decoder capacity")
else:
    print("✓ Model seems to be working, may just need more training")

print("=" * 80)
