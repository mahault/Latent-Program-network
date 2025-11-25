# 🔧 TROUBLESHOOTING GUIDE - Fixing the Issues

## 📊 What Happened

You ran the code and got these results:

### Track 1 (Product of Experts):
- ✅ Completed training
- ❌ **Validation accuracy: 2.89%** (expected: 75-85%)
- ❌ **PoE worse than baseline** (-15.38%)
- ✅ Agreement scores good (0.93-0.95)
- ❌ Model not actually learning the transformations

### Track 2 (Spatial):
- ❌ **Crashed with tensor reshape error**
- Fixed: Changed `.view()` to `.reshape()`

---

## 🐛 Root Causes

### Problem 1: PoE Model Not Learning
**Why:** The decoder is too simple and doesn't properly use the latent information.

**Evidence:**
- All models (baseline AND PoE) getting 2-3% accuracy
- Only "count" program has any success (21%)
- Agreement is high but predictions are wrong

**Causes:**
1. Decoder doesn't integrate latent well enough
2. Beta (KL weight) might be too high (0.1)
3. Model architecture may need more capacity

### Problem 2: Spatial Model Tensor Error
**Why:** `.view()` requires contiguous tensors, but after conv operations tensor wasn't contiguous.

**Fix:** Changed to `.reshape()` which handles non-contiguous tensors.

---

## ✅ Fixes Applied

### Fix 1: Spatial Model (DONE)
```python
# Changed in spatial_model.py line 74-75:
# OLD:
input_features = input_features.view(batch_size * num_examples, -1)

# NEW:
input_features = input_features.reshape(batch_size * num_examples, -1)
```

### Fix 2: Improved PoE Model (NEW FILE)
Created `poe_model_v2.py` with:
- **Better latent integration**: Latent used at every decoder timestep
- **Latent-conditioned LSTM**: Initial state from latent
- **Stronger decoder**: More capacity and skip connections
- **Better output**: Combines decoded features with latent

---

## 🚀 What To Do Next

### Option 1: Diagnose Current Model (RECOMMENDED FIRST)
```bash
# Run diagnostic to see what's wrong
python diagnose.py
```

This will tell you:
- If predictions are stuck (constant output)
- If losses are reasonable
- If latent is being used

### Option 2: Try Improved PoE Model
```bash
# Train with improved model
python train_poe.py --num_epochs 50 --beta 0.01  # Lower beta!

# Or use the v2 model (need to update imports)
```

### Option 3: Reduce Beta (Quick Fix)
```bash
# Current beta=0.1 might be too high, crushing reconstruction
python train_poe.py --num_epochs 50 --beta 0.01  # 10x smaller
```

### Option 4: Try Spatial Model Again
```bash
# Now that reshape bug is fixed
python train_spatial.py --num_epochs 30 --batch_size 16
```

---

## 🎯 Quick Diagnosis Commands

```bash
# 1. Check what's wrong
python diagnose.py

# 2. Try lower beta for PoE
python train_poe.py --num_epochs 20 --beta 0.01 --lr 0.001

# 3. Try spatial model (bug fixed)
python train_spatial.py --num_epochs 10 --batch_size 16

# 4. Compare baseline vs PoE
python compare_methods.py
```

---

## 📋 Expected Results After Fixes

### With Lower Beta (--beta 0.01):
- Validation accuracy: 40-60% (better than 2.89%)
- Reconstruction loss: Should decrease significantly
- Model actually learns mappings

### With Improved Decoder (poe_model_v2.py):
- Validation accuracy: 60-75%
- Better latent utilization
- Closer to expected performance

### Spatial Model:
- Should train without crashing
- Grid accuracy: >80% (it's easy tasks)
- Pixel accuracy: >95%

---

## 🔍 How to Use Improved Model

### Method 1: Quick Test
```python
# In Python:
from poe_model_v2 import ImprovedPoELPN

model = ImprovedPoELPN(latent_dim=64, hidden_dim=128)
# Train as normal
```

### Method 2: Update train_poe.py
Replace import:
```python
# OLD:
from poe_model import ProductOfExpertsLPN

# NEW:
from poe_model_v2 import ImprovedPoELPN as ProductOfExpertsLPN
```

---

## 💡 Understanding the Issues

### Why Beta Matters:
```
Total Loss = Reconstruction + Beta * KL

If Beta too high (0.1):
  - KL dominates
  - Model focuses on matching prior
  - Ignores reconstruction
  - Accuracy suffers

If Beta too low (0.001):
  - No regularization
  - May overfit
  - But at least learns!

Sweet spot: 0.01 - 0.05
```

### Why Decoder Matters:
```
Simple Decoder (current):
  z → [concat with input] → LSTM → output
  Problem: z used once, then ignored

Improved Decoder (v2):
  z → [initialize LSTM state]
  z → [concat at each timestep]
  z → [concat with output]
  Better: z used throughout decoding
```

---

## 🎯 Action Plan

**Immediate (next 5 minutes):**
1. Run `python diagnose.py` to see the issue
2. Note what the diagnostic says

**Short-term (next hour):**
3. Try training with lower beta:
   ```bash
   python train_poe.py --num_epochs 20 --beta 0.01
   ```
4. Try spatial model:
   ```bash
   python train_spatial.py --num_epochs 10
   ```

**Medium-term (today):**
5. If still struggling, use improved model (poe_model_v2.py)
6. Compare results

---

## 📞 Troubleshooting Checklist

Before asking for help, check:

- [ ] Ran `python diagnose.py`
- [ ] Tried lower beta (0.01 instead of 0.1)
- [ ] Checked if spatial model now works
- [ ] Verified data files exist and are correct
- [ ] Checked that baseline model also has low accuracy
      (if yes → problem is universal, not just PoE)

---

## 🆘 If Still Not Working

### Check These:

1. **Data Issue?**
   ```python
   # Quick data check
   from lpn_model import ListOpsDataset
   ds = ListOpsDataset('./list_ops_data/train.json')
   print(len(ds))  # Should be 2070
   sample = ds[0]
   print(sample['program_type'])  # Should show program name
   ```

2. **Model Loading Issue?**
   ```bash
   # Fresh start
   rm best_poe_model.pt
   python train_poe.py --num_epochs 20 --beta 0.01
   ```

3. **CPU vs GPU?**
   ```bash
   # Force CPU (more stable)
   python train_poe.py --device cpu --beta 0.01
   ```

---

## 📊 Success Indicators

You'll know it's working when:

- ✅ Validation accuracy > 40% (at least!)
- ✅ Training loss decreases smoothly
- ✅ Reconstruction loss < 50
- ✅ Model outputs vary (not constant)
- ✅ Some programs get >70% accuracy

---

**Start with:** `python diagnose.py`

Then try: `python train_poe.py --beta 0.01 --num_epochs 20`
