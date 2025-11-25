# Complete Setup and Running Guide

## 🚀 Quick Start (5 minutes)

### Step 1: Install Dependencies
```bash
pip install torch numpy matplotlib seaborn tqdm
```

Or use the automated setup:
```bash
python setup.py
```

### Step 2: Run Experiments
```bash
# Quick test (10 epochs, ~5 minutes)
python run_experiments.py --experiment baseline --quick

# Full baseline experiment (~20 minutes)
python run_experiments.py --experiment baseline

# Full pipeline (all experiments, ~2 hours)
python run_experiments.py --experiment all
```

### Step 3: View Results
```bash
# Results are in:
analysis_outputs/           # Visualizations
test_results.json          # Detailed metrics
*.pt                       # Trained models
```

---

## 📋 Detailed Setup

### Prerequisites
- Python 3.8+
- 4GB+ RAM
- GPU optional (10x faster training)

### Installation

#### Option 1: Automated Setup (Recommended)
```bash
python setup.py
```

This will:
- ✓ Check and install dependencies
- ✓ Generate list operations dataset (3,000 tasks)
- ✓ Generate synthetic grid dataset (600 tasks)
- ✓ Create sample ARC tasks

#### Option 2: Manual Setup
```bash
# 1. Install packages
pip install torch numpy matplotlib seaborn tqdm

# 2. Generate data
python generate_list_data.py
python generate_grid_data.py
python arc_data.py

# 3. Verify
ls list_ops_data/        # Should see train.json, val.json, test.json
ls synthetic_grid_data/  # Should see train.json, val.json, test.json
ls arc_data/training/    # Should see sample_*.json files
```

---

## 🎯 Running Experiments

### Experiment 1: Baseline LPN (List Operations)

**What it does:** Trains LSTM-based LPN on list transformations

```bash
python run_experiments.py --experiment baseline
```

Or manually:
```bash
# Train
python train_lpn.py --num_epochs 50

# Test
python test_lpn.py

# Analyze
python analyze_results.py
```

**Expected results:**
- Training: ~20 minutes (CPU), ~5 minutes (GPU)
- Validation accuracy: 70-85%
- Test accuracy (no search): 70-80%
- Test accuracy (with search): 75-85%

**Outputs:**
- `best_lpn_model.pt` - Trained model
- `training_history.json` - Training metrics
- `test_results.json` - Test metrics
- `analysis_outputs/` - Visualizations

---

### Experiment 2: Product of Experts LPN

**What it does:** Trains PoE-LPN with Bayesian inference

```bash
python run_experiments.py --experiment poe
```

Or manually:
```bash
# Train
python train_poe.py --num_epochs 50

# Test
python test_poe.py
```

**Expected results:**
- Training: ~25 minutes (CPU), ~6 minutes (GPU)
- Validation accuracy: 75-88%
- Test accuracy (no search): 75-85%
- Test accuracy (with search): 80-90%
- **Improvement over baseline: +5-10%**

**Outputs:**
- `best_poe_model.pt` - Trained model
- `poe_training_history.json` - Training metrics
- `poe_test_results.json` - Test metrics

---

### Experiment 3: Compare Baseline vs PoE

**What it does:** Side-by-side comparison of both methods

```bash
python run_experiments.py --experiment compare
```

Or manually:
```bash
# (Requires both models trained)
python compare_methods.py
```

**Expected results:**
- Comparison table showing accuracy improvements
- Key insights on which method is better
- Recommendations for production use

**Outputs:**
- `comparison_results.json` - Detailed comparison

---

### Experiment 4: Spatial LPN (Grid Tasks)

**What it does:** Trains CNN-based LPN on 2D grid transformations

```bash
python run_experiments.py --experiment spatial
```

Or manually:
```bash
# Train
python train_spatial.py --num_epochs 30
```

**Expected results:**
- Training: ~30 minutes (CPU), ~8 minutes (GPU)
- Validation accuracy: 85-95% (grids are simpler)
- Perfect accuracy on simple transformations (transpose, rotate)

**Outputs:**
- `best_spatial_model.pt` - Trained model
- `spatial_training_history.json` - Training metrics

---

### Experiment 5: Run Everything

**What it does:** Runs all experiments in sequence

```bash
# Full run (~2 hours CPU, ~30 min GPU)
python run_experiments.py --experiment all

# Quick test run (~30 min CPU, ~10 min GPU)
python run_experiments.py --experiment all --quick
```

---

## 📊 Understanding Results

### Training Metrics

**Loss components:**
- `recon_loss`: How well model predicts outputs
- `kl_loss`: Regularization on latent space
- `total_loss`: recon + beta * kl

**Monitoring:**
- Loss should decrease smoothly
- Validation accuracy should increase
- KL should stabilize around 10-50

### Test Metrics

**Accuracy:**
- Exact match: All elements correct (strict)
- Typically 70-90% for list ops
- 85-95% for grid tasks

**Improvement with search:**
- Baseline: +5-10%
- PoE: +5-15%
- Better improvement = smoother latent space

### Agreement Score (PoE only)

**What it measures:** Consistency between examples
- 0.7-1.0: Examples agree (good)
- 0.3-0.7: Some disagreement
- 0.0-0.3: Examples conflict (bad)

---

## 🔧 Troubleshooting

### Issue: "Out of memory"
**Solution:**
```bash
# Reduce batch size
python train_lpn.py --batch_size 16  # or 8
```

### Issue: "Data not found"
**Solution:**
```bash
# Re-run setup
python setup.py

# Or generate manually
python generate_list_data.py
```

### Issue: "Model not found" in compare
**Solution:**
```bash
# Train both models first
python run_experiments.py --experiment baseline
python run_experiments.py --experiment poe
# Then compare
python run_experiments.py --experiment compare
```

### Issue: Training is slow
**Solution:**
```bash
# Use GPU if available
python train_lpn.py --device cuda

# Or reduce epochs for testing
python train_lpn.py --num_epochs 10
```

### Issue: Low accuracy (<50%)
**Possible causes:**
- Not enough epochs (try 50+)
- Learning rate too high/low
- Model too small (increase hidden_dim)

**Solutions:**
```bash
# More epochs
python train_lpn.py --num_epochs 100

# Bigger model
python train_lpn.py --hidden_dim 256

# Different learning rate
python train_lpn.py --lr 0.0005
```

---

## 📂 File Structure

```
.
├── setup.py                    # Automated setup
├── run_experiments.py          # Master experiment runner
│
├── generate_list_data.py       # Data generation
├── generate_grid_data.py
├── arc_data.py
│
├── lpn_model.py               # Baseline LPN
├── train_lpn.py
├── test_lpn.py
│
├── poe_model.py               # Product of Experts LPN
├── train_poe.py
├── test_poe.py
├── compare_methods.py
│
├── spatial_model.py           # Spatial (grid) LPN
├── train_spatial.py
│
├── analyze_results.py         # Visualization
│
└── docs/                      # Documentation
    ├── README.md
    ├── BAYESIAN_APPROACH.md
    ├── PRODUCT_OF_EXPERTS.md
    └── ROADMAP.md
```

---

## 🎓 Learning Path

### Day 1: Get Started (1-2 hours)
```bash
# Setup and run quick test
python setup.py
python run_experiments.py --experiment baseline --quick

# Examine results
ls analysis_outputs/
cat summary_report.txt
```

### Day 2: Compare Methods (2-3 hours)
```bash
# Run full experiments
python run_experiments.py --experiment baseline
python run_experiments.py --experiment poe
python run_experiments.py --experiment compare

# Study differences
cat comparison_results.json
```

### Day 3: Understand Theory (2-3 hours)
```bash
# Read documentation
cat BAYESIAN_APPROACH.md
cat PRODUCT_OF_EXPERTS.md

# Experiment with parameters
python train_poe.py --beta 0.05  # Different KL weight
python train_poe.py --latent_dim 128  # Larger latent space
```

### Week 2: Spatial Models (3-5 hours)
```bash
# Run spatial experiments
python run_experiments.py --experiment spatial

# Visualize what it learned
python -c "
from spatial_model import SpatialLPN
import torch
model = SpatialLPN()
model.load_state_dict(torch.load('best_spatial_model.pt'))
# ... inspect latent space
"
```

---

## 🚀 Next Steps

After completing the experiments:

1. **Read the theory:**
   - [BAYESIAN_APPROACH.md](BAYESIAN_APPROACH.md)
   - [PRODUCT_OF_EXPERTS.md](PRODUCT_OF_EXPERTS.md)

2. **Check the roadmap:**
   - [ROADMAP.md](ROADMAP.md) - 12-month development plan

3. **Extend the models:**
   - Add new transformations to `generate_list_data.py`
   - Implement equivariant layers
   - Try real ARC tasks

4. **Contribute:**
   - Implement Phase 1 improvements
   - Test on ARC-1 dataset
   - Develop object-centric representations

---

## 📞 Getting Help

**Common questions:**

Q: How long does training take?
A: 10-30 min on GPU, 1-3 hours on CPU (50 epochs)

Q: What accuracy should I expect?
A: 70-85% for list ops, 85-95% for grid tasks

Q: Which experiment should I run first?
A: Start with baseline, then PoE, then compare

Q: Can I train on CPU?
A: Yes! Just slower. Use `--quick` flag for testing

Q: Where are the results?
A: `analysis_outputs/` for plots, `*.json` for metrics

---

## ✅ Verification Checklist

Before running experiments, verify:

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`torch`, `numpy`, etc.)
- [ ] Data generated (`list_ops_data/` exists)
- [ ] 4GB+ RAM available
- [ ] GPU drivers (optional, for faster training)

After running experiments, you should have:

- [ ] `best_lpn_model.pt` (baseline)
- [ ] `best_poe_model.pt` (PoE)
- [ ] `analysis_outputs/` directory with plots
- [ ] `test_results.json` with metrics
- [ ] Validation accuracy > 70%

---

**Ready to start? Run:**
```bash
python setup.py && python run_experiments.py --experiment baseline --quick
```
