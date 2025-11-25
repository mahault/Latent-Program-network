# 📋 Quick Reference - Bayesian LPN

## 🚀 Getting Started (3 Commands)

```bash
# 1. Setup everything
python setup.py

# 2. Run experiments
python run_experiments.py --experiment baseline --quick

# 3. View results
cat results/metrics/test_results.json
```

---

## 📖 Documentation Map

**Start Here:**
- `docs/START_HERE.md` - Complete getting started guide

**Theory:**
- `docs/BAYESIAN_APPROACH.md` - Bayesian framework (45 min)
- `docs/PRODUCT_OF_EXPERTS.md` - PoE implementation (30 min)

**Planning:**
- `docs/ROADMAP.md` - 12-month development plan
- `README.md` - Project overview & vision

**Help:**
- `docs/TROUBLESHOOTING.md` - Problem solving
- `docs/QUICK_REFERENCE.md` - This file!

---

## 🧠 Product of Experts Commands

```bash
# Train
python src/training/train_poe.py --num_epochs 50 --batch_size 32 --beta 0.1

# Test
python src/testing/test_poe.py

# Compare
python src/analysis/compare_methods.py

# Analyze
python src/analysis/analyze_poe_results.py
```

**Outputs:** `results/models/best_poe_model.pt`, `results/metrics/poe_test_results.json`

---

## 🖼️ Spatial LPN Commands

```bash
# Generate data
python src/data_generation/generate_grid_data.py

# Train
python src/training/train_spatial.py --num_epochs 30 --batch_size 16

# Test
python src/testing/test_spatial.py --visualize
```

**Outputs:** `results/models/best_spatial_model.pt`, `results/metrics/spatial_test_results.json`

---

## 🎛️ Common Parameters

```bash
--num_epochs 50      # Training epochs
--batch_size 32      # Batch size
--lr 0.001           # Learning rate
--beta 0.1           # KL weight
--latent_dim 64      # Latent dimension
--hidden_dim 128     # Hidden layer size
--device cuda        # cuda or cpu
```

---

## 📊 Key Metrics

### PoE Metrics:
- **Accuracy:** % of exact matches (target: >75%)
- **Agreement:** Consistency between examples (target: >0.5)
- **MSE:** Mean squared error (lower = better)

### Spatial Metrics:
- **Grid Accuracy:** % of perfect grids (target: >80%)
- **Pixel Accuracy:** % of correct pixels (target: >95%)

---

## 📁 Important Files & Directories

### Models:
- `results/models/best_lpn_model.pt` - Baseline model
- `results/models/best_poe_model.pt` - PoE model
- `results/models/best_spatial_model.pt` - Spatial model

### Results:
- `results/metrics/test_results.json` - Test metrics
- `results/metrics/poe_test_results.json` - PoE test metrics
- `results/metrics/*_training_history.json` - Training curves
- `results/metrics/comparison_results.json` - Method comparison

### Data:
- `data/list_ops_data/` - List operations (2,070 tasks)
- `data/synthetic_grid_data/` - Grid tasks (600 tasks)

---

## 🐛 Quick Fixes

```bash
# Missing data?
python setup.py
# Or manually:
python src/data_generation/generate_list_data.py
python src/data_generation/generate_grid_data.py

# Out of memory?
python run_experiments.py --experiment baseline --batch_size 8

# Not converging?
python src/training/train_lpn.py --num_epochs 100 --lr 0.0005
```

---

## ✅ Success Checklist

- [ ] Setup test passes
- [ ] PoE trains (val acc >75%)
- [ ] Agreement >0.5
- [ ] PoE beats baseline by 5%+
- [ ] Spatial trains (grid acc >80%)
- [ ] Visualizations work

---

## 🔬 Quick Experiments

```bash
# Try different latent sizes
python train_poe.py --latent_dim 32
python train_poe.py --latent_dim 128

# Try different KL weights
python train_poe.py --beta 0.01
python train_poe.py --beta 0.5

# More training
python train_poe.py --num_epochs 100
```

---

## 🆘 Help

1. Check error message
2. Read `docs/TROUBLESHOOTING.md`
3. Verify data files exist in `data/`
4. Try reducing batch size
5. See `docs/START_HERE.md` for detailed guide

---

**Quick Start:** `python setup.py && python run_experiments.py --experiment baseline --quick`
