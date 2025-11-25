# 📋 Quick Reference Card - Bayesian LPN

## 🚀 Getting Started (3 Commands)

```bash
# 1. Test setup
python run_all.py --test-only

# 2. Run everything
python run_all.py --track both

# 3. View results
python analyze_poe_results.py
```

---

## 🧠 Product of Experts Commands

```bash
# Train
python train_poe.py --num_epochs 50 --batch_size 32 --beta 0.1

# Test
python test_poe.py

# Compare
python compare_methods.py

# Analyze
python analyze_poe_results.py
```

**Outputs:** `best_poe_model.pt`, `poe_test_results.json`

---

## 🖼️ Spatial LPN Commands

```bash
# Generate data
python generate_grid_data.py

# Train
python train_spatial.py --num_epochs 30 --batch_size 16

# Test
python test_spatial.py --visualize
```

**Outputs:** `best_spatial_model.pt`, `spatial_test_results.json`

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

## 📁 Important Files

### Generated:
- `best_poe_model.pt` - Trained PoE model
- `best_spatial_model.pt` - Trained spatial model
- `*_test_results.json` - Test metrics
- `*_training_history.json` - Training curves

### Data:
- `list_ops_data/` - List operations (3K tasks)
- `synthetic_grid_data/` - Grid tasks (600 tasks)
- `arc_data/` - Real ARC dataset

---

## 🐛 Quick Fixes

```bash
# Missing data?
python generate_list_data.py
python generate_grid_data.py

# Out of memory?
--batch_size 8

# Not converging?
--num_epochs 100 --lr 0.0005
```

---

## 📚 Documentation Map

1. **RUNNING_THE_CODE.md** - How to run
2. **CODE_INDEX.md** - All files explained
3. **BAYESIAN_APPROACH.md** - Theory
4. **PRODUCT_OF_EXPERTS.md** - PoE guide
5. **ROADMAP.md** - Future plans

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
2. Read RUNNING_THE_CODE.md troubleshooting
3. Verify data files exist
4. Try reducing batch size

---

## 📞 File Locations

All files in: `/mnt/user-data/outputs/`

Python: 18 files
Docs: 11 files
Total: ~4,500 lines

---

**Start:** `python run_all.py --test-only`
