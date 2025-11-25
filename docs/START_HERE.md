# 🚀 START HERE

## Welcome to Bayesian Latent Program Networks!

This is a complete, runnable implementation of Bayesian LPN for ARC-AGI tasks with data generation, training, testing, and visualization.

---

## ⚡ Quick Start (3 Commands)

```bash
# 1. Setup everything (auto-installs dependencies & generates data)
python setup.py

# 2. Run a quick test (10 epochs, ~5 minutes on GPU)
python run_experiments.py --experiment baseline --quick

# 3. View results
cat results/metrics/test_results.json
ls analysis_outputs/
```

**That's it!** You now have a trained LPN model with test results and visualizations.

---

## 📋 What You Have

### ✅ Organized Project Structure

```
Latent-Program-Network/
├── src/
│   ├── models/              # Model architectures
│   │   ├── lpn_model.py         # Baseline LSTM-based LPN
│   │   ├── poe_model.py         # Product of Experts LPN
│   │   └── spatial_model.py     # CNN-based spatial LPN
│   ├── training/            # Training scripts
│   │   ├── train_lpn.py
│   │   ├── train_poe.py
│   │   └── train_spatial.py
│   ├── testing/             # Testing & evaluation
│   │   ├── test_lpn.py
│   │   ├── test_poe.py
│   │   └── test_spatial.py
│   ├── data_generation/     # Dataset creation
│   │   ├── generate_list_data.py
│   │   ├── generate_grid_data.py
│   │   └── arc_data.py
│   └── analysis/            # Visualization tools
│       ├── analyze_results.py
│       ├── analyze_poe_results.py
│       └── compare_methods.py
│
├── data/                    # All datasets
│   ├── list_ops_data/           # 2,070 list operation tasks
│   └── synthetic_grid_data/     # 600 grid transformation tasks
│
├── results/                 # Training outputs
│   ├── models/                  # .pt model checkpoints
│   └── metrics/                 # .json result files
│
├── docs/                    # Documentation
│   ├── START_HERE.md            # This file!
│   ├── BAYESIAN_APPROACH.md     # Theory (45 min)
│   ├── PRODUCT_OF_EXPERTS.md    # PoE guide (30 min)
│   ├── ROADMAP.md               # Development plan
│   ├── TROUBLESHOOTING.md       # Problem solving
│   └── QUICK_REFERENCE.md       # Command cheat sheet
│
├── setup.py                 # One-command setup
├── run_experiments.py       # Master experiment runner
└── README.md                # Project overview
```

---

## 🎯 Running Experiments

### Experiment 1: Baseline LPN (List Operations)

**What it does:** Trains LSTM-based LPN on list transformations

```bash
# Quick test (10 epochs, ~5 minutes on GPU)
python run_experiments.py --experiment baseline --quick

# Full training (50 epochs, ~20 minutes on GPU)
python run_experiments.py --experiment baseline
```

**Expected Performance:**
- Validation accuracy: 70-85%
- Test accuracy (no search): 70-80%
- Test accuracy (with search): 75-85%
- Improvement from search: +5-10%

**Outputs:**
- `results/models/best_lpn_model.pt` - Trained model
- `results/metrics/training_history.json` - Training curves
- `results/metrics/test_results.json` - Test metrics
- `analysis_outputs/` - Visualizations

---

### Experiment 2: Product of Experts (Bayesian)

**What it does:** Trains PoE-LPN with Bayesian inference over programs

```bash
python run_experiments.py --experiment poe
```

**Why it's better:**
- Combines evidence from multiple examples
- Uses Product of Experts for consistency
- Computes agreement scores
- Better generalization on ambiguous tasks

**Expected Performance:**
- Validation accuracy: 75-88%
- Test accuracy (with search): 80-90%
- **Improvement over baseline: +5-10%**
- Agreement score: >0.5

**Outputs:**
- `results/models/best_poe_model.pt`
- `results/metrics/poe_test_results.json` (includes agreement scores)

---

###Experiment 3: Spatial LPN (Grid Tasks)

**What it does:** Trains CNN-based LPN on 2D grid transformations

```bash
python run_experiments.py --experiment spatial
```

**Why it's different:**
- Uses CNNs instead of LSTMs
- Handles 2D grids (ARC-style tasks)
- 12 transformation types
- Ready for ARC dataset

**Expected Performance:**
- Validation accuracy: 85-95%
- Perfect on simple transforms (rotate, mirror)

---

### Experiment 4: Compare All Methods

```bash
# Requires baseline and PoE models trained first
python run_experiments.py --experiment compare
```

Compares: Baseline (no search) | Baseline + search | PoE (no search) | PoE + search

**Outputs:**
- `results/metrics/comparison_results.json`
- Performance breakdown by program type

---

### Experiment 5: Run Everything

```bash
# Full pipeline (~2 hours on GPU, ~6 hours on CPU)
python run_experiments.py --experiment all

# Quick pipeline (~30 min on GPU)
python run_experiments.py --experiment all --quick
```

Runs all experiments in sequence.

---

## 📊 What You'll Get

After running experiments, you'll have:

**Trained Models:**
- `best_lpn_model.pt` - Baseline LPN
- `best_poe_model.pt` - Product of Experts LPN
- `best_spatial_model.pt` - Spatial LPN

**Results & Metrics:**
- `test_results.json` - Detailed test metrics
- `training_history.json` - Training curves
- `comparison_results.json` - Method comparison

**Visualizations:**
- `analysis_outputs/training_curves.png`
- `analysis_outputs/test_comparison.png`
- `analysis_outputs/accuracy_heatmap.png`
- `analysis_outputs/summary_report.txt`

**Expected Performance:**
- Baseline: 70-80% accuracy
- With search: +5-10% improvement
- PoE: +5-10% over baseline
- Spatial: 85-95% accuracy

---

## 📖 Documentation Guide

**I want to...**

**...run code immediately**
→ This file! (See "Quick Start" above)

**...understand the theory**
→ `docs/BAYESIAN_APPROACH.md` → `docs/PRODUCT_OF_EXPERTS.md`

**...see the development plan**
→ `docs/ROADMAP.md` (12-month timeline)

**...find command reference**
→ `docs/QUICK_REFERENCE.md` (command cheat sheet)

**...troubleshoot issues**
→ `docs/TROUBLESHOOTING.md` (detailed problem solving)

**...understand the project vision**
→ `README.md` (main overview)

---

## 🎓 Learning Path

### Beginner Track
1. Run `python setup.py`
2. Run `python run_experiments.py --experiment baseline --quick`
3. Look at `analysis_outputs/summary_report.txt`
4. Read `README.md` overview

### Intermediate Track
1. Run all experiments: `python run_experiments.py --experiment all`
2. Compare results in `comparison_results.json`
3. Read `BAYESIAN_APPROACH.md`
4. Read `PRODUCT_OF_EXPERTS.md`

### Advanced Track
1. Read all documentation
2. Modify `generate_list_data.py` to add custom operations
3. Implement equivariant layers (see `ROADMAP.md` Phase 3)
4. Train on real ARC tasks

---

## 🔧 Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- CPU (works but slower)

**Recommended:**
- Python 3.9+
- 8GB+ RAM
- NVIDIA GPU with CUDA (10x faster)

**Dependencies:**
```bash
pip install torch numpy matplotlib seaborn tqdm
```
Or just run `python setup.py` - it installs everything!

---

## 🚨 Troubleshooting

### "No module named 'torch'"
```bash
python setup.py  # Auto-installs all dependencies
# Or manually:
pip install torch numpy matplotlib seaborn tqdm
```

### "Data not found" or "FileNotFoundError"
```bash
python setup.py  # Regenerates all datasets
# Or manually:
python src/data_generation/generate_list_data.py
python src/data_generation/generate_grid_data.py
```

### "Out of memory" / CUDA OOM
```bash
# Reduce batch size
python run_experiments.py --experiment baseline --batch_size 16
# Or even smaller:
python run_experiments.py --experiment baseline --batch_size 8
```

### "Model not found" in compare experiment
```bash
# Train both models first
python run_experiments.py --experiment baseline
python run_experiments.py --experiment poe
# Then compare
python run_experiments.py --experiment compare
```

### Training too slow
```bash
# Use GPU if available
python src/training/train_lpn.py --device cuda

# Or reduce epochs for quick testing
python run_experiments.py --experiment baseline --quick
```

### Low accuracy (<50%)
**Possible causes:**
- Not enough epochs (try 50+)
- Learning rate too high/low
- Model too small

**Solutions:**
```bash
# More epochs
python src/training/train_lpn.py --num_epochs 100

# Bigger model
python src/training/train_lpn.py --hidden_dim 256

# Different learning rate
python src/training/train_lpn.py --lr 0.0005
```

For more detailed troubleshooting, see `docs/TROUBLESHOOTING.md`.

---

## 🎯 Success Checklist

After running `python setup.py`, you should have:
- [ ] `data/list_ops_data/` directory (2,070+ tasks)
- [ ] `data/synthetic_grid_data/` directory (600+ tasks)
- [ ] No error messages
- [ ] All dependencies installed

After running experiments, you should have:
- [ ] `results/models/*.pt` model files
- [ ] `results/metrics/*.json` result files
- [ ] `analysis_outputs/` directory with plots
- [ ] Validation accuracy > 70%

---

## 💡 Key Concepts

**Latent Program Network (LPN):**
- Learns programs as continuous vectors
- Can search in latent space at test time
- Generalizes to new tasks

**Product of Experts (PoE):**
- Bayesian inference over programs
- Each example provides evidence
- Combines evidence for consistency
- +5-10% accuracy improvement

**Test-Time Search:**
- Optimizes latent vector for each test task
- Adapts without changing model weights
- +5-15% accuracy improvement

**Spatial LPN:**
- Handles 2D grids (ARC tasks)
- Uses CNNs instead of LSTMs
- Ready for visual reasoning tasks

---

## 🌟 Project Highlights

**What makes this special:**

1. **Complete & Runnable**
   - No missing pieces
   - One-command setup
   - Actual data generation

2. **Bayesian Approach**
   - Product of Experts
   - Uncertainty quantification
   - Principled inference

3. **Multiple Modalities**
   - 1D sequences (lists)
   - 2D grids (images)
   - ARC-ready

4. **Comprehensive Docs**
   - Theory explained
   - Implementation detailed
   - 12-month roadmap

5. **Based on Real Research**
   - Lab meeting insights
   - Active Inference framework
   - Path to ARC-AGI

---

## 📞 Next Steps

**After running experiments:**

1. **Examine Results**
   ```bash
   cat analysis_outputs/summary_report.txt
   cat comparison_results.json
   ```

2. **Learn Theory**
   - Read `BAYESIAN_APPROACH.md`
   - Study `PRODUCT_OF_EXPERTS.md`
   - Review `ROADMAP.md`

3. **Extend Implementation**
   - Add new transformations
   - Implement equivariance
   - Try ARC tasks

4. **Contribute**
   - Follow `ROADMAP.md` phases
   - Implement improvements
   - Share findings

---

## ✅ Final Checklist

Before starting:
- [ ] Read this file
- [ ] Have Python 3.8+ installed
- [ ] Have 4GB+ RAM available
- [ ] Ready to wait 5-30 minutes

To begin:
```bash
python setup.py
```

To test:
```bash
python run_experiments.py --experiment baseline --quick
```

To go deep:
```bash
python run_experiments.py --experiment all
cat BAYESIAN_APPROACH.md
```

---

## 🎉 You're Ready!

**Everything is set up and ready to run.**

**Start now:**
```bash
python setup.py
python run_experiments.py --experiment baseline --quick
```

**Questions?**
- See `docs/TROUBLESHOOTING.md` for detailed help
- See `docs/QUICK_REFERENCE.md` for command reference
- See `README.md` for project overview
- See `docs/BAYESIAN_APPROACH.md` for theory

**Good luck! 🚀**
