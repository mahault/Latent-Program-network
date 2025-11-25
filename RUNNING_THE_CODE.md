# Running the Bayesian LPN Implementation

This guide shows you how to run the **Product of Experts** and **Spatial LPN** implementations.

## 🎯 Two Tracks Available

### Track 1: Product of Experts on List Operations (Recommended First)
Test the Bayesian inference framework on existing list operations data.

### Track 2: Spatial LPN on Grid Tasks
Test spatial architecture on synthetic grid transformations (ARC-style).

---

## 🚀 Track 1: Product of Experts LPN

### Prerequisites
```bash
# You should already have the list operations data from before
# If not, generate it:
python generate_list_data.py
```

### Step 1: Train PoE-LPN
```bash
python train_poe.py --num_epochs 50 --batch_size 32 --lr 0.001 --beta 0.1
```

**Expected output:**
- Training takes ~20-30 minutes on GPU
- Best model saved to `best_poe_model.pt`
- History saved to `poe_training_history.json`

**Key metrics to watch:**
- Train Agreement: Should be > 0.5 (higher = examples agree more)
- Val Accuracy: Target 75-85%

### Step 2: Test PoE-LPN
```bash
python test_poe.py
```

**Expected output:**
- Results saved to `poe_test_results.json`
- Per-program breakdown showing which tasks benefit most from PoE

### Step 3: Compare PoE vs Baseline
```bash
python compare_methods.py
```

**Expected output:**
- Side-by-side comparison of 4 methods:
  1. Baseline without search
  2. Baseline with search
  3. PoE without search
  4. PoE with search

**Success criteria:**
- ✓ PoE improves over baseline by 5-10%
- ✓ Agreement score > 0.5 for consistent tasks
- ✓ PoE benefits more from search than baseline

---

## 🖼️ Track 2: Spatial LPN on Grid Tasks

### Step 1: Generate Synthetic Grid Data
```bash
python generate_grid_data.py
```

**What this does:**
- Creates 12 transformation types (rotate, mirror, transpose, etc.)
- Generates 50 tasks per transformation = 600 total tasks
- Splits into train/val/test (70/15/15)
- Saves to `./synthetic_grid_data/`

**Expected output:**
```
✓ Saved 420 tasks to synthetic_grid_data/train.json
✓ Saved 90 tasks to synthetic_grid_data/val.json
✓ Saved 90 tasks to synthetic_grid_data/test.json
```

### Step 2: Train Spatial LPN
```bash
python train_spatial.py --num_epochs 30 --batch_size 16 --lr 0.001
```

**Expected output:**
- Training takes ~15-20 minutes on GPU
- Best model saved to `best_spatial_model.pt`
- History saved to `spatial_training_history.json`

**Key metrics to watch:**
- Train Accuracy: Should reach > 90% (these are simple tasks)
- Val Accuracy: Target > 80%
- Val Pixel Accuracy: Should be > 95%

### Step 3: Test Spatial LPN
```bash
python test_spatial.py --visualize
```

**Expected output:**
- Results saved to `spatial_test_results.json`
- Per-transformation breakdown
- Sample predictions shown (if --visualize used)

**Success criteria:**
- ✓ Grid accuracy > 80%
- ✓ Pixel accuracy > 95%
- ✓ Test-time search improves results

---

## 📊 Understanding the Results

### Product of Experts Metrics

**Agreement Score (0-1):**
- > 0.7: Examples strongly agree on the program
- 0.5-0.7: Moderate agreement
- < 0.5: Examples are inconsistent (task may be ambiguous)

**Use case:** High agreement = PoE will work well

### Spatial Model Metrics

**Grid Accuracy:**
- Percentage of grids where EVERY pixel is correct
- Strict metric, shows if model truly learned transformation

**Pixel Accuracy:**
- Percentage of individual pixels correct
- More lenient, shows model is "mostly right"

---

## 🔬 Experiments to Try

### Experiment 1: Effect of Number of Examples
```python
# Edit train_poe.py or train_spatial.py
# Change the dataset to use 2, 3, 5, or 10 training examples
# Does PoE benefit more from additional examples?
```

### Experiment 2: Latent Dimension
```bash
# Train with different latent dimensions
python train_poe.py --latent_dim 32  # Smaller
python train_poe.py --latent_dim 128  # Larger

# Which performs better?
```

### Experiment 3: KL Weight (Beta)
```bash
# Less regularization
python train_poe.py --beta 0.01

# More regularization  
python train_poe.py --beta 0.5

# Find optimal balance
```

### Experiment 4: Test-Time Search Steps
```python
# Edit test_poe.py or test_spatial.py
# Change num_steps in predict_with_search()
# num_steps = 10, 30, 50, 100
# Does more search always help?
```

---

## 🐛 Troubleshooting

### "FileNotFoundError: list_ops_data/train.json"
**Solution:** Run `python generate_list_data.py` first

### "FileNotFoundError: synthetic_grid_data/train.json"
**Solution:** Run `python generate_grid_data.py` first

### "RuntimeError: CUDA out of memory"
**Solution:** Reduce batch size: `--batch_size 8`

### "Model not converging"
**Check:**
- Is loss decreasing?
- Is KL too large? (reduce beta)
- Is learning rate too high/low?

### "PoE not better than baseline"
**Possible causes:**
- Tasks may not benefit from multiple examples
- Agreement score is low (examples inconsistent)
- Beta too high (over-regularization)

**Try:**
- Check agreement scores in output
- Reduce beta: `--beta 0.05`
- Train longer: `--num_epochs 100`

---

## 📈 Expected Performance

### List Operations (PoE)

| Method | Accuracy | Notes |
|--------|----------|-------|
| Baseline (no search) | 70-75% | Original LPN |
| Baseline (with search) | 75-80% | +5-10% improvement |
| PoE (no search) | 75-80% | +5% over baseline |
| PoE (with search) | 80-85% | Best performance |

### Synthetic Grids (Spatial)

| Transformation | Accuracy | Difficulty |
|----------------|----------|------------|
| Identity | >95% | Easy |
| Rotate 90/180/270 | >90% | Easy |
| Mirror H/V | >90% | Easy |
| Transpose | >85% | Medium |
| Shift | >80% | Medium |
| Add border | >85% | Medium |

---

## 🚀 Next Steps After Success

Once you have good results on synthetic tasks:

### 1. Try on ARC Dataset
```bash
# Download ARC data
git clone https://github.com/fchollet/ARC-AGI.git

# Copy data
cp -r ARC-AGI/data/training/*.json ./arc_data/training/
cp -r ARC-AGI/data/evaluation/*.json ./arc_data/evaluation/

# Test loading
python arc_data.py

# Train (will need to adapt training script)
# python train_spatial_arc.py
```

### 2. Implement Object-Centric Approach
- Flood-fill segmentation
- What/where decomposition
- Track objects over time

### 3. Add Action Conditioning
- Extend decoder to handle actions
- Train on interactive tasks
- Active inference for action selection

### 4. Combine PoE + Spatial
- Use PoE combiner with spatial encoder
- Best of both approaches!

---

## 📝 Files Overview

**Core Models:**
- `poe_model.py` - Product of Experts LPN
- `spatial_model.py` - Spatial LPN for grids

**Training:**
- `train_poe.py` - Train PoE on list operations
- `train_spatial.py` - Train spatial on grid tasks

**Testing:**
- `test_poe.py` - Test PoE with detailed metrics
- `test_spatial.py` - Test spatial with visualization
- `compare_methods.py` - Compare PoE vs baseline

**Data:**
- `generate_grid_data.py` - Create synthetic grid tasks
- `arc_data.py` - Load ARC dataset

**Existing (reused):**
- `lpn_model.py` - Baseline LPN (from before)
- `generate_list_data.py` - List operations data

---

## 💡 Tips for Success

1. **Start with synthetic tasks** - They're easier and faster to train
2. **Monitor agreement scores** - High agreement = PoE will help
3. **Use test-time search** - Usually improves results by 5-10%
4. **Compare methods** - Run compare_methods.py to see what helps
5. **Visualize** - Use `--visualize` flag to see actual predictions
6. **Save checkpoints** - Best models are automatically saved

---

## 🆘 Getting Help

**If something doesn't work:**
1. Check error message carefully
2. Verify data files exist
3. Try reducing batch size
4. Check GPU memory with `nvidia-smi`
5. Review troubleshooting section above

**Common gotchas:**
- Forgetting to generate data first
- Wrong paths to data directories
- CUDA out of memory (reduce batch size)
- Model not converging (adjust learning rate/beta)

---

## ✅ Validation Checklist

Before moving to ARC:

- [ ] PoE trains successfully on list operations
- [ ] PoE achieves > 75% validation accuracy
- [ ] Agreement scores > 0.5 for most tasks
- [ ] PoE outperforms baseline by > 5%
- [ ] Spatial model trains on synthetic grids
- [ ] Spatial achieves > 80% grid accuracy
- [ ] Test-time search improves results
- [ ] Can visualize predictions successfully

---

**Ready to start? Begin with Track 1 (Product of Experts) or Track 2 (Spatial) above!**
