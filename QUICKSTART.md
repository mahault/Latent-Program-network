# LPN Quick Start Guide

## 🎯 What You're Building

A neural network that learns to solve programming tasks by:
1. Learning a continuous "program space"
2. Encoding example solutions into this space
3. Searching through it at test-time to solve new problems

## 📋 The Pipeline (5 Steps)

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Generate Data                                       │
│  python generate_list_data.py                               │
│                                                              │
│  Creates: 3,000 tasks across 30 program types               │
│  Output: list_ops_data/ folder                              │
│  Time: ~10 seconds                                           │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Train Model                                        │
│  python train_lpn.py --num_epochs 50                        │
│                                                              │
│  Trains encoder, decoder, and latent space                  │
│  Output: best_lpn_model.pt + training_history.json          │
│  Time: 10-30 min (GPU) or 2-3 hours (CPU)                  │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Test Model                                         │
│  python test_lpn.py                                         │
│                                                              │
│  Tests with AND without test-time search                    │
│  Output: test_results.json + comparison tables              │
│  Time: 5-10 minutes                                          │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Analyze Results                                    │
│  python analyze_results.py                                  │
│                                                              │
│  Creates plots and summary report                           │
│  Output: analysis_outputs/ folder with 5 files              │
│  Time: ~30 seconds                                           │
└─────────────────────────────────────────────────────────────┘
```

## 💡 Core Concept

### Without Test-Time Search:
```
Examples → [Encoder] → Latent (fixed) → [Decoder] → Output
```
Just uses the encoder's initial guess.

### With Test-Time Search:
```
Examples → [Encoder] → Latent (initial) 
                          ↓
                      [Optimize for 50 steps]
                          ↓
                       Latent (refined) → [Decoder] → Output
```
Refines the latent representation to better fit the examples.

## 📊 What Success Looks Like

**Training Phase:**
- Loss decreases smoothly
- Validation accuracy reaches 70-80%

**Testing Phase:**
- Base accuracy: 60-75%
- With search: 70-85% (+5-20% improvement)
- Structural operations benefit most

## 🎨 Example Task

**Program: "square"**

Training examples given to model:
```
[2, 3, 4]    → [4, 9, 16]
[-1, 5, 2]   → [1, 25, 4]
[0, 3, -2]   → [0, 9, 4]
```

Test (what model must predict):
```
[6, -3, 1]   → [36, 9, 1]  ✓
```

The model must:
1. Infer the "square" program from examples
2. Apply it to new input

## 🔧 Minimal Setup

```bash
# Install dependencies
pip install torch numpy matplotlib seaborn tqdm

# Run everything
python generate_list_data.py
python train_lpn.py --num_epochs 20  # Quick test (20 epochs)
python test_lpn.py
python analyze_results.py
```

**Total time:** ~30 minutes on GPU

## 🎓 Key Hyperparameters

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `latent_dim` | 64 | Size of program space (bigger = more capacity) |
| `hidden_dim` | 128 | Network width (bigger = more capacity) |
| `beta` | 0.1 | KL weight (higher = smoother latent space) |
| `num_epochs` | 50 | Training iterations |
| `lr` | 0.001 | Learning rate |

## 📈 Expected Timeline

- **Epoch 1-10**: Model starts learning basic patterns
- **Epoch 10-30**: Accuracy climbs to 60-70%
- **Epoch 30-50**: Fine-tuning, reaches 70-80%
- **Test-time search**: Adds 5-20% accuracy boost

## 🚨 Common Issues

**"ImportError: No module named torch"**
→ Run: `pip install torch`

**"CUDA out of memory"**
→ Run: `python train_lpn.py --batch_size 16`

**"Accuracy stuck at 30%"**
→ Train longer or increase model size:
  `python train_lpn.py --num_epochs 100 --hidden_dim 256`

## 🎯 Success Criteria

✓ Training converges (loss goes down)
✓ Validation accuracy > 65%
✓ Test-time search improves accuracy
✓ Different program types have different accuracies

## 🔬 Experiment Ideas

1. **Double latent space:** `--latent_dim 128`
2. **More programs:** Add operations in `generate_list_data.py`
3. **Longer search:** Change `num_steps=50` to `num_steps=100` in `test_lpn.py`
4. **Less regularization:** `--beta 0.05`

## 📚 Understanding the Output Files

| File | Contents |
|------|----------|
| `best_lpn_model.pt` | Trained neural network weights |
| `training_history.json` | Loss curves and metrics |
| `test_results.json` | Per-program accuracy data |
| `training_curves.png` | Visual training progress |
| `test_comparison.png` | With/without search bars |
| `accuracy_heatmap.png` | All programs, all metrics |
| `summary_report.txt` | Human-readable summary |

---

**Ready?** Start with: `python generate_list_data.py`
