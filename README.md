# Latent Program Network (LPN) Experiment

Implementation of **Latent Program Network** based on ["Searching Latent Program Spaces"](https://arxiv.org/abs/2411.08706) by Bonnet et al., 2024.

This experiment trains an LPN on list operation tasks and evaluates the impact of test-time search on generalization.

## 📁 Project Structure

```
.
├── generate_list_data.py      # Generate synthetic dataset
├── lpn_model.py               # LPN model architecture
├── train_lpn.py               # Training script
├── test_lpn.py                # Testing script with/without search
├── analyze_results.py         # Analysis and visualization
├── requirements.txt           # Dependencies
└── list_ops_data/            # Generated dataset (created by step 1)
    ├── train.json
    ├── val.json
    └── test.json
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- PyTorch (>=2.0.0)
- NumPy
- Matplotlib
- Seaborn
- tqdm

### Step 2: Generate Dataset

```bash
python generate_list_data.py
```

This creates 3,000 synthetic tasks (30 program types × 100 instances):
- **Train**: 2,070 tasks (70%)
- **Val**: 485 tasks (15%)
- **Test**: 445 tasks (15%)

**Program types include:**
- Mapping: square, negate, abs, add_N, multiply_N, etc.
- Filtering: filter_positive, filter_even, etc.
- Structural: reverse, sort, take_first_N, duplicate_each, etc.
- Reduction: sum, max, min, count, mean
- Combination: cumsum, differences, alternating_sign

### Step 3: Train the Model

```bash
python train_lpn.py --num_epochs 50 --batch_size 32 --lr 0.001
```

**Training parameters:**
- `--latent_dim`: Latent space dimension (default: 64)
- `--hidden_dim`: Hidden layer size (default: 128)
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--beta`: KL divergence weight (default: 0.1)
- `--device`: cuda or cpu (auto-detected)

**Outputs:**
- `best_lpn_model.pt` - Trained model weights
- `training_history.json` - Training metrics

**Training time:** ~10-30 minutes on GPU, ~2-3 hours on CPU

### Step 4: Test with Test-Time Search

```bash
python test_lpn.py
```

This evaluates the model twice:
1. **Without test-time search**: Direct encoder → decoder
2. **With test-time search**: Optimize latent for 50 steps

**Output:**
- `test_results.json` - Detailed per-program results
- Console output with comparison tables

### Step 5: Analyze Results

```bash
python analyze_results.py
```

**Generates visualizations:**
- `training_curves.png` - Loss and accuracy curves
- `test_comparison.png` - With/without search comparison
- `program_categories.png` - Performance by program type
- `accuracy_heatmap.png` - Per-program accuracy heatmap
- `summary_report.txt` - Text summary of findings

All outputs saved to `analysis_outputs/` directory.

## 📊 What to Expect

### Expected Results

Based on the paper, you should see:

1. **Training convergence**: Validation accuracy should reach 60-80% after 50 epochs
2. **Test-time search benefit**: Accuracy improvement of 5-20% with test-time search
3. **Program-specific patterns**:
   - Simple mapping operations (square, negate): High accuracy even without search
   - Structural operations (sort, reverse): Benefit more from search
   - Filtering operations: May struggle without enough training

### Key Metrics

- **Accuracy**: Exact match (all elements correct within ±0.5)
- **MSE**: Mean squared error on outputs
- **Improvement**: Change in accuracy with test-time search

## 🧪 Understanding the Model

### Architecture

```
┌─────────────────┐
│   Input-Output  │
│   Examples      │
└────────┬────────┘
         │
         ▼
    ┌─────────┐
    │ Encoder │  → [μ, σ]
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ Latent  │  (64-dim continuous space)
    │ Program │
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ Decoder │  + Input → Output
    └─────────┘

Test-time: Optimize latent via gradient ascent
```

### Key Components

1. **Encoder**: 
   - Processes example pairs
   - Outputs distribution over latent programs
   - Uses LSTM to aggregate across examples

2. **Latent Space**:
   - Continuous 64-dimensional space
   - Represents implicit programs
   - Regularized with KL divergence

3. **Decoder**:
   - Takes latent + input
   - Generates output autoregressively
   - Uses LSTM for sequence modeling

4. **Test-Time Optimization**:
   - Refines latent via gradient ascent
   - Maximizes likelihood on training examples
   - 50 optimization steps (takes ~0.1s per task)

## 🔬 Experiment Variations

Try these modifications:

### 1. Larger Latent Space
```bash
python train_lpn.py --latent_dim 128
```

### 2. More Training
```bash
python train_lpn.py --num_epochs 100
```

### 3. Different Search Steps
Edit `test_lpn.py` line with `num_steps=50` to try 10, 20, or 100 steps.

### 4. Different Program Types
Edit `generate_list_data.py` to add custom operations:
```python
def _my_operation(self, lst: List[int]) -> List[int]:
    return [x * 2 + 1 for x in lst]
```

## 📈 Interpreting Results

### Good Signs ✓
- Validation accuracy > 70%
- Test-time search improves accuracy by 5-20%
- Structural operations benefit most from search
- Training loss decreases smoothly

### Warning Signs ⚠
- Validation accuracy < 50% → Need more training or larger model
- No improvement with search → Latent space may not be smooth
- High KL divergence → Increase beta weight

## 🐛 Troubleshooting

**Out of memory?**
- Reduce `batch_size` to 16 or 8
- Reduce `hidden_dim` to 64

**Slow training?**
- Use GPU: `--device cuda`
- Reduce dataset size (edit `generate_list_data.py`)

**Poor accuracy?**
- Train longer: `--num_epochs 100`
- Increase model size: `--hidden_dim 256 --latent_dim 128`
- Adjust KL weight: `--beta 0.05`

## 📚 References

**Paper:**
```
Bonnet, C., & Macfarlane, M. V. (2024). 
Searching Latent Program Spaces. 
arXiv preprint arXiv:2411.08706.
```

**Key Insights:**
- Test-time search enables adaptation without parameter updates
- Continuous latent space bypasses discrete program search
- VAE regularization prevents overfitting to training distribution

## 🤝 Next Steps

1. **Try ARC-AGI tasks**: Replace list ops with visual grid transformations
2. **Add more program types**: Extend the generator with complex operations
3. **Visualize latent space**: Use t-SNE to see program clustering
4. **Compositional learning**: Test if model can combine learned programs

## 📝 License

MIT License - feel free to use and modify!

---

**Questions?** Check the paper or open an issue.
