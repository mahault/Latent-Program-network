# Latent Program Network (LPN) - Bayesian Active Inference Approach

Implementation of **Latent Program Network** inspired by ["Searching Latent Program Spaces"](https://arxiv.org/abs/2411.08706) (Bonnet et al., 2024), extended with **Bayesian inference** and **Active Inference** principles for ARC-AGI tasks.

## 🎯 Vision: Bayesian Program Induction

This project aims to learn **compositional programs** through:
- **Bayesian model selection** over program complexity
- **Product of Experts** for multi-example consistency
- **Object-centric representations** (what/where decomposition)
- **Equivariant architectures** via data augmentation
- **Test-time adaptation** through inference over latent programs

### Current vs. Target

| Aspect | Current (v0.1) | Target (v1.0) |
|--------|----------------|---------------|
| **Data** | 1D list operations | 2D grid transformations (ARC) |
| **Latent** | Single θ vector | Compositional object-centric θ |
| **Inference** | Amortized + gradient search | Product of Experts + recursive |
| **Architecture** | LSTM sequence model | Spatial CNN/ViT + equivariance |
| **Generalization** | In-distribution | Compositional OOD |

## 📁 Project Structure

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
│   ├── START_HERE.md            # Getting started guide
│   ├── BAYESIAN_APPROACH.md     # Theory (45 min)
│   ├── PRODUCT_OF_EXPERTS.md    # PoE implementation (30 min)
│   ├── ROADMAP.md               # 12-month development plan
│   ├── TROUBLESHOOTING.md       # Problem solving
│   └── QUICK_REFERENCE.md       # Command cheat sheet
│
├── utils/                   # Utility scripts
│   └── diagnose.py
│
├── setup.py                 # One-command setup
├── run_experiments.py       # Master experiment runner
├── run_all.py               # Run all experiments
└── README.md                # This file
```

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Setup everything in one command
python setup.py

# Run experiments
python run_experiments.py --experiment baseline --quick
```

### Option 2: Manual Training
```bash
# Install dependencies
pip install torch numpy matplotlib seaborn tqdm

# Generate data
python src/data_generation/generate_list_data.py

# Train model
python src/training/train_lpn.py --num_epochs 50

# Test with search
python src/testing/test_lpn.py

# Analyze results
python src/analysis/analyze_results.py
```

**Current capabilities:**
- ✅ 30 list operation programs (square, filter, reverse, etc.)
- ✅ Product of Experts Bayesian inference
- ✅ Spatial CNN model for 2D grids
- ✅ Test-time gradient search in latent space
- ✅ 2,670 synthetic tasks (list + grid)
- ✅ Complete training, testing, and visualization pipeline

**For detailed instructions, see** `docs/START_HERE.md`

## 🎓 Theoretical Foundation

### Probabilistic Model

```
Generative Model:
p(y | x, a, θ) = Decoder(x, a, θ)

Inference:
q(θ | x₁:ₙ, y₁:ₙ, a₁:ₙ) = Encoder(examples)

Learning:
max ELBO = 𝔼[log p(y|x,a,θ)] - KL[q(θ)||p(θ)]
```

Where:
- **x**: Input (list or grid)
- **a**: Action (optional, for interactive tasks)
- **y**: Output (transformed list or grid)  
- **θ**: Latent program representation

### Key Innovations

1. **Product of Experts** (Bayesian)
   ```
   q(θ | all examples) ∝ ∏ᵢ q(θ | example_i)
   ```
   Each example provides evidence; find θ consistent with ALL.

2. **Equivariance** (Symmetry)
   - Color permutations shouldn't change program
   - Rotations/shifts preserve task structure
   - Implemented via data augmentation

3. **Hierarchical Abstraction** (Composition)
   - Low-level: Individual objects
   - High-level: Object groups with shared dynamics
   - Model selection chooses simplest explanation

## 📊 Current Results (List Operations)

Using the baseline LSTM-based LPN:

**Training:**
- Dataset: 2,070 tasks (30 program types)
- Validation accuracy: ~75-85%
- Training time: 20-30 min (GPU)

**Testing:**
- Without search: ~70% accuracy
- With search (50 steps): ~75-80% accuracy
- Improvement: **+5-10% absolute**

See `analysis_outputs/` for detailed plots and metrics.

## 🔬 Next Steps: Toward ARC-AGI

### Phase 1: Spatial Architecture ⏳
- Replace LSTM with CNN/Vision Transformer
- Process 64×64 grids instead of 1D sequences
- Test on simple synthetic grid tasks

### Phase 2: Product of Experts ⏳
- Implement Bayesian inference over θ
- Each example → distribution over programs
- Combine via PoE for consistency

### Phase 3: Object-Centric ⏳
- Flood-fill segmentation (per Alexander's approach)
- What/where decomposition
- Sparse object interactions

### Phase 4: ARC Integration ⏳
- Load ARC-1/2/3 datasets
- Action conditioning for ARC-3
- Equivariance via augmentation

### Phase 5: Compositional Generalization ⏳
- DSL-based data generation (Re-ARC)
- Hierarchical object grouping
- Transfer learning across ARC versions

## 📚 Documentation

**Getting Started:**
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [README.md](README.md) - This file

**Theory:**
- [BAYESIAN_APPROACH.md](docs/BAYESIAN_APPROACH.md) - Bayesian inference framework
- [PRODUCT_OF_EXPERTS.md](docs/PRODUCT_OF_EXPERTS.md) - PoE implementation

**Development:**
- [ROADMAP.md](docs/ROADMAP.md) - Development phases
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Technical architecture

**Fixes & Notes:**
- [BUGFIX.md](BUGFIX.md) - Shape mismatch fix
- [GRADIENT_FIX.md](GRADIENT_FIX.md) - Test-time search fix
- [UNICODE_FIX.md](UNICODE_FIX.md) - Windows encoding fix

## 🤝 Related Work

**Latent Program Networks:**
- [Searching Latent Program Spaces](https://arxiv.org/abs/2411.08706) - Bonnet et al., 2024

**Bayesian Program Induction:**
- DreamCoder - Lake et al.
- Fast Structure Learning - Tenenbaum et al.

**ARC-AGI:**
- [ARC Dataset](https://github.com/fchollet/ARC-AGI) - Chollet, 2019
- TRM (Tiny Recursive Model) - He et al.
- Vision tricks for ARC - Kaiming He et al.

**Active Inference:**
- Free Energy Principle - Friston et al.
- Active Inference for Goal-Directed Behavior

## 🎯 Project Goals

1. **Short-term**: Establish baseline on ARC-1/2 with spatial LPN
2. **Medium-term**: Implement Product of Experts + object-centric representations
3. **Long-term**: Achieve compositional generalization on ARC-3

**Success Metrics:**
- ARC-1 validation accuracy > 50%
- ARC-2 generalization with test-time search
- ARC-3 interactive task solving

## 👥 Contributors

Based on discussions with Active Inference lab:
- Anson Lei (Framework design, implementation)
- Alexander Tschantz (Object-centric representations, ARC-3)
- Christopher Buckley (Bayesian framework, active inference)

## 📝 Citation

```bibtex
@article{bonnet2024searching,
  title={Searching Latent Program Spaces},
  author={Bonnet, Clément and Macfarlane, Matthew V},
  journal={arXiv preprint arXiv:2411.08706},
  year={2024}
}
```

## 📄 License

MIT License - See LICENSE file for details

---

**Current Status:** v0.1 - Baseline list operations working, spatial implementation in progress
