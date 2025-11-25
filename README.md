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
.
├── generate_list_data.py      # [v0.1] Generate synthetic list ops
├── lpn_model.py               # [v0.1] Current LSTM-based LPN
├── train_lpn.py               # [v0.1] Training script
├── test_lpn.py                # [v0.1] Testing with/without search
├── analyze_results.py         # [v0.1] Visualization
│
├── spatial_lpn/               # [v1.0] NEW: Spatial architecture
│   ├── spatial_model.py       # CNN/ViT for grids
│   ├── product_of_experts.py  # Bayesian inference
│   └── arc_data.py            # ARC dataset loader
│
├── docs/
│   ├── BAYESIAN_APPROACH.md   # Theory and motivation
│   ├── PRODUCT_OF_EXPERTS.md  # PoE implementation guide
│   └── ROADMAP.md             # Development phases
│
└── list_ops_data/             # Generated training data
```

## 🚀 Quick Start (Current v0.1)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train on List Operations (Baseline)
```bash
# 1. Generate data
python generate_list_data.py

# 2. Train model
python train_lpn.py --num_epochs 50

# 3. Test with search
python test_lpn.py

# 4. Analyze results
python analyze_results.py
```

**Current capabilities:**
- ✅ 30 list operation programs (square, filter, reverse, etc.)
- ✅ Amortized inference via LSTM encoder
- ✅ Test-time gradient search in latent space
- ✅ 3,000 synthetic tasks for training

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
