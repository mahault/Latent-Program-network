# Development Roadmap: Bayesian LPN for ARC-AGI

## 🎯 Project Vision

Build a **Bayesian Latent Program Network** that can:
1. Learn compositional programs from few examples
2. Generalize to out-of-distribution tasks
3. Perform test-time adaptation via inference
4. Scale to ARC-AGI interactive tasks

## 📊 Current Status (v0.1)

**✅ Completed:**
- LSTM-based encoder/decoder
- Amortized inference
- Test-time gradient search
- List operations dataset (3,000 tasks)
- Training pipeline working
- Analysis and visualization

**Current Performance:**
- List operations: ~75-80% accuracy
- Test-time search improvement: +5-10%
- Training time: 20-30 min (GPU)

## 🗺️ Development Phases

### Phase 1: Product of Experts (2-3 weeks) ⏳

**Goal:** Implement Bayesian inference for current list operations

**Tasks:**
- [ ] Implement `SingleExampleEncoder`
- [ ] Implement `ProductOfExpertsCombiner`
- [ ] Refactor training loop for PoE
- [ ] Compare PoE vs. baseline amortized
- [ ] Visualize latent distributions

**Expected Results:**
- Accuracy: 80-85% (+5-10% over baseline)
- Better uncertainty quantification
- Improved few-shot learning

**Files to Create:**
```
poe_model.py          # PoE-LPN implementation
train_poe.py          # Training script for PoE
compare_methods.py    # Benchmark PoE vs baseline
```

**Validation:**
- [ ] PoE outperforms averaging by 5%+
- [ ] Agreement score > 0.5 for consistent tasks
- [ ] Uncertainty decreases with more examples

---

### Phase 2: Spatial Architecture (3-4 weeks) 🔄

**Goal:** Handle 2D grid inputs for ARC tasks

**Tasks:**
- [ ] Implement CNN/ViT encoder for grids
- [ ] Create spatial decoder (transposed conv)
- [ ] Generate simple synthetic grid tasks
- [ ] Adapt PoE for spatial inputs
- [ ] Test on synthetic blob movement

**Expected Results:**
- Successfully processes 64×64 grids
- Learns simple spatial transformations
- Maintains PoE benefits in spatial domain

**Files to Create:**
```
spatial_model.py      # CNN/ViT architecture
grid_data.py          # Grid dataset generator
spatial_poe.py        # PoE for spatial data
```

**New Data:**
```python
# Synthetic grid tasks
- move_object_left
- move_object_right
- rotate_90
- mirror_horizontal
- color_swap
```

**Validation:**
- [ ] 90%+ accuracy on synthetic tasks
- [ ] Equivariance to rotations/reflections
- [ ] Faster than processing as sequences

---

### Phase 3: Equivariance & Augmentation (2 weeks) 🔄

**Goal:** Build invariance to irrelevant transformations

**Tasks:**
- [ ] Implement data augmentation pipeline
  - Color permutations
  - Rotations (90°, 180°, 270°)
  - Reflections (horizontal/vertical)
  - Random shifts
- [ ] Add equivariant conv layers
- [ ] Test on augmented data
- [ ] Measure robustness

**Expected Results:**
- Invariant to color permutations
- Robust to spatial transformations
- Better OOD generalization

**Files to Create:**
```
augmentation.py       # Data augmentation
equivariant_layers.py # Equivariant architectures
```

**Validation:**
- [ ] Same accuracy on rotated inputs
- [ ] Same accuracy with permuted colors
- [ ] Improves ARC generalization

---

### Phase 4: ARC-1 Integration (3-4 weeks) 🎯

**Goal:** Train and evaluate on real ARC-1 tasks

**Tasks:**
- [ ] Download and parse ARC-1 data
- [ ] Implement ARC data loader
- [ ] Adapt model for variable-size grids
- [ ] Train on ARC-1 training set
- [ ] Evaluate on ARC-1 eval set
- [ ] Benchmark against baselines

**Expected Results:**
- ARC-1 validation: 20-30% accuracy
- Better than random baseline
- Identifies failure modes

**Files to Create:**
```
arc_loader.py         # ARC dataset loader
arc_train.py          # Training on ARC
arc_eval.py           # Evaluation script
```

**ARC-1 Data:**
- Training: 400 tasks
- Evaluation: 400 tasks
- Source: https://github.com/fchollet/ARC-AGI

**Validation:**
- [ ] Above random baseline (>5%)
- [ ] Learns some task patterns
- [ ] PoE helps with few-shot

---

### Phase 5: Object-Centric Representations (4-5 weeks) 🔄

**Goal:** Decompose scenes into objects with what/where

**Tasks:**
- [ ] Implement flood-fill segmentation
- [ ] Extract object "what" (appearance hash)
- [ ] Extract object "where" (bounding box)
- [ ] Track objects over time
- [ ] PoE over object-centric latents
- [ ] Hierarchical object grouping

**Expected Results:**
- Meaningful object segmentation
- Sparse object interactions
- Better interpretability

**Files to Create:**
```
object_extraction.py  # Flood-fill segmentation
object_model.py       # Object-centric LPN
object_dynamics.py    # Object transition models
```

**Based on Alexander's work:**
- 16 discrete colors → flood-fill
- What: discrete hash of pixel pattern
- Where: bounding box (x0, y0, width, height)
- Track objects across frames

**Validation:**
- [ ] Correctly segments >80% of objects
- [ ] Tracks objects across frames
- [ ] Reduces to <10 objects per scene

---

### Phase 6: Action Conditioning (2-3 weeks) 🎮

**Goal:** Handle interactive ARC-3 tasks

**Tasks:**
- [ ] Add action input to decoder
- [ ] Implement action embeddings
- [ ] Collect ARC-3 interaction data
- [ ] Train on action-conditioned tasks
- [ ] Active inference for action selection

**Expected Results:**
- Model responds to actions
- Learns action effects
- Basic interactive task solving

**Files to Create:**
```
action_model.py       # Action-conditioned decoder
arc3_loader.py        # ARC-3 data with actions
active_inference.py   # Action selection via EFE
```

**Action Space:**
- Directional: up, down, left, right
- Interactive: click(x, y), fire
- Reduced: click(object_id)

**Validation:**
- [ ] Correctly predicts action effects
- [ ] Solves simple interactive tasks
- [ ] Active inference selects good actions

---

### Phase 7: Compositional Generalization (4-6 weeks) 🚀

**Goal:** Combine learned sub-programs for novel tasks

**Tasks:**
- [ ] Hierarchical latent structure
- [ ] Sub-program library learning
- [ ] Compositional decoder
- [ ] Test on compositional tasks
- [ ] Model selection via marginal likelihood

**Expected Results:**
- Composes learned primitives
- Zero-shot on compositions
- Better ARC-2 generalization

**Files to Create:**
```
hierarchical_model.py # Hierarchical θ
composition.py        # Program composition
library_learning.py   # Learn reusable modules
```

**Example Compositions:**
```
filter_red + move_left = "move red objects left"
sort + reverse = "sort descending"
count + replicate = "duplicate N times"
```

**Validation:**
- [ ] Solves unseen compositions
- [ ] Library grows with experience
- [ ] Model selection picks simplest

---

### Phase 8: DSL Integration (3-4 weeks) 📚

**Goal:** Generate unlimited training data

**Tasks:**
- [ ] Integrate Re-ARC generator
- [ ] DSL for ARC-like tasks
- [ ] Procedural task generation
- [ ] Pre-training on synthetic data
- [ ] Transfer to real ARC

**Expected Results:**
- Unlimited training data
- Better sample efficiency
- Improved ARC-1/2 performance

**Files to Create:**
```
dsl_generator.py      # DSL-based data generation
synthetic_tasks.py    # Task templates
pretrain.py           # Pre-training script
```

**Data Sources:**
- Re-ARC: https://github.com/xu3kev/arc-dsl
- BARC: Benchmark ARC
- Custom DSL primitives

**Validation:**
- [ ] Generates diverse tasks
- [ ] Pre-training improves ARC accuracy
- [ ] Transfer learning works

---

### Phase 9: Full ARC-3 (5-6 weeks) 🎯

**Goal:** Solve interactive ARC-3 games

**Tasks:**
- [ ] Full object-centric pipeline
- [ ] Recursive inference (TRM-style)
- [ ] Active learning for exploration
- [ ] Long-horizon planning
- [ ] Benchmark on 6 released games

**Expected Results:**
- Solves 2-3 of 6 games
- Better than random baseline
- Identifies improvement areas

**Files to Create:**
```
arc3_model.py         # Full ARC-3 system
recursive_inference.py # TRM-style refinement
planning.py           # Multi-step planning
```

**ARC-3 Games (Released):**
1. Waterfall game
2. Object movement game
3. Color change game
4. (3 more to identify)

**Validation:**
- [ ] Above random on all 6 games
- [ ] Solves at least 1 game completely
- [ ] Learning from interaction

---

## 📈 Success Metrics

### Short-term (3 months)
- [ ] PoE improves accuracy by 10%+
- [ ] Spatial architecture working
- [ ] ARC-1 validation: 20-30%

### Medium-term (6 months)
- [ ] Object-centric representations
- [ ] ARC-2 with generalization
- [ ] Action conditioning working

### Long-term (12 months)
- [ ] ARC-1 validation: 50%+
- [ ] ARC-2 with test-time search
- [ ] ARC-3: Solve 2+ games

## 🔧 Infrastructure Needs

### Compute
- **Training:** 1 GPU (RTX 3090 or better)
- **Experiments:** CPU sufficient for quick tests
- **ARC-3:** May need multi-GPU for scale

### Data Storage
- ARC datasets: ~100 MB
- Synthetic data: ~1-10 GB
- Model checkpoints: ~500 MB each

### Code Organization
```
src/
├── models/
│   ├── base_lpn.py
│   ├── poe_lpn.py
│   ├── spatial_lpn.py
│   └── object_lpn.py
├── data/
│   ├── list_ops.py
│   ├── arc_loader.py
│   └── augmentation.py
├── inference/
│   ├── amortized.py
│   ├── poe.py
│   └── recursive.py
└── utils/
    ├── visualization.py
    └── metrics.py
```

## 🤝 Collaboration Points

### With Alexander (Object-centric)
- Share object extraction code
- Test on ARC-3 games
- Coordinate on data formats

### With Chris (Bayesian theory)
- Validate PoE implementation
- Model selection framework
- Active inference formulation

### With Anson (Framework)
- Merge code bases
- Standardize APIs
- Share infrastructure

## 📊 Milestones & Timeline

**Month 1-2:** Phases 1-2 (PoE + Spatial)
**Month 3-4:** Phases 3-4 (Equivariance + ARC-1)
**Month 5-6:** Phases 5-6 (Objects + Actions)
**Month 7-9:** Phases 7-8 (Composition + DSL)
**Month 10-12:** Phase 9 (Full ARC-3)

## 🎓 Learning Resources

**To Read:**
1. Bonnet et al. - Searching Latent Program Spaces
2. He et al. - TRM for ARC
3. Ellis et al. - DreamCoder
4. Hinton - Training Products of Experts
5. Friston et al. - Active Inference

**To Implement:**
1. PoE from scratch
2. Equivariant convolutions
3. Flood-fill segmentation
4. Recursive refinement
5. Expected Free Energy

## 🚀 Quick Wins

Priority items for immediate impact:

1. **PoE Implementation** (1 week)
   - Biggest theoretical improvement
   - Clean baseline comparison
   
2. **Simple Grid Tasks** (1 week)
   - Validate spatial architecture
   - Debug CNN issues early
   
3. **Data Augmentation** (3 days)
   - Easy to implement
   - Large impact on robustness

---

**Current Phase:** Phase 1 (Product of Experts)  
**Next Milestone:** PoE working on list operations  
**Target Date:** End of Week 3
