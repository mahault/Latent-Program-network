# 📖 Documentation Quick Reference

## 🎯 Where to Start

### I want to... → Read this:

**...run the current code**
→ [QUICKSTART.md](QUICKSTART.md) (5 min setup)

**...understand the vision**
→ [README.md](README.md) → "Vision" section

**...learn the theory**
→ [BAYESIAN_APPROACH.md](BAYESIAN_APPROACH.md)

**...implement Product of Experts**
→ [PRODUCT_OF_EXPERTS.md](PRODUCT_OF_EXPERTS.md)

**...see the development plan**
→ [ROADMAP.md](ROADMAP.md)

**...fix bugs**
→ BUGFIX.md, GRADIENT_FIX.md, UNICODE_FIX.md

## 📚 Documentation Map

```
START HERE
    ↓
README.md ────────────────┐
    ↓                     │
    ├→ Current code?      │
    │  → QUICKSTART.md    │
    │                     │
    ├→ Theory?            │
    │  → BAYESIAN_APPROACH.md
    │                     │
    ├→ Implementation?    │
    │  → PRODUCT_OF_EXPERTS.md
    │                     │
    └→ Future plans?      │
       → ROADMAP.md       │
                          │
                      Need help?
                          ↓
                  DOCUMENTATION_UPDATE.md
```

## 🎓 Learning Path

### Day 1: Understand Current System
1. Read QUICKSTART.md (10 min)
2. Run `python train_lpn.py` (30 min)
3. Examine results in analysis_outputs/

### Day 2: Learn Theory
1. Read BAYESIAN_APPROACH.md (45 min)
2. Focus on "Product of Experts" section
3. Understand probabilistic framework

### Day 3: Implementation
1. Read PRODUCT_OF_EXPERTS.md (30 min)
2. Study code examples
3. Start implementing SingleExampleEncoder

### Week 2-3: Build PoE
1. Follow ROADMAP.md Phase 1
2. Implement and test PoE components
3. Compare to baseline

## 🔍 Quick Answers

### Q: What's working now?
**A:** List operations with LSTM encoder/decoder. See QUICKSTART.md

### Q: What's the goal?
**A:** Bayesian LPN for ARC-AGI. See README.md vision section.

### Q: What's Product of Experts?
**A:** Bayesian method to combine evidence from multiple examples. See PRODUCT_OF_EXPERTS.md

### Q: What's the timeline?
**A:** 12 months, 9 phases. See ROADMAP.md

### Q: Where's the code?
**A:** Current code works as-is. New code follows ROADMAP.md phases.

### Q: How does this relate to ARC?
**A:** Phases 4-9 build toward ARC-1/2/3. See ROADMAP.md

## 📊 Document Purposes

| Document | Purpose | Length | Audience |
|----------|---------|--------|----------|
| **README.md** | Overview & vision | 10 min | Everyone |
| **QUICKSTART.md** | Get started fast | 5 min | New users |
| **BAYESIAN_APPROACH.md** | Theory & motivation | 45 min | Researchers |
| **PRODUCT_OF_EXPERTS.md** | Implementation guide | 30 min | Developers |
| **ROADMAP.md** | Development timeline | 20 min | Contributors |
| **DOCUMENTATION_UPDATE.md** | What changed | 10 min | Returning users |

## 🚀 Action Items by Role

### Researcher
1. Read BAYESIAN_APPROACH.md
2. Review related work in README.md
3. Validate mathematical framework

### Developer
1. Follow QUICKSTART.md to run code
2. Read PRODUCT_OF_EXPERTS.md
3. Start Phase 1 implementation

### Project Manager
1. Review ROADMAP.md timeline
2. Check success metrics
3. Coordinate with team

### Student/Learning
1. QUICKSTART.md → run experiments
2. BAYESIAN_APPROACH.md → understand theory
3. PRODUCT_OF_EXPERTS.md → implement from scratch

## 🔗 External Links

**Papers:**
- [LPN Paper](https://arxiv.org/abs/2411.08706) - Bonnet et al., 2024
- [ARC Dataset](https://github.com/fchollet/ARC-AGI) - Chollet, 2019

**Code:**
- [Re-ARC Generator](https://github.com/xu3kev/arc-dsl)
- [TRM Implementation](https://github.com/clement-bonnet/lpn)

## 💡 Key Concepts

**Latent Program θ:**
Continuous vector representing a program/transformation

**Product of Experts (PoE):**
Bayesian method: q(θ|all) ∝ ∏ᵢ q(θ|exampleᵢ)

**Amortized Inference:**
Single forward pass: θ = Encoder(examples)

**Test-Time Search:**
Optimize θ at test time for each new task

**Object-Centric:**
Represent scenes as sets of objects (what/where)

**Equivariance:**
Invariance to irrelevant transformations (color, rotation)

## 📈 Progress Tracker

**v0.1 (Current):**
- [x] List operations dataset
- [x] LSTM encoder/decoder
- [x] Amortized inference
- [x] Test-time gradient search

**v0.2 (Next):**
- [ ] Product of Experts
- [ ] Spatial CNN architecture
- [ ] Simple grid tasks

**v1.0 (Future):**
- [ ] Object-centric representations
- [ ] ARC-1/2/3 integration
- [ ] Compositional generalization

## 🆘 Troubleshooting

**"Where do I start?"**
→ QUICKSTART.md, then README.md

**"How do I implement PoE?"**
→ PRODUCT_OF_EXPERTS.md has full code

**"What's the research direction?"**
→ BAYESIAN_APPROACH.md + ROADMAP.md

**"Code isn't working"**
→ Check BUGFIX.md, GRADIENT_FIX.md, UNICODE_FIX.md

**"Want to contribute?"**
→ ROADMAP.md shows what needs building

---

**Still confused?** Read [DOCUMENTATION_UPDATE.md](DOCUMENTATION_UPDATE.md) for the big picture!
