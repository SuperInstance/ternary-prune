# ternary-prune

*What do you cut when every weight is already just a trit?*

---

In float networks, pruning means zeroing small weights. In ternary networks, every nonzero weight is ±1 — they're all the same magnitude. So what do you prune?

**Uncertainty.** You prune the weights that are *least sure* about their sign. The ones that flip between -1 and +1 most often during training. The ones whose gradient signal is weakest. The rows and columns that contribute least to the output.

This crate implements five pruning strategies:
- **Magnitude pruning** (flip-count based) — weights with highest flip counts are most uncertain
- **Gradient pruning** — weights with near-zero accumulated gradient have no strong direction
- **Structured row pruning** — remove entire low-norm rows
- **Structured column pruning** — remove entire low-norm columns
- **Random pruning** — baseline comparison

Plus a `PruneSchedule` for iterative pruning (gradual sparsity increase) and L1 norm utilities for row/column importance scoring.

9 tests covering all pruning strategies, statistics, norms, and scheduling.

Part of [SuperInstance](https://github.com/SuperInstance/SuperInstance).

License: MIT
