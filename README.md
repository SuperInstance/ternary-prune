# Ternary Prune — Network Pruning for Ternary Neural Networks

**Ternary Prune** implements weight pruning strategies specifically for ternary networks where weights are in {-1, 0, +1}. Pruning in ternary land means setting non-zero weights to 0. The crate provides three strategies: **magnitude pruning** (prune weights that flip frequently — they're uncertain), **gradient pruning** (prune weights with smallest gradient magnitudes), and **structured pruning** (prune entire rows/columns/channels).

## Why It Matters

Ternary networks are already 16× smaller than FP32 networks, but they can be compressed further. A typical ternary network has 30-50% zero weights after training; pruning can push that to 70-90% with minimal accuracy loss. This matters for deployment: sparse ternary matrices can skip zero entries entirely (no computation needed), making inference proportional to the non-zero count rather than the matrix size. The three pruning strategies target different signals: magnitude pruning identifies structurally unimportant weights (those that oscillate between -1 and +1), gradient pruning finds weights that aren't learning, and structured pruning enables hardware-friendly sparsity patterns.

## How It Works

### Magnitude Pruning

In ternary networks, "magnitude" is meaningless (all non-zero weights have |w| = 1). Instead, **flip count** serves as the magnitude proxy: weights that frequently changed between -1 and +1 during training are uncertain and safe to prune. The algorithm:

1. Sort non-zero weights by flip count (ascending)
2. Prune the lowest-flip-count weights until target sparsity is reached
3. Set those weights to 0

O(n log n) for sorting, O(n) for pruning.

### Gradient Pruning

Prune weights with the smallest accumulated gradient magnitudes — they contribute least to learning:

1. Sort non-zero weights by |gradient| (ascending)
2. Prune the smallest-gradient weights to reach target sparsity

O(n log n). This is the ternary analog of standard gradient-based pruning.

### Structured Pruning

Remove entire structures (rows, columns, channels) where the average weight magnitude is low:

1. Compute per-structure statistics (mean |non-zero weight|, non-zero fraction)
2. Rank structures by sparsity (most sparse = least important)
3. Zero out entire structures until target is reached

Structured sparsity enables hardware optimizations that unstructured sparsity cannot.

### Statistics

`PruneStats` tracks: `original_nonzero`, `pruned_to_zero`, `sparsity` (fraction zero), `density` (fraction non-zero). These are computed in O(n).

## Quick Start

```rust
use ternary_prune::{magnitude_prune, PruneStats, Trit};

let mut weights: Vec<Trit> = vec![1, -1, 0, 1, -1, 1, 0, -1, 1, -1];
let flip_counts: Vec<usize> = vec![5, 3, 0, 1, 8, 2, 0, 1, 15, 4];

// Prune to 60% sparsity
let stats = magnitude_prune(&mut weights, &flip_counts, 0.6);
println!("Pruned {} weights. Sparsity: {:.0}%", stats.pruned_to_zero, stats.sparsity * 100.0);
```

```bash
cargo add ternary-prune
```

## API

| Type / Function | Description |
|---|---|
| `magnitude_prune(&mut [Trit], &[usize], f64)` | Prune by flip count (uncertain weights first) |
| `gradient_prune(&mut [Trit], &[f64], f64)` | Prune by gradient magnitude |
| `PruneStats` | `{ original_nonzero, pruned_to_zero, sparsity, density }` |
| `PruneStats::from_weights(&[Trit])` | Compute current sparsity |

## Architecture Notes

Pruning increases the η (entropy/zero) fraction in the γ + η = C conservation law: every pruned weight moves from γ (active, ±1) to η (inactive, 0). The conservation total C (parameter count) is fixed — pruning redistributes the budget. See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Han, Song et al. "Learning Both Weights and Connections for Efficient Neural Networks," *NeurIPS*, 2015 — magnitude pruning.
- Zhu, Maode & Gupta, Suyog. "To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression," *ICLR Workshop*, 2018.
- Li, Hao et al. "Pruning Filters for Efficient ConvNets," *ICLR*, 2017 — structured pruning.

## License

MIT
