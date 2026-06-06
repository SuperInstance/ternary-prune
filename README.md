# ternary-prune

*Weight pruning for ternary networks. The question isn't "which weights to remove?" — it's "which {-1, +1} weights should become 0?"*

## Why This Exists

In float networks, pruning means setting small weights to zero. In ternary networks, all non-zero weights have the same magnitude (±1). So "magnitude pruning" doesn't apply the same way — you can't just threshold by |w|. Instead, ternary pruning asks: *which weights contribute least to the network's output?*

This crate implements five pruning strategies, each answering that question differently, plus a scheduled pruning regime that gradually increases sparsity during training.

## Pruning Strategies

### 1. Magnitude Prune (`magnitude_prune`)
Uses accumulated flip counts — weights that change sign frequently during training are uncertain. Prune the most uncertain weights first.

### 2. Gradient Prune (`gradient_prune`)
Uses gradient magnitude. Weights with near-zero accumulated gradients have minimal impact on the loss. Prune them.

### 3. Structured Row Prune (`structured_prune_row`)
Remove entire rows (output neurons) based on L1 norm. A row with low L1 norm contributes little.

### 4. Structured Column Prune (`structured_prune_col`)
Remove entire columns (input features) based on L1 norm.

### 5. Random Prune (`random_prune`)
Baseline: randomly set weights to 0. Useful for ablation studies.

### Scheduled Pruning (`PruneSchedule`)
Gradually increase sparsity over training: start at 0%, ramp to target over N epochs.

## Usage

```rust
use ternary_prune::*;

let mut weights: Vec<i8> = vec![-1, 1, 0, -1, 1, 0, -1, 1];
let flip_counts: Vec<usize> = vec![2, 15, 0, 3, 12, 0, 1, 8];

// Prune weights that flip most often (30% sparsity target)
let stats = magnitude_prune(&mut weights, &flip_counts, 0.3);
println!("Pruned {} weights to 0", stats.zeros_added);

// Structured pruning — keep only top 3 rows
let mut row_weights = vec![-1i8, 1, 0, -1, 1, 1, -1, 1, 0, 1, -1, 1];
let row_norms: Vec<f64> = vec![2.0, 3.0, 2.0];
let stats = structured_prune_row(&mut row_weights, 3, 2, &row_norms, 2);
```

## The Deeper Insight

Ternary pruning is secretly a form of *feature selection*. When you prune a column (input feature) to all zeros, you've removed that feature entirely — the network no longer sees it. This means ternary networks with pruning can discover which features matter, without separate feature importance analysis.

The connection to `ternary-quantize` is direct: quantization converts float → ternary, pruning converts ternary → sparser ternary. Together they form the compression pipeline for deployment.

## Related Crates

- `ternary-distill` — Knowledge distillation (compression via teaching)
- `ternary-quantize` — Float → ternary quantization
- `ternary-checkpoint` — Save pruned model checkpoints
- `ternary-accumulator` — Gradient tracking for informed pruning
