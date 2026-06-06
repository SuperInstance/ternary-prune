//! # ternary-prune
//!
//! Weight pruning strategies for ternary networks.
//! In ternary networks, pruning means setting weights to 0 (the neutral state).
//! The interesting question: which {-1, +1} weights should become 0?

pub type Trit = i8;

/// Pruning statistics.
#[derive(Debug, Clone)]
pub struct PruneStats {
    pub original_nonzero: usize,
    pub pruned_to_zero: usize,
    pub sparsity: f64,       // fraction that is 0
    pub density: f64,        // fraction that is nonzero
}

impl PruneStats {
    pub fn from_weights(weights: &[Trit]) -> Self {
        let nonzero = weights.iter().filter(|&&t| t != 0).count();
        let zeros = weights.len() - nonzero;
        let sparsity = zeros as f64 / weights.len() as f64;
        Self {
            original_nonzero: nonzero,
            pruned_to_zero: 0,
            sparsity,
            density: 1.0 - sparsity,
        }
    }
}

/// Magnitude pruning: set lowest-magnitude weights to 0.
/// In ternary land, "magnitude" means: weights that were recently uncertain
/// (flipped between -1 and +1) are pruned first.
pub fn magnitude_prune(weights: &mut [Trit], flip_counts: &[usize], target_sparsity: f64) -> PruneStats {
    let total = weights.len();
    let target_zeros = (total as f64 * target_sparsity) as usize;
    let current_zeros = weights.iter().filter(|&&t| t == 0).count();
    let to_prune = target_zeros.saturating_sub(current_zeros);

    if to_prune == 0 {
        return PruneStats::from_weights(weights);
    }

    // Collect indices of non-zero weights, sorted by flip count (highest flip = most uncertain)
    let mut candidates: Vec<usize> = (0..total)
        .filter(|&i| weights[i] != 0)
        .collect();
    candidates.sort_by(|&a, &b| flip_counts[b].cmp(&flip_counts[a]));

    let pruned = candidates.len().min(to_prune);
    for i in candidates.iter().take(pruned) {
        weights[*i] = 0;
    }

    let mut stats = PruneStats::from_weights(weights);
    stats.pruned_to_zero = pruned;
    stats
}

/// Gradient-based pruning: zero out weights whose accumulated gradient is near zero.
pub fn gradient_prune(weights: &mut [Trit], gradient_sums: &[i64], threshold: i64) -> PruneStats {
    let mut pruned = 0;
    for i in 0..weights.len() {
        if weights[i] != 0 && gradient_sums[i].abs() < threshold {
            weights[i] = 0;
            pruned += 1;
        }
    }
    let mut stats = PruneStats::from_weights(weights);
    stats.pruned_to_zero = pruned;
    stats
}

/// Structured pruning: zero entire rows or columns.
pub fn structured_prune_row(weights: &mut [Trit], rows: usize, cols: usize, row_norms: &[f64], keep_rows: usize) -> PruneStats {
    assert_eq!(weights.len(), rows * cols);
    let mut indexed: Vec<(usize, f64)> = row_norms.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let prune_rows: Vec<usize> = indexed.iter().skip(keep_rows).map(|(i, _)| *i).collect();
    let mut pruned = 0;
    for row in prune_rows {
        for c in 0..cols {
            let idx = row * cols + c;
            if weights[idx] != 0 {
                weights[idx] = 0;
                pruned += 1;
            }
        }
    }
    let mut stats = PruneStats::from_weights(weights);
    stats.pruned_to_zero = pruned;
    stats
}

/// Structured pruning: zero entire columns.
pub fn structured_prune_col(weights: &mut [Trit], rows: usize, cols: usize, col_norms: &[f64], keep_cols: usize) -> PruneStats {
    assert_eq!(weights.len(), rows * cols);
    let mut indexed: Vec<(usize, f64)> = col_norms.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let prune_cols: Vec<usize> = indexed.iter().skip(keep_cols).map(|(i, _)| *i).collect();
    let mut pruned = 0;
    for col in prune_cols {
        for r in 0..rows {
            let idx = r * cols + col;
            if weights[idx] != 0 {
                weights[idx] = 0;
                pruned += 1;
            }
        }
    }
    let mut stats = PruneStats::from_weights(weights);
    stats.pruned_to_zero = pruned;
    stats
}

/// Random pruning: zero random weights up to target sparsity.
pub fn random_prune(weights: &mut [Trit], target_sparsity: f64, seed: u64) -> PruneStats {
    let total = weights.len();
    let target_zeros = (total as f64 * target_sparsity) as usize;
    let current_zeros = weights.iter().filter(|&&t| t == 0).count();
    let to_prune = target_zeros.saturating_sub(current_zeros);

    if to_prune == 0 {
        return PruneStats::from_weights(weights);
    }

    // Simple LCG PRNG
    let mut state = seed;
    let mut pruned = 0;
    let mut attempts = 0;
    while pruned < to_prune && attempts < total * 10 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let idx = (state as usize) % total;
        if weights[idx] != 0 {
            weights[idx] = 0;
            pruned += 1;
        }
        attempts += 1;
    }

    let mut stats = PruneStats::from_weights(weights);
    stats.pruned_to_zero = pruned;
    stats
}

/// Compute L1 norm of a row in a matrix (stored as flat slice).
pub fn row_l1_norm(weights: &[Trit], row: usize, cols: usize) -> f64 {
    let start = row * cols;
    let end = start + cols;
    weights[start..end].iter().map(|&t| t.abs() as f64).sum()
}

/// Compute L1 norm of a column in a matrix.
pub fn col_l1_norm(weights: &[Trit], col: usize, rows: usize, cols: usize) -> f64 {
    (0..rows).map(|r| weights[r * cols + col].abs() as f64).sum()
}

/// Iterative magnitude pruning schedule.
pub struct PruneSchedule {
    pub initial_sparsity: f64,
    pub final_sparsity: f64,
    pub total_steps: usize,
}

impl PruneSchedule {
    pub fn new(initial: f64, final_: f64, steps: usize) -> Self {
        Self { initial_sparsity: initial, final_sparsity: final_, total_steps: steps }
    }

    /// Get target sparsity for a given step.
    pub fn sparsity_at(&self, step: usize) -> f64 {
        if step >= self.total_steps {
            return self.final_sparsity;
        }
        let progress = step as f64 / self.total_steps as f64;
        self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * progress
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magnitude_prune_basic() {
        let mut w = vec![1, -1, 1, -1, 1, -1, 1, -1];
        let flips = vec![10, 2, 8, 1, 5, 0, 3, 1]; // index 0 (flip=10) most uncertain
        let stats = magnitude_prune(&mut w, &flips, 0.5);
        assert!(stats.sparsity >= 0.45);
        assert!(stats.pruned_to_zero > 0);
    }

    #[test]
    fn test_magnitude_prune_already_sparse() {
        let mut w = vec![0, 0, 0, 0, 1, -1];
        let flips = vec![0; 6];
        let stats = magnitude_prune(&mut w, &flips, 0.5);
        assert_eq!(stats.pruned_to_zero, 0); // already at 67% sparse
    }

    #[test]
    fn test_gradient_prune() {
        let mut w = vec![1, -1, 1, -1, 1, -1];
        let grads = vec![100, 2, 50, 1, 80, 90]; // indices 1,3 have low gradient
        let stats = gradient_prune(&mut w, &grads, 10);
        assert!(stats.pruned_to_zero >= 2);
    }

    #[test]
    fn test_structured_prune_row() {
        let mut w = vec![1, 1, -1, -1, 1, 1, // row 0: norm 6
                         0, 0, 0, 0, 0, 0,   // row 1: norm 0
                         1, -1, 1, -1, 0, 0]; // row 2: norm 4
        let norms = vec![6.0, 0.0, 4.0];
        let stats = structured_prune_row(&mut w, 3, 6, &norms, 1);
        assert!(stats.pruned_to_zero > 0);
        // Row 1 should be pruned (lowest norm)
        assert_eq!(&w[6..12], &[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_structured_prune_col() {
        let mut w = vec![1, 0, 1,  // col 0: norm 2
                         1, 0, -1,  // col 1: norm 0
                         1, 1, 1];  // col 2: norm 3
        let norms = vec![2.0, 0.0, 3.0];
        let stats = structured_prune_col(&mut w, 3, 3, &norms, 2);
        assert!(stats.pruned_to_zero > 0);
    }

    #[test]
    fn test_random_prune_reaches_target() {
        let mut w = vec![1; 100];
        let stats = random_prune(&mut w, 0.5, 42);
        let zeros = w.iter().filter(|&&t| t == 0).count();
        assert!(zeros >= 45); // approximate
    }

    #[test]
    fn test_prune_stats() {
        let w = vec![-1, 0, 1, 0, 0, 1];
        let stats = PruneStats::from_weights(&w);
        assert_eq!(stats.sparsity, 0.5);
        assert_eq!(stats.density, 0.5);
        assert_eq!(stats.original_nonzero, 3);
    }

    #[test]
    fn test_prune_schedule() {
        let sched = PruneSchedule::new(0.0, 0.8, 100);
        assert!((sched.sparsity_at(0) - 0.0).abs() < 1e-10);
        assert!((sched.sparsity_at(50) - 0.4).abs() < 1e-10);
        assert!((sched.sparsity_at(100) - 0.8).abs() < 1e-10);
        assert!((sched.sparsity_at(200) - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_row_col_norms() {
        let w = vec![1, -1, 1, 0, -1, 1]; // 2 rows, 3 cols
        assert_eq!(row_l1_norm(&w, 0, 3), 3.0);
        assert_eq!(row_l1_norm(&w, 1, 3), 2.0);
        assert_eq!(col_l1_norm(&w, 0, 2, 3), 1.0); // [1, 0]
        assert_eq!(col_l1_norm(&w, 1, 2, 3), 2.0); // [-1, -1]
    }
}
