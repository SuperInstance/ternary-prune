//! # ternary-prune
//!
//! Weight pruning strategies for ternary networks.
//! In ternary networks, pruning means setting weights to 0 (the neutral state).
//! The interesting question: which {-1, +1} weights should become 0?
//!
//! Connected to [`ternary-types`](https://github.com/SuperInstance/ternary-types)
//! via its dependency — use `ternary_types::Ternary` for cross-crate interop.

/// A trit value: -1, 0, or +1.
pub type Trit = i8;

/// Pruning statistics.
#[derive(Debug, Clone)]
pub struct PruneStats {
    pub original_nonzero: usize,
    pub pruned_to_zero: usize,
    pub sparsity: f64,    // fraction that is 0
    pub density: f64,     // fraction that is nonzero
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

    // Find indices of non-zero weights, sorted by flip count (ascending)
    let mut candidates: Vec<(usize, usize)> = weights.iter()
        .enumerate()
        .filter(|(_, &w)| w != 0)
        .map(|(i, _)| (i, flip_counts[i]))
        .collect();

    candidates.sort_by_key(|&(_, flips)| flips);

    let pruned = candidates.len().min(to_prune);
    for &(idx, _) in &candidates[..pruned] {
        weights[idx] = 0;
    }

    let mut stats = PruneStats::from_weights(weights);
    stats.pruned_to_zero = pruned;
    stats
}

/// Gradient magnitude pruning: prune weights with smallest gradient magnitudes.
pub fn gradient_prune(weights: &mut [Trit], gradients: &[f64], target_sparsity: f64) -> PruneStats {
    let total = weights.len();
    let target_zeros = (total as f64 * target_sparsity) as usize;
    let current_zeros = weights.iter().filter(|&&t| t == 0).count();
    let to_prune = target_zeros.saturating_sub(current_zeros);

    if to_prune == 0 {
        return PruneStats::from_weights(weights);
    }

    let mut candidates: Vec<(usize, f64)> = weights.iter()
        .enumerate()
        .filter(|(_, &w)| w != 0)
        .map(|(i, _)| (i, gradients[i].abs()))
        .collect();

    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let pruned = candidates.len().min(to_prune);
    for &(idx, _) in &candidates[..pruned] {
        weights[idx] = 0;
    }

    let mut stats = PruneStats::from_weights(weights);
    stats.pruned_to_zero = pruned;
    stats
}

/// Structured pruning: prune entire rows/columns where uncertainty is high.
/// Counts flips per row, prunes rows with most flip activity.
pub fn structured_prune(weights: &mut [Vec<Trit>], flip_counts: &[Vec<usize>], target_sparsity: f64) -> PruneStats {
    let total_rows = weights.len();
    if total_rows == 0 {
        return PruneStats::from_weights(&[]);
    }
    let total_weights: usize = weights.iter().map(|r| r.len()).sum();
    let target_zeros = (total_weights as f64 * target_sparsity) as usize;
    let current_zeros: usize = weights.iter().flat_map(|r| r.iter()).filter(|&&t| t == 0).count();
    let to_prune = target_zeros.saturating_sub(current_zeros);

    if to_prune == 0 || total_weights == 0 {
        return PruneStats::from_weights(&weights.iter().flat_map(|r| r.iter().copied()).collect::<Vec<_>>());
    }

    // Score each row by average flip count
    let mut row_scores: Vec<(usize, f64)> = (0..total_rows)
        .map(|i| {
            let avg = if weights[i].is_empty() { 0.0 }
                     else { flip_counts[i].iter().sum::<usize>() as f64 / weights[i].len() as f64 };
            (i, avg)
        })
        .collect();
    row_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut pruned_count = 0;
    for &(row_idx, _) in &row_scores {
        if pruned_count >= to_prune {
            break;
        }
        for w in &mut weights[row_idx] {
            if *w != 0 {
                *w = 0;
                pruned_count += 1;
                if pruned_count >= to_prune {
                    break;
                }
            }
        }
    }

    let flat: Vec<Trit> = weights.iter().flat_map(|r| r.iter().copied()).collect();
    let mut stats = PruneStats::from_weights(&flat);
    stats.pruned_to_zero = pruned_count;
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_weights_all_zero() {
        let w = vec![0, 0, 0];
        let s = PruneStats::from_weights(&w);
        assert_eq!(s.sparsity, 1.0);
        assert_eq!(s.density, 0.0);
    }

    #[test]
    fn test_from_weights_all_nonzero() {
        let w = vec![-1, 1, -1];
        let s = PruneStats::from_weights(&w);
        assert_eq!(s.sparsity, 0.0);
        assert_eq!(s.density, 1.0);
    }

    #[test]
    fn test_magnitude_prune_half() {
        let mut w = vec![1, -1, 1, -1];
        let flips = vec![0, 10, 5, 1];
        let s = magnitude_prune(&mut w, &flips, 0.5);
        // target 2 zeros, 0 already zero → prune 2
        assert_eq!(s.pruned_to_zero, 2);
        assert_eq!(w.iter().filter(|&&t| t == 0).count(), 2);
    }

    #[test]
    fn test_gradient_prune() {
        let mut w = vec![1, -1, 1, -1];
        let grads = vec![0.1, 10.0, 0.05, 100.0];
        let s = gradient_prune(&mut w, &grads, 0.5);
        assert_eq!(s.pruned_to_zero, 2);
        // Indices 0 and 2 (smallest gradients) should be pruned
        assert_eq!(w[0], 0);
        assert_eq!(w[2], 0);
    }

    #[test]
    fn test_structured_prune() {
        let mut w = vec![
            vec![-1, -1, -1],  // row 0
            vec![1, 1, 1],     // row 1
        ];
        let flips = vec![
            vec![5, 5, 5],    // row 0: high uncertainty → pruned first
            vec![0, 0, 0],    // row 1: stable
        ];
        let s = structured_prune(&mut w, &flips, 0.5);
        assert_eq!(s.pruned_to_zero, 3);
        assert!(w[0].iter().all(|&t| t == 0));
    }

    #[test]
    fn test_no_prune_needed() {
        let mut w = vec![0, 0, 0, 1];
        let flips = vec![0, 0, 0, 5];
        let s = magnitude_prune(&mut w, &flips, 0.75);
        assert_eq!(s.pruned_to_zero, 0);
    }
}
