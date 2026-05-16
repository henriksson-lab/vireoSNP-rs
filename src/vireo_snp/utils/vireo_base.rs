use ndarray::{Array1, Array2, Array3, ArrayBase, Axis, Data, Dimension, RemoveAxis};
use statrs::function::beta::ln_beta;
use statrs::function::gamma::{digamma, ln_gamma};

/// Result of [`optimal_match`]: row indices, matched column indices, and an
/// optional difference matrix.
type OptimalMatchResult = (Vec<usize>, Vec<usize>, Option<Array2<f64>>);

/// Solves a rectangular linear assignment problem on `cost` using the
/// Hungarian algorithm, returning `(row_indices, col_indices)` of the
/// minimum-cost assignment. Handles `rows > cols` by transposing.
fn hungarian_minimize(cost: &Array2<f64>) -> (Vec<usize>, Vec<usize>) {
    let rows = cost.nrows();
    let cols = cost.ncols();
    if rows == 0 || cols == 0 {
        return (Vec::new(), Vec::new());
    }
    if rows > cols {
        let transposed = cost.t().to_owned();
        let (cols_as_rows, rows_as_cols) = hungarian_minimize(&transposed);
        let mut pairs: Vec<(usize, usize)> = rows_as_cols.into_iter().zip(cols_as_rows).collect();
        pairs.sort_by_key(|&(row, _)| row);
        return pairs.into_iter().unzip();
    }

    // Hungarian algorithm for rectangular cost matrices with rows <= columns.
    // This follows the standard 1-indexed potential formulation.
    let n = rows;
    let m = cols;
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; m + 1];
    let mut p = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=m {
                if used[j] {
                    continue;
                }
                let cur = cost[[i0 - 1, j - 1]] - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![usize::MAX; n];
    for j in 1..=m {
        if p[j] != 0 {
            assignment[p[j] - 1] = j - 1;
        }
    }
    let idx0: Vec<usize> = (0..n).filter(|&i| assignment[i] != usize::MAX).collect();
    let idx1: Vec<usize> = idx0.iter().map(|&i| assignment[i]).collect();
    (idx0, idx1)
}

/// Computes log binomial coefficients `log(C(dp, ad))` for entries where
/// `dp > 0`, clipping each value to `max_val` and rounding to `f32` precision.
///
/// The 64-bit floating point representation tops out around `e**700`, close to
/// `binom(1000, 500)`, hence the `max_val` cap. Prefer [`logbincoeff`] for new
/// code.
pub fn get_binom_coeff<S1, S2, D>(
    ad: &ArrayBase<S1, D>,
    dp: &ArrayBase<S2, D>,
    max_val: f64,
) -> Vec<f64>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    D: Dimension,
{
    ad.iter()
        .zip(dp.iter())
        .filter(|(_, dp)| **dp > 0.0)
        .map(|(ad, dp)| {
            let v = ln_gamma(dp + 1.0) - ln_gamma(ad + 1.0) - ln_gamma(dp - ad + 1.0);
            v.min(max_val) as f32 as f64
        })
        .collect()
}

/// Computes `log(n! / (k! (n - k)!))` element-wise via `ln_gamma`.
///
/// Returns `None` if `n` and `k` do not have matching shapes.
pub fn logbincoeff<S1, S2, D>(
    n: &ArrayBase<S1, D>,
    k: &ArrayBase<S2, D>,
) -> Option<ndarray::Array<f64, D>>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    D: Dimension,
{
    if n.shape() != k.shape() {
        return None;
    }
    let mut out = n.to_owned();
    for ((out, n), k) in out.iter_mut().zip(n.iter()).zip(k.iter()) {
        *out = ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0);
    }
    Some(out)
}

/// Normalizes a tensor so that elements along `axis` sum to 1.
///
/// When `axis` is `None`, the last axis is used. Returns `None` if `axis` is
/// out of bounds.
pub fn normalize<S, D>(x: &ArrayBase<S, D>, axis: Option<usize>) -> Option<ndarray::Array<f64, D>>
where
    S: Data<Elem = f64>,
    D: Dimension + RemoveAxis,
{
    let axis = axis.unwrap_or_else(|| x.ndim().saturating_sub(1));
    if axis >= x.ndim() {
        return None;
    }
    let sums = x.sum_axis(Axis(axis));
    let mut out = x.to_owned();
    for (mut lane, sum) in out.lanes_mut(Axis(axis)).into_iter().zip(sums.iter()) {
        lane.mapv_inplace(|v| v / *sum);
    }
    Some(out)
}

/// Alias for [`normalize`] kept for parity with the Python API.
pub fn tensor_normalize<S, D>(
    x: &ArrayBase<S, D>,
    axis: Option<usize>,
) -> Option<ndarray::Array<f64, D>>
where
    S: Data<Elem = f64>,
    D: Dimension + RemoveAxis,
{
    normalize(x, axis)
}

/// Amplifies a log-likelihood tensor by subtracting the maximum along `axis`
/// from each lane, improving numerical stability before exponentiation.
///
/// When `axis` is `None`, the last axis is used. Returns `None` if `axis` is
/// out of bounds.
pub fn loglik_amplify<S, D>(
    x: &ArrayBase<S, D>,
    axis: Option<usize>,
) -> Option<ndarray::Array<f64, D>>
where
    S: Data<Elem = f64>,
    D: Dimension + RemoveAxis,
{
    let axis = axis.unwrap_or_else(|| x.ndim().saturating_sub(1));
    if axis >= x.ndim() {
        return None;
    }
    let maxes = x.map_axis(Axis(axis), |lane| {
        lane.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    });
    let mut out = x.to_owned();
    for (mut lane, max_value) in out.lanes_mut(Axis(axis)).into_iter().zip(maxes.iter()) {
        lane.mapv_inplace(|v| v - *max_value);
    }
    Some(out)
}

/// Computes the entropy of beta distributions parameterized by the rows of
/// `x` (shape `(N, 2)`). When `x_prior` is supplied, returns the
/// Kullback-Leibler divergence between the two sets of beta distributions
/// instead.
///
/// # Arguments
/// * `x` - Beta-distribution shape parameters with shape `(N, 2)`.
/// * `x_prior` - Optional prior shape parameters with the same shape as `x`.
/// * `_axis` - Unused; kept for parity with the Python signature.
///
/// # Returns
/// A length-`N` vector of entropies (or KL divergences), or `None` if the
/// shapes are incompatible.
pub fn beta_entropy(
    x: &Array2<f64>,
    x_prior: Option<&Array2<f64>>,
    _axis: Option<usize>,
) -> Option<Array1<f64>> {
    let q = x_prior.unwrap_or(x);
    if x.ncols() != 2 || q.ncols() != 2 || x.nrows() != q.nrows() {
        return None;
    }
    let mut rv = Array1::<f64>::zeros(x.nrows());
    for i in 0..x.nrows() {
        let p0 = x[[i, 0]];
        let p1 = x[[i, 1]];
        let q0 = q[[i, 0]];
        let q1 = q[[i, 1]];
        let cross = ln_beta(q0, q1) - (q0 - 1.0) * digamma(p0) - (q1 - 1.0) * digamma(p1)
            + (q0 + q1 - 2.0) * digamma(p0 + p1);
        if x_prior.is_some() {
            let entropy = ln_beta(p0, p1) - (p0 - 1.0) * digamma(p0) - (p1 - 1.0) * digamma(p1)
                + (p0 + p1 - 2.0) * digamma(p0 + p1);
            rv[i] = cross - entropy;
        } else {
            rv[i] = cross;
        }
    }
    Some(rv)
}

/// Returns the cross entropy `-E_p[log q]` for pairs of beta distributions
/// whose shape parameters are given by the rows of `xp` and `xq`.
///
/// Both inputs must have two columns and the same number of rows; otherwise
/// `None` is returned. For the plain entropy, call with `xp == xq`.
pub fn _beta_cross_entropy(xp: &Array2<f64>, xq: &Array2<f64>) -> Option<Array1<f64>> {
    if xp.ncols() != 2 || xq.ncols() != 2 || xp.nrows() != xq.nrows() {
        return None;
    }
    let mut rv = Array1::<f64>::zeros(xp.nrows());
    for i in 0..xp.nrows() {
        rv[i] = ln_beta(xq[[i, 0]], xq[[i, 1]])
            - (xq[[i, 0]] - 1.0) * digamma(xp[[i, 0]])
            - (xq[[i, 1]] - 1.0) * digamma(xp[[i, 1]])
            + (xq[[i, 0]] + xq[[i, 1]] - 2.0) * digamma(xp[[i, 0]] + xp[[i, 1]]);
    }
    Some(rv)
}

/// Maps `new_ids` to `ref_ids`, returning, for each entry of `ref_ids`, the
/// index of the matching entry in `new_ids` (or `None` if absent).
///
/// `ref_ids` may contain repeated values, but `new_ids` is expected to hold
/// unique values, so that indexing `new_ids` with the result reproduces
/// `ref_ids`. When `uniq_ref_only` is `true`, each entry of `new_ids` is
/// consumed by at most one match.
///
/// # Arguments
/// * `ref_ids` - Reference identifiers (may contain duplicates).
/// * `new_ids` - Identifiers to be mapped onto `ref_ids`.
/// * `uniq_ref_only` - If `true`, advance the `new_ids` cursor after a match.
pub fn r#match(ref_ids: &[String], new_ids: &[String], uniq_ref_only: bool) -> Vec<Option<usize>> {
    let mut idx1: Vec<usize> = (0..ref_ids.len()).collect();
    let mut idx2: Vec<usize> = (0..new_ids.len()).collect();
    idx1.sort_by(|&a, &b| ref_ids[a].cmp(&ref_ids[b]));
    idx2.sort_by(|&a, &b| new_ids[a].cmp(&new_ids[b]));
    let mut rt_idx1 = Vec::new();
    let mut rt_idx2 = Vec::new();
    let mut i = 0;
    let mut j = 0;
    while i < idx1.len() {
        if j == idx2.len() || ref_ids[idx1[i]] < new_ids[idx2[j]] {
            rt_idx1.push(idx1[i]);
            rt_idx2.push(None);
            i += 1;
        } else if ref_ids[idx1[i]] == new_ids[idx2[j]] {
            rt_idx1.push(idx1[i]);
            rt_idx2.push(Some(idx2[j]));
            i += 1;
            if uniq_ref_only {
                j += 1;
            }
        } else {
            j += 1;
        }
    }
    let mut origin_idx: Vec<usize> = (0..rt_idx1.len()).collect();
    origin_idx.sort_by_key(|&i| rt_idx1[i]);
    origin_idx.into_iter().map(|i| rt_idx2[i]).collect()
}

/// Matches the slices of `z` to those of `x` along `axis` so as to minimize
/// the mean absolute difference, using the Hungarian algorithm.
///
/// After the match, `z` indexed by `idx1` aligns with `x` indexed by `idx0`
/// along `axis`. When `return_delta` is `true`, the full pairwise difference
/// matrix is also returned.
///
/// `axis` defaults to 1; values greater than 1 are rejected with `None`.
pub fn optimal_match(
    x: &Array2<f64>,
    z: &Array2<f64>,
    axis: Option<usize>,
    return_delta: bool,
) -> Option<OptimalMatchResult> {
    let axis = axis.unwrap_or(1);
    if axis > 1 {
        return None;
    }
    let n = if axis == 0 { x.nrows() } else { x.ncols() };
    let m = if axis == 0 { z.nrows() } else { z.ncols() };
    let mut diff = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let mut total = 0.0;
            let mut count = 0usize;
            if axis == 0 {
                for k in 0..x.ncols().min(z.ncols()) {
                    total += (x[[i, k]] - z[[j, k]]).abs();
                    count += 1;
                }
            } else {
                for k in 0..x.nrows().min(z.nrows()) {
                    total += (x[[k, i]] - z[[k, j]]).abs();
                    count += 1;
                }
            }
            diff[[i, j]] = total / count as f64;
        }
    }
    let (idx0, idx1) = hungarian_minimize(&diff);
    Some((idx0, idx1, return_delta.then_some(diff)))
}

/// Deprecated convenience wrapper that returns only the matched indices from
/// [`optimal_match`]. Prefer calling [`optimal_match`] directly.
pub fn greed_match(x: &Array2<f64>, z: &Array2<f64>, axis: Option<usize>) -> Option<Vec<usize>> {
    optimal_match(x, z, axis, false).map(|(_, idx1, _)| idx1)
}

/// Selects `n_donor` donors from a candidate set that may contain extras.
///
/// The genotype prior `gt_prob` and identity assignments `id_prob` may
/// describe more donors than `n_donor`. Two selection strategies are
/// supported via `mode`:
/// * `"size"` - keep the `n_donor` donors with the largest assigned cell mass.
/// * `"distance"` (default) - greedily keep the `n_donor` donors whose
///   genotype profiles are most distinct from one another.
///
/// Returns the trimmed `id_prob` columns, floored at `1e-10`, or `None` if
/// the donor counts in `gt_prob` and `id_prob` disagree.
pub fn donor_select(
    gt_prob: &Array3<f64>,
    id_prob: &Array2<f64>,
    n_donor: usize,
    mode: Option<&str>,
) -> Option<Array2<f64>> {
    let mode = mode.unwrap_or("distance");
    if id_prob.ncols() != gt_prob.shape()[1] {
        return None;
    }
    let gt = gt_prob;
    let id = id_prob;
    let donor_cnt = id.sum_axis(Axis(0));
    let mut donor_idx: Vec<usize>;
    if mode == "size" {
        donor_idx = (0..donor_cnt.len()).collect();
        donor_idx.sort_by(|&a, &b| donor_cnt[b].total_cmp(&donor_cnt[a]));
    } else {
        let donors = gt.shape()[1];
        let mut gt_diff = Array2::<f64>::zeros((donors, donors));
        for i in 0..donors {
            for j in 0..donors {
                let mut total = 0.0;
                let mut count = 0usize;
                for v in 0..gt.shape()[0] {
                    for g in 0..gt.shape()[2] {
                        total += (gt[[v, i, g]] - gt[[v, j, g]]).abs();
                        count += 1;
                    }
                }
                gt_diff[[i, j]] = total / count as f64;
            }
        }
        donor_idx = vec![(0..donor_cnt.len())
            .max_by(|&a, &b| donor_cnt[a].total_cmp(&donor_cnt[b]))
            .unwrap_or(0)];
        while donor_idx.len() < donors {
            let mut best = None;
            for j in 0..donors {
                if donor_idx.contains(&j) {
                    continue;
                }
                let min_dist = donor_idx
                    .iter()
                    .map(|&i| gt_diff[[i, j]])
                    .fold(f64::INFINITY, f64::min);
                if best.is_none_or(|(_, v)| min_dist > v) {
                    best = Some((j, min_dist));
                }
            }
            if let Some((j, _)) = best {
                donor_idx.push(j);
            } else {
                break;
            }
        }
    }
    let keep = n_donor.min(donor_idx.len());
    let mut out = Array2::<f64>::zeros((id.nrows(), keep));
    for (new_j, &old_j) in donor_idx.iter().take(keep).enumerate() {
        for i in 0..id.nrows() {
            out[[i, new_j]] = id[[i, old_j]].max(1e-10);
        }
    }
    Some(out)
}
