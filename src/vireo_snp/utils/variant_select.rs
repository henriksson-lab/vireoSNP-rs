use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use statrs::function::gamma::digamma;
use std::collections::BTreeMap;

pub fn barcode_entropy(x: &[String], y: Option<&[String]>) -> Option<(f64, Vec<String>)> {
    let z_str = match y {
        None => x.to_vec(),
        Some(y) if y.len() == x.len() => x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| format!("{a}{b}"))
            .collect(),
        Some(_) => return None,
    };
    let mut counts = BTreeMap::<String, usize>::new();
    for z in &z_str {
        *counts.entry(z.clone()).or_insert(0) += 1;
    }
    let total = z_str.len() as f64;
    let entropy = counts.values().fold(0.0, |acc, &count| {
        let p = count as f64 / total;
        acc - p * p.log2()
    });
    Some((entropy, z_str))
}

pub fn variant_select(
    gt: &Array2<f64>,
    var_count: Option<&[f64]>,
    rand_seed: u64,
) -> Option<(f64, Vec<String>, Vec<usize>)> {
    if let Some(counts) = var_count {
        if counts.len() != gt.nrows() {
            return None;
        }
    }
    let mut rng = StdRng::seed_from_u64(rand_seed);
    let k = gt.ncols();
    let mut entropy_now = 0.0;
    let mut variant_set = Vec::new();
    let mut barcode_set = vec!["#".to_string(); k];
    let mut entropy_all = vec![0.0; gt.nrows()];
    let mut barcode_all = vec![barcode_set.clone(); gt.nrows()];
    loop {
        for i in 0..gt.nrows() {
            let y: Vec<String> = gt.row(i).iter().map(|v| v.to_string()).collect();
            let (entropy, barcode) = barcode_entropy(&barcode_set, Some(&y))?;
            entropy_all[i] = entropy;
            barcode_all[i] = barcode;
        }
        let max_entropy = entropy_all
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if max_entropy == entropy_now {
            break;
        }
        let mut idx: Vec<usize> = entropy_all
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v == max_entropy { Some(i) } else { None })
            .collect();
        if let Some(counts) = var_count {
            let mut selected_counts: Vec<f64> = idx.iter().map(|&i| counts[i]).collect();
            selected_counts.sort_by(f64::total_cmp);
            let median = selected_counts[selected_counts.len() / 2];
            idx.retain(|&i| counts[i] >= median);
        }
        let idx_use = idx[rng.gen_range(0..idx.len())];
        variant_set.push(idx_use);
        barcode_set = barcode_all[idx_use].clone();
        entropy_now = entropy_all[idx_use];
    }
    Some((entropy_now, barcode_set, variant_set))
}

pub fn variant_ELBO_gain(
    id_prob: &Array2<f64>,
    ad: &Array2<f64>,
    dp: &Array2<f64>,
    pseudocount: f64,
) -> Option<Array1<f64>> {
    if ad.raw_dim() != dp.raw_dim() || ad.ncols() != id_prob.nrows() {
        return None;
    }
    let bd = dp - ad;
    let s1_m2 = ad.dot(id_prob) + pseudocount;
    let s2_m2 = bd.dot(id_prob) + pseudocount;
    let ss_m2 = dp.dot(id_prob) + pseudocount * 2.0;
    let mut elbo2 = Array1::<f64>::zeros(s1_m2.nrows());
    for i in 0..s1_m2.nrows() {
        let row: Vec<f64> = (0..s1_m2.ncols())
            .map(|j| {
                s1_m2[[i, j]] * digamma(s1_m2[[i, j]]) + s2_m2[[i, j]] * digamma(s2_m2[[i, j]])
                    - ss_m2[[i, j]] * digamma(ss_m2[[i, j]])
            })
            .collect();
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        elbo2[i] = max + row.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
    }
    let s1_m1 = ad.sum_axis(Axis(1)).insert_axis(Axis(1)) + pseudocount;
    let s2_m1 = bd.sum_axis(Axis(1)).insert_axis(Axis(1)) + pseudocount;
    let ss_m1 = dp.sum_axis(Axis(1)).insert_axis(Axis(1)) + pseudocount * 2.0;
    let mut elbo1 = Array1::<f64>::zeros(s1_m1.nrows());
    for i in 0..s1_m1.nrows() {
        let v = s1_m1[[i, 0]] * digamma(s1_m1[[i, 0]]) + s2_m1[[i, 0]] * digamma(s2_m1[[i, 0]])
            - ss_m1[[i, 0]] * digamma(ss_m1[[i, 0]]);
        elbo1[i] = v;
    }
    Some(&elbo2 - &elbo1)
}
