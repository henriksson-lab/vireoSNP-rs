use crate::vireo_snp::utils::variant_select;
use crate::vireo_snp::utils::vireo_base;
use ndarray::{Array1, Array2, Array3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub fn predict_doublet(
    gt_prob: &Array3<f64>,
    beta_mu: &Array2<f64>,
    beta_sum: &Array2<f64>,
    id_prior: Option<&Array2<f64>>,
    ad: &Array2<f64>,
    dp: &Array2<f64>,
    _update_gt: bool,
    _update_id: bool,
    doublet_rate_prior: Option<f64>,
) -> Option<(Array2<f64>, Array2<f64>, Vec<f64>)> {
    if ad.raw_dim() != dp.raw_dim() {
        return None;
    }
    let id_prior = match id_prior {
        Some(x) => x.clone(),
        None => Array2::from_elem(
            (ad.ncols(), gt_prob.shape()[1]),
            1.0 / gt_prob.shape()[1] as f64,
        ),
    };
    let gt_both = add_doublet_GT(gt_prob)?;
    let (beta_mu_both, beta_sum_both) = add_doublet_theta(beta_mu, beta_sum)?;
    let n_donor = gt_prob.shape()[1];
    let n_doublet_pair = gt_both.shape()[1] - n_donor;
    let doublet_rate_prior = doublet_rate_prior.unwrap_or(0.5f64.min(ad.ncols() as f64 / 100000.0));
    let mut id_prior_both = Array2::<f64>::zeros((ad.ncols(), gt_both.shape()[1]));
    for i in 0..ad.ncols() {
        for d in 0..n_donor {
            id_prior_both[[i, d]] =
                id_prior[[i.min(id_prior.nrows() - 1), d]] * (1.0 - doublet_rate_prior);
        }
        for d in 0..n_doublet_pair {
            id_prior_both[[i, n_donor + d]] = doublet_rate_prior / n_doublet_pair as f64;
        }
    }
    let bd = dp - ad;
    let mut log_lik_id = Array2::<f64>::zeros((ad.ncols(), gt_both.shape()[1]));
    for g in 0..gt_both.shape()[2] {
        let mut weighted1 = Array2::<f64>::zeros((gt_both.shape()[0], gt_both.shape()[1]));
        let mut weighted2 = weighted1.clone();
        let mut weighteds = weighted1.clone();
        for v in 0..gt_both.shape()[0] {
            let theta_row = if beta_mu_both.nrows() == 1 { 0 } else { v };
            for d in 0..gt_both.shape()[1] {
                weighted1[[v, d]] = gt_both[[v, d, g]]
                    * statrs::function::gamma::digamma(
                        beta_sum_both[[theta_row, g]] * beta_mu_both[[theta_row, g]],
                    );
                weighted2[[v, d]] = gt_both[[v, d, g]]
                    * statrs::function::gamma::digamma(
                        beta_sum_both[[theta_row, g]] * (1.0 - beta_mu_both[[theta_row, g]]),
                    );
                weighteds[[v, d]] = gt_both[[v, d, g]]
                    * statrs::function::gamma::digamma(beta_sum_both[[theta_row, g]]);
            }
        }
        log_lik_id =
            log_lik_id + ad.t().dot(&weighted1) + bd.t().dot(&weighted2) - dp.t().dot(&weighteds);
    }
    let mut log_lik_ratio = Vec::with_capacity(ad.ncols());
    for i in 0..ad.ncols() {
        let singlet = (0..n_donor)
            .map(|d| log_lik_id[[i, d]])
            .fold(f64::NEG_INFINITY, f64::max);
        let doublet = (n_donor..gt_both.shape()[1])
            .map(|d| log_lik_id[[i, d]])
            .fold(f64::NEG_INFINITY, f64::max);
        log_lik_ratio.push(doublet - singlet);
    }
    let log_lik_id_prior = &log_lik_id + &id_prior_both.mapv(f64::ln);
    let id_prob_exp = match vireo_base::loglik_amplify(&log_lik_id_prior, None) {
        Some(x) => x.mapv(f64::exp),
        None => return None,
    };
    let id_prob_both = vireo_base::normalize(&id_prob_exp, None)?;
    let mut prob_singlet = Array2::<f64>::zeros((ad.ncols(), n_donor));
    let mut prob_doublet = Array2::<f64>::zeros((ad.ncols(), n_doublet_pair));
    for i in 0..ad.ncols() {
        for d in 0..n_donor {
            prob_singlet[[i, d]] = id_prob_both[[i, d]];
        }
        for d in 0..n_doublet_pair {
            prob_doublet[[i, d]] = id_prob_both[[i, n_donor + d]];
        }
    }
    Some((prob_doublet, prob_singlet, log_lik_ratio))
}

pub fn add_doublet_theta(
    beta_mu: &Array2<f64>,
    beta_sum: &Array2<f64>,
) -> Option<(Array2<f64>, Array2<f64>)> {
    if beta_mu.raw_dim() != beta_sum.raw_dim() {
        return None;
    }
    let n_pair = beta_mu.ncols() * (beta_mu.ncols() - 1) / 2;
    let mut mu_out = Array2::<f64>::zeros((beta_mu.nrows(), beta_mu.ncols() + n_pair));
    let mut sum_out = Array2::<f64>::zeros((beta_sum.nrows(), beta_sum.ncols() + n_pair));
    for i in 0..beta_mu.nrows() {
        for j in 0..beta_mu.ncols() {
            mu_out[[i, j]] = beta_mu[[i, j]];
            sum_out[[i, j]] = beta_sum[[i, j]];
        }
    }
    let mut idx = beta_mu.ncols();
    for a in 0..beta_mu.ncols() {
        for b in (a + 1)..beta_mu.ncols() {
            for i in 0..beta_mu.nrows() {
                mu_out[[i, idx]] = (beta_mu[[i, a]] + beta_mu[[i, b]]) / 2.0;
                sum_out[[i, idx]] = (beta_sum[[i, a]] * beta_sum[[i, b]]).sqrt();
            }
            idx += 1;
        }
    }
    Some((mu_out, sum_out))
}

pub fn add_doublet_GT(gt_prob: &Array3<f64>) -> Option<Array3<f64>> {
    let n_var = gt_prob.shape()[0];
    let n_sample = gt_prob.shape()[1];
    let n_gt = gt_prob.shape()[2];
    let n_gt_pair = n_gt * (n_gt - 1) / 2;
    let n_sample_pair = n_sample * (n_sample - 1) / 2;
    let mut gt_prob2 = Array3::<f64>::zeros((n_var, n_sample_pair, n_gt + n_gt_pair));
    let mut sample_pairs = Vec::new();
    for a in 0..n_sample {
        for b in (a + 1)..n_sample {
            sample_pairs.push((a, b));
        }
    }
    let mut gt_pairs = Vec::new();
    for a in 0..n_gt {
        for b in (a + 1)..n_gt {
            gt_pairs.push((a, b));
        }
    }
    for (sp, &(s1, s2)) in sample_pairs.iter().enumerate() {
        for v in 0..n_var {
            for g in 0..n_gt {
                gt_prob2[[v, sp, g]] = gt_prob[[v, s1, g]] * gt_prob[[v, s2, g]];
            }
            for (gp, &(g1, g2)) in gt_pairs.iter().enumerate() {
                gt_prob2[[v, sp, n_gt + gp]] = gt_prob[[v, s1, g1]] * gt_prob[[v, s2, g2]]
                    + gt_prob[[v, s1, g2]] * gt_prob[[v, s2, g1]];
            }
        }
    }
    let gt_prob2 = vireo_base::normalize(&gt_prob2, Some(2))?;
    let mut out = Array3::<f64>::zeros((n_var, n_sample + n_sample_pair, n_gt + n_gt_pair));
    for v in 0..n_var {
        for s in 0..n_sample {
            for g in 0..n_gt {
                out[[v, s, g]] = gt_prob[[v, s, g]];
            }
        }
        for sp in 0..n_sample_pair {
            for g in 0..(n_gt + n_gt_pair) {
                out[[v, n_sample + sp, g]] = gt_prob2[[v, sp, g]];
            }
        }
    }
    Some(out)
}

pub fn _fit_EM_ambient(
    ad: &Array1<f64>,
    dp: &Array1<f64>,
    theta_mat: &Array2<f64>,
    n_donor: Option<usize>,
    max_iter: Option<usize>,
    min_iter: Option<usize>,
    epsilon_conv: Option<f64>,
    hessian: bool,
    _verbose: bool,
) -> Option<(Array1<f64>, Vec<f64>, f64)> {
    if ad.len() != dp.len() || ad.len() != theta_mat.nrows() {
        return None;
    }
    let n_donor = n_donor.unwrap_or(theta_mat.ncols());
    let max_iter = max_iter.unwrap_or(200);
    let min_iter = min_iter.unwrap_or(20);
    let epsilon_conv = epsilon_conv.unwrap_or(1e-3);
    let bd = dp - ad;
    let mut rng = StdRng::seed_from_u64(0);
    let mut psi = Array1::<f64>::zeros(theta_mat.ncols());
    for v in psi.iter_mut() {
        *v = -rng.gen::<f64>().ln();
    }
    psi /= psi.sum();
    let mut log_lik = vec![0.0; max_iter];
    let mut last = 0usize;
    for it in 0..max_iter {
        let mut mask_idx = Vec::new();
        if it >= min_iter.saturating_sub(3) && theta_mat.ncols() > n_donor {
            let mut order: Vec<usize> = (0..psi.len()).collect();
            order.sort_by(|&a, &b| psi[a].total_cmp(&psi[b]));
            mask_idx.extend_from_slice(&order[..theta_mat.ncols() - n_donor]);
        }
        let mut z1 = theta_mat.clone();
        let mut z0 = theta_mat.mapv(|v| 1.0 - v);
        for v in 0..theta_mat.nrows() {
            let mut s1 = 0.0;
            let mut s0 = 0.0;
            for d in 0..theta_mat.ncols() {
                if mask_idx.contains(&d) {
                    z1[[v, d]] = 0.0;
                    z0[[v, d]] = 0.0;
                } else {
                    z1[[v, d]] *= psi[d];
                    z0[[v, d]] *= psi[d];
                    s1 += z1[[v, d]];
                    s0 += z0[[v, d]];
                }
            }
            for d in 0..theta_mat.ncols() {
                z1[[v, d]] /= s1;
                z0[[v, d]] /= s0;
            }
        }
        let mut psi_raw = Array1::<f64>::zeros(psi.len());
        for d in 0..psi.len() {
            for v in 0..ad.len() {
                psi_raw[d] += ad[v] * z1[[v, d]] + bd[v] * z0[[v, d]];
            }
        }
        psi = &psi_raw / psi_raw.sum();
        let theta_vct = theta_mat.dot(&psi);
        log_lik[it] = (0..ad.len())
            .map(|v| ad[v] * theta_vct[v].ln() + bd[v] * (1.0 - theta_vct[v]).ln())
            .sum();
        last = it;
        if it > min_iter
            && log_lik[it] >= log_lik[it - 1]
            && log_lik[it] - log_lik[it - 1] < epsilon_conv
        {
            break;
        }
    }
    let var_crbound = if !hessian {
        vec![f64::NAN; psi.len()]
    } else {
        let theta_vct = theta_mat.dot(&psi);
        let mut fisher = Array1::<f64>::zeros(psi.len());
        for d in 0..psi.len() {
            for v in 0..ad.len() {
                fisher[d] += (theta_mat[[v, d]] / theta_vct[v]).powi(2) * ad[v]
                    + (theta_mat[[v, d]] / (1.0 - theta_vct[v])).powi(2) * bd[v];
            }
        }
        fisher.iter().map(|v| 1.0 / v).collect()
    };
    let mut log_lik_null = vec![0.0; theta_mat.ncols()];
    for item in log_lik_null.iter_mut().take(theta_mat.ncols()) {
        let min_p = 0.0;
        let mut psi_null =
            Array1::<f64>::from_elem(theta_mat.ncols(), min_p / (theta_mat.ncols() - 1) as f64);
        let max_i = psi
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        psi_null[max_i] = 1.0 - min_p;
        let theta_null = theta_mat.dot(&psi_null);
        *item = (0..ad.len())
            .map(|v| ad[v] * theta_null[v].ln() + bd[v] * (1.0 - theta_null[v]).ln())
            .sum();
    }
    let log_lik_ratio = log_lik[last] - log_lik_null.into_iter().fold(f64::NEG_INFINITY, f64::max);
    Some((psi, var_crbound, log_lik_ratio))
}

pub fn predit_ambient(
    gt_prob: &Array3<f64>,
    beta_mu: &Array2<f64>,
    id_prob: &Array2<f64>,
    ad: &Array2<f64>,
    dp: &Array2<f64>,
    nproc: usize,
    min_elbo_gain: Option<f64>,
) -> Option<(Array2<f64>, Array2<f64>, Vec<f64>)> {
    if ad.raw_dim() != dp.raw_dim() {
        return None;
    }
    let mut theta_mat = Array2::<f64>::zeros((gt_prob.shape()[0], gt_prob.shape()[1]));
    for v in 0..gt_prob.shape()[0] {
        for d in 0..gt_prob.shape()[1] {
            for g in 0..gt_prob.shape()[2] {
                theta_mat[[v, d]] += gt_prob[[v, d, g]] * beta_mu[[0, g]];
            }
        }
    }
    let min_elbo_gain = min_elbo_gain.unwrap_or((ad.ncols() as f64).sqrt() / 3.0);
    let elbo_gain = variant_select::variant_ELBO_gain(id_prob, ad, dp, 0.5)?;
    let snp_idx: Vec<usize> = elbo_gain
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v >= min_elbo_gain { Some(i) } else { None })
        .collect();
    let mut theta_use = Array2::<f64>::zeros((snp_idx.len(), theta_mat.ncols()));
    let mut ad_use = Array2::<f64>::zeros((snp_idx.len(), ad.ncols()));
    let mut dp_use = Array2::<f64>::zeros((snp_idx.len(), dp.ncols()));
    for (new_i, &old_i) in snp_idx.iter().enumerate() {
        theta_use.row_mut(new_i).assign(&theta_mat.row(old_i));
        ad_use.row_mut(new_i).assign(&ad.row(old_i));
        dp_use.row_mut(new_i).assign(&dp.row(old_i));
    }
    #[cfg(feature = "parallel")]
    let per_cell: Vec<(Array1<f64>, Vec<f64>, f64)> = if nproc > 1 && ad_use.ncols() > 1 {
        (0..ad_use.ncols())
            .into_par_iter()
            .map(|cell| {
                let ad_col = ad_use.column(cell).to_owned();
                let dp_col = dp_use.column(cell).to_owned();
                _fit_EM_ambient(
                    &ad_col, &dp_col, &theta_use, None, None, None, None, true, false,
                )
            })
            .collect::<Option<Vec<_>>>()?
    } else {
        (0..ad_use.ncols())
            .map(|cell| {
                let ad_col = ad_use.column(cell).to_owned();
                let dp_col = dp_use.column(cell).to_owned();
                _fit_EM_ambient(
                    &ad_col, &dp_col, &theta_use, None, None, None, None, true, false,
                )
            })
            .collect::<Option<Vec<_>>>()?
    };
    #[cfg(not(feature = "parallel"))]
    let per_cell: Vec<(Array1<f64>, Vec<f64>, f64)> = (0..ad_use.ncols())
        .map(|cell| {
            let ad_col = ad_use.column(cell).to_owned();
            let dp_col = dp_use.column(cell).to_owned();
            _fit_EM_ambient(
                &ad_col, &dp_col, &theta_use, None, None, None, None, true, false,
            )
        })
        .collect::<Option<Vec<_>>>()?;

    let mut psi_mat = Array2::<f64>::zeros((ad_use.ncols(), theta_use.ncols()));
    let mut psi_var = Array2::<f64>::zeros((ad_use.ncols(), theta_use.ncols()));
    let mut psi_loglik = Vec::with_capacity(ad_use.ncols());
    for (cell, (psi, var, llr)) in per_cell.into_iter().enumerate() {
        if psi.len() != theta_use.ncols() || var.len() != theta_use.ncols() {
            return None;
        }
        for (d, v) in psi.iter().enumerate() {
            psi_mat[[cell, d]] = *v;
        }
        for (d, v) in var.iter().enumerate() {
            psi_var[[cell, d]] = *v;
        }
        psi_loglik.push(llr);
    }
    Some((psi_mat, psi_var, psi_loglik))
}
