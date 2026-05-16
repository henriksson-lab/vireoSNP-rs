//! Wrap function for running the Vireo model with multiple initialisations.

use crate::vireo_snp::utils::vireo_base;
use crate::vireo_snp::utils::vireo_doublet;
use crate::vireo_snp::utils::vireo_model::Vireo;
use ndarray::{Array2, Array3};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Aggregated outputs from [`vireo_wrap`].
///
/// Holds the final cell-donor assignments, donor genotype posteriors, doublet
/// predictions, allelic rate parameters, and optional ambient-RNA estimates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VireoWrapResult {
    pub id_prob: Array2<f64>,
    pub gt_prob: ndarray::Array3<f64>,
    pub doublet_llr: Vec<f64>,
    pub doublet_prob: Array2<f64>,
    pub theta_shapes: Array2<f64>,
    pub theta_mean: Array2<f64>,
    pub theta_sum: Array2<f64>,
    pub ambient_psi: Option<Array2<f64>>,
    pub psi_var: Option<Array2<f64>>,
    pub psi_llratio: Option<Vec<f64>>,
    pub lb_list: Vec<f64>,
    pub lb_doublet: f64,
}

/// Helper that fits a single [`Vireo`] model and returns it.
///
/// Used to run model fitting as a parallel task across multiple initialisations.
pub fn _model_fit(
    mut model: Vireo,
    ad: &Array2<f64>,
    dp: &Array2<f64>,
    max_iter: usize,
    delay_fit_theta: usize,
) -> Option<Vireo> {
    model.fit(ad, dp, max_iter, 5, None, delay_fit_theta, false, None, 1)?;
    Some(model)
}

/// Reshape a `(n_variant, n_donor, n_gt)` tensor into a 2-D matrix of shape
/// `(n_variant * n_gt, n_donor)` for matching purposes.
fn flatten_gt_by_donor(gt: &Array3<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((gt.shape()[0] * gt.shape()[2], gt.shape()[1]));
    for v in 0..gt.shape()[0] {
        for d in 0..gt.shape()[1] {
            for g in 0..gt.shape()[2] {
                out[[v * gt.shape()[2] + g, d]] = gt[[v, d, g]];
            }
        }
    }
    out
}

/// Return a new genotype tensor keeping only the donor indices in `donors`,
/// preserving their order along the donor axis.
fn subset_gt_donors(gt: &Array3<f64>, donors: &[usize]) -> Array3<f64> {
    let mut out = Array3::<f64>::zeros((gt.shape()[0], donors.len(), gt.shape()[2]));
    for v in 0..gt.shape()[0] {
        for (new_d, &old_d) in donors.iter().enumerate() {
            for g in 0..gt.shape()[2] {
                out[[v, new_d, g]] = gt[[v, old_d, g]];
            }
        }
    }
    out
}

/// Reorder the donor columns of a cell-by-donor identity probability matrix
/// according to `donors`.
fn reorder_id_donors(id_prob: &Array2<f64>, donors: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((id_prob.nrows(), donors.len()));
    for (new_d, &old_d) in donors.iter().enumerate() {
        out.column_mut(new_d).assign(&id_prob.column(old_d));
    }
    out
}

/// Build and fit one initialisation of the Vireo model.
///
/// The RNG seed is derived from `random_seed` and the initialisation index
/// `im` so that each initialisation explores a different starting point.
fn fit_initial_model(
    im: usize,
    ad_arr: &Array2<f64>,
    dp_arr: &Array2<f64>,
    gt_prior_use: Option<&Array3<f64>>,
    n_donor_use: usize,
    n_gt_usize: usize,
    learn_gt_bool: bool,
    ase_mode_bool: bool,
    fix_beta_sum_bool: bool,
    max_iter_init_usize: usize,
    delay_fit_theta_usize: usize,
    random_seed: u64,
) -> Option<Vireo> {
    let mut model = Vireo::default();
    model.set_rng_seed(random_seed.wrapping_add(im as u64));
    let gt_init = gt_prior_use.cloned();
    model.__init__(
        ad_arr.ncols(),
        ad_arr.nrows(),
        n_donor_use,
        n_gt_usize,
        learn_gt_bool,
        true,
        ase_mode_bool,
        fix_beta_sum_bool,
        None,
        None,
        None,
        gt_init.clone(),
    )?;
    model.set_prior(gt_init, None, None, None, None)?;
    model.fit(
        ad_arr,
        dp_arr,
        max_iter_init_usize,
        5,
        None,
        delay_fit_theta_usize,
        false,
        None,
        1,
    )?;
    Some(model)
}

/// Run Vireo with multiple initialisations and return the best result.
///
/// Performs `n_init` initialisations (optionally in parallel), keeps the model
/// with the highest ELBO, optionally explores extra donors and refines the
/// genotype prior, and finally predicts doublets and (optionally) ambient RNA
/// contamination.
///
/// Returns `None` if no donor count can be determined or if any of the
/// underlying model-fitting or prediction steps fail.
pub fn vireo_wrap(
    ad_arr: &Array2<f64>,
    dp_arr: &Array2<f64>,
    gt_prior_arr: Option<&ndarray::Array3<f64>>,
    n_donor: Option<usize>,
    learn_gt_bool: bool,
    mut n_init: usize,
    random_seed: Option<u64>,
    check_doublet: bool,
    max_iter_init_usize: usize,
    delay_fit_theta_usize: usize,
    mut n_extra_donor: usize,
    extra_donor_mode_str: Option<&str>,
    check_ambient: bool,
    nproc: usize,
    ase_mode_bool: bool,
    fix_beta_sum_bool: bool,
    n_gt_usize: usize,
) -> Option<VireoWrapResult> {
    let mut n_donor_use_base = n_donor;
    if n_donor_use_base.is_none() {
        n_donor_use_base = gt_prior_arr.map(|x| x.shape()[1]);
    }
    let n_donor_base = n_donor_use_base?;
    if !learn_gt_bool && n_extra_donor > 0 {
        n_extra_donor = 0;
    }
    if !learn_gt_bool && n_init > 1 {
        n_init = 1;
    }
    let mut n_donor_use = n_donor_base + n_extra_donor;
    let gt_prior_use: Option<Array3<f64>> = if let Some(gt) = gt_prior_arr {
        if n_donor_use <= gt.shape()[1] {
            n_donor_use = gt.shape()[1];
            Some(gt.clone())
        } else {
            None
        }
    } else {
        None
    };
    let random_seed = random_seed.unwrap_or(0);
    #[cfg(feature = "parallel")]
    let mut models: Vec<Vireo> = if nproc > 1 && n_init > 1 {
        (0..n_init)
            .into_par_iter()
            .map(|im| {
                fit_initial_model(
                    im,
                    ad_arr,
                    dp_arr,
                    gt_prior_use.as_ref(),
                    n_donor_use,
                    n_gt_usize,
                    learn_gt_bool,
                    ase_mode_bool,
                    fix_beta_sum_bool,
                    max_iter_init_usize,
                    delay_fit_theta_usize,
                    random_seed,
                )
            })
            .collect::<Option<Vec<_>>>()?
    } else {
        (0..n_init)
            .map(|im| {
                fit_initial_model(
                    im,
                    ad_arr,
                    dp_arr,
                    gt_prior_use.as_ref(),
                    n_donor_use,
                    n_gt_usize,
                    learn_gt_bool,
                    ase_mode_bool,
                    fix_beta_sum_bool,
                    max_iter_init_usize,
                    delay_fit_theta_usize,
                    random_seed,
                )
            })
            .collect::<Option<Vec<_>>>()?
    };
    #[cfg(not(feature = "parallel"))]
    let mut models: Vec<Vireo> = (0..n_init)
        .map(|im| {
            fit_initial_model(
                im,
                ad_arr,
                dp_arr,
                gt_prior_use.as_ref(),
                n_donor_use,
                n_gt_usize,
                learn_gt_bool,
                ase_mode_bool,
                fix_beta_sum_bool,
                max_iter_init_usize,
                delay_fit_theta_usize,
                random_seed,
            )
        })
        .collect::<Option<Vec<_>>>()?;
    let elbo_all: Vec<f64> = models
        .iter()
        .map(|m| *m.elbo_.last().unwrap_or(&f64::NEG_INFINITY))
        .collect();
    let best_idx = elbo_all
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut model = models.remove(best_idx);
    if n_extra_donor == 0 {
        model.fit(ad_arr, dp_arr, 200, 5, None, 0, false, None, 1)?;
    } else {
        let gt_prob = model.gt_prob.clone();
        let id_prob_arr = model.id_prob.clone();
        let id_prob =
            vireo_base::donor_select(&gt_prob, &id_prob_arr, n_donor_base, extra_donor_mode_str)?;
        let mut next = Vireo::default();
        next.set_rng_seed(random_seed.wrapping_add(n_init as u64));
        let beta_mu = Some(model.beta_mu.clone());
        let beta_sum = Some(model.beta_sum.clone());
        let id_prob = Some(id_prob);
        let gt_init = gt_prior_use.clone();
        next.__init__(
            ad_arr.ncols(),
            ad_arr.nrows(),
            n_donor_base,
            n_gt_usize,
            learn_gt_bool,
            true,
            ase_mode_bool,
            fix_beta_sum_bool,
            beta_mu,
            beta_sum,
            id_prob,
            gt_init.clone(),
        )?;
        next.set_prior(gt_init, None, None, None, None)?;
        next.fit(
            ad_arr,
            dp_arr,
            200,
            5,
            None,
            delay_fit_theta_usize,
            false,
            None,
            1,
        )?;
        model = next;
    }
    if let Some(gt_prior) = gt_prior_arr {
        if n_donor_base < gt_prior.shape()[1] {
            let mut donor_idx: Vec<usize> = (0..model.id_prob.ncols()).collect();
            donor_idx.sort_by(|&a, &b| {
                let sa = model.id_prob.column(a).sum();
                let sb = model.id_prob.column(b).sum();
                sb.total_cmp(&sa)
            });
            donor_idx.truncate(n_donor_base);
            let gt_prior_use = subset_gt_donors(gt_prior, &donor_idx);
            let mut next = Vireo::default();
            next.set_rng_seed(random_seed.wrapping_add(n_init as u64 + 1));
            next.__init__(
                ad_arr.ncols(),
                ad_arr.nrows(),
                n_donor_base,
                n_gt_usize,
                false,
                true,
                ase_mode_bool,
                fix_beta_sum_bool,
                None,
                None,
                None,
                Some(gt_prior_use),
            )?;
            next.fit(ad_arr, dp_arr, 200, 20, None, 0, false, None, 1)?;
            model = next;
        } else if n_donor_base > gt_prior.shape()[1] {
            let mut gt_prior_use = model.gt_prob.clone();
            let ref_gt = flatten_gt_by_donor(gt_prior);
            let new_gt = flatten_gt_by_donor(&gt_prior_use);
            let (_, idx, _) = vireo_base::optimal_match(&ref_gt, &new_gt, Some(1), false)?;
            for v in 0..gt_prior.shape()[0] {
                for (prior_d, &target_d) in idx.iter().enumerate() {
                    for g in 0..gt_prior.shape()[2] {
                        gt_prior_use[[v, target_d, g]] = gt_prior[[v, prior_d, g]];
                    }
                }
            }
            let mut idx_order = idx.clone();
            idx_order.extend((0..n_donor_base).filter(|d| !idx.contains(d)));
            gt_prior_use = subset_gt_donors(&gt_prior_use, &idx_order);
            let id_prob_use = reorder_id_donors(&model.id_prob, &idx_order);
            let mut next = Vireo::default();
            next.set_rng_seed(random_seed.wrapping_add(n_init as u64 + 1));
            next.__init__(
                ad_arr.ncols(),
                ad_arr.nrows(),
                n_donor_base,
                n_gt_usize,
                learn_gt_bool,
                true,
                ase_mode_bool,
                fix_beta_sum_bool,
                Some(model.beta_mu.clone()),
                Some(model.beta_sum.clone()),
                Some(id_prob_use),
                Some(gt_prior_use.clone()),
            )?;
            next.set_prior(Some(gt_prior_use), None, None, None, None)?;
            next.fit(ad_arr, dp_arr, 200, 20, None, 0, false, None, 1)?;
            model = next;
        }
    }
    let (doublet_prob, id_prob, doublet_llr) = if check_doublet {
        match vireo_doublet::predict_doublet(
            &model.gt_prob,
            &model.beta_mu,
            &model.beta_sum,
            Some(&model.id_prior),
            ad_arr,
            dp_arr,
            true,
            true,
            None,
        ) {
            Some((doublet, singlet, llr)) => (doublet, singlet, llr),
            _ => return None,
        }
    } else {
        let pairs = n_donor_base * (n_donor_base - 1) / 2;
        (
            Array2::<f64>::zeros((ad_arr.ncols(), pairs)),
            model.id_prob.clone(),
            vec![0.0; ad_arr.ncols()],
        )
    };
    let s1 = &model.beta_mu * &model.beta_sum;
    let s2 = (&model.beta_mu * -1.0 + 1.0) * &model.beta_sum;
    let mut theta_shapes = Array2::<f64>::zeros((s1.shape()[0] + s2.shape()[0], s1.shape()[1]));
    for i in 0..s1.shape()[0] {
        for j in 0..s1.shape()[1] {
            theta_shapes[[i, j]] = s1[[i, j]];
            theta_shapes[[i + s1.shape()[0], j]] = s2[[i, j]];
        }
    }
    let (ambient_psi, psi_var, psi_llratio) = if check_ambient {
        match vireo_doublet::predit_ambient(
            &model.gt_prob,
            &model.beta_mu,
            &model.id_prob,
            ad_arr,
            dp_arr,
            nproc,
            None,
        ) {
            Some((psi, var, llr)) => (Some(psi), Some(var), Some(llr)),
            None => return None,
        }
    } else {
        (None, None, None)
    };
    Some(VireoWrapResult {
        id_prob,
        gt_prob: model.gt_prob,
        doublet_llr,
        doublet_prob,
        theta_shapes,
        theta_mean: model.beta_mu,
        theta_sum: model.beta_sum,
        ambient_psi,
        psi_var,
        psi_llratio,
        lb_list: elbo_all,
        lb_doublet: *model.elbo_.last().unwrap_or(&0.0),
    })
}
