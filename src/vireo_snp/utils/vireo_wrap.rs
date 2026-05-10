use crate::vireo_snp::utils::vireo_base;
use crate::vireo_snp::utils::vireo_doublet;
use crate::vireo_snp::utils::vireo_model::Vireo;
use ndarray::{Array2, Array3};

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

pub fn vireo_wrap(
    ad_arr: &Array2<f64>,
    dp_arr: &Array2<f64>,
    gt_prior_arr: Option<&ndarray::Array3<f64>>,
    n_donor: Option<usize>,
    learn_gt_bool: bool,
    mut n_init: usize,
    _random_seed: Option<u64>,
    check_doublet: bool,
    max_iter_init_usize: usize,
    delay_fit_theta_usize: usize,
    mut n_extra_donor: usize,
    extra_donor_mode_str: Option<&str>,
    _check_ambient: bool,
    _nproc: usize,
    ase_mode_bool: bool,
    fix_beta_sum_bool: bool,
    n_gt_usize: usize,
) -> Option<VireoWrapResult> {
    let mut n_donor_use_base = n_donor;
    if n_donor_use_base.is_none() {
        n_donor_use_base = gt_prior_arr.map(|x| x.shape()[1]);
    }
    let Some(n_donor_base) = n_donor_use_base else {
        return None;
    };
    if !learn_gt_bool && n_extra_donor > 0 {
        n_extra_donor = 0;
    }
    if !learn_gt_bool && n_init > 1 {
        n_init = 1;
    }
    let n_donor_use = n_donor_base + n_extra_donor;
    let gt_prior_use: Option<Array3<f64>> = if let Some(gt) = gt_prior_arr {
        if n_donor_use <= gt.shape()[1] {
            Some(gt.clone())
        } else {
            None
        }
    } else {
        None
    };
    let mut models = Vec::new();
    for _ in 0..n_init {
        let mut model = Vireo::default();
        let gt_init = gt_prior_use.clone();
        if model
            .__init__(
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
            )
            .is_none()
        {
            return None;
        }
        if model.set_prior(gt_init, None, None, None, None).is_none() {
            return None;
        }
        if model
            .fit(
                &ad_arr,
                &dp_arr,
                max_iter_init_usize,
                5,
                None,
                delay_fit_theta_usize,
                false,
                None,
                1,
            )
            .is_none()
        {
            return None;
        }
        models.push(model);
    }
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
        if model
            .fit(&ad_arr, &dp_arr, 200, 5, None, 0, false, None, 1)
            .is_none()
        {
            return None;
        }
    } else {
        let gt_prob = model.gt_prob.clone();
        let id_prob_arr = model.id_prob.clone();
        let id_prob = match vireo_base::donor_select(
            &gt_prob,
            &id_prob_arr,
            n_donor_base,
            extra_donor_mode_str,
        ) {
            Some(x) => x,
            None => return None,
        };
        let mut next = Vireo::default();
        let beta_mu = Some(model.beta_mu.clone());
        let beta_sum = Some(model.beta_sum.clone());
        let id_prob = Some(id_prob);
        let gt_init = gt_prior_use.clone();
        if next
            .__init__(
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
            )
            .is_none()
        {
            return None;
        }
        if next.set_prior(gt_init, None, None, None, None).is_none() {
            return None;
        }
        if next
            .fit(
                &ad_arr,
                &dp_arr,
                200,
                5,
                None,
                delay_fit_theta_usize,
                false,
                None,
                1,
            )
            .is_none()
        {
            return None;
        }
        model = next;
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
    Some(VireoWrapResult {
        id_prob,
        gt_prob: model.gt_prob,
        doublet_llr,
        doublet_prob,
        theta_shapes,
        theta_mean: model.beta_mu,
        theta_sum: model.beta_sum,
        ambient_psi: None,
        psi_var: None,
        psi_llratio: None,
        lb_list: elbo_all,
        lb_doublet: *model.elbo_.last().unwrap_or(&0.0),
    })
}
