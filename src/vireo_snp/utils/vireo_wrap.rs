use crate::vireo_snp::utils::vireo_base;
use crate::vireo_snp::utils::vireo_doublet;
use crate::vireo_snp::utils::vireo_model::Vireo;
use crate::PyValue;
use ndarray::{Array2, Ix3};
use std::collections::BTreeMap;

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
) -> Option<BTreeMap<String, PyValue>> {
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
    let gt_prior_use = if let Some(gt) = gt_prior_arr {
        if n_donor_use <= gt.shape()[1] {
            PyValue::ArrayF64(gt.clone().into_dyn())
        } else {
            PyValue::None
        }
    } else {
        PyValue::None
    };
    let mut models = Vec::new();
    for _ in 0..n_init {
        let mut model = Vireo::default();
        let gt_init = match &gt_prior_use {
            PyValue::ArrayF64(x) => match x.clone().into_dimensionality::<Ix3>() {
                Ok(x) => Some(x),
                Err(_) => return None,
            },
            _ => None,
        };
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
        let gt_init = match &gt_prior_use {
            PyValue::ArrayF64(x) => match x.clone().into_dimensionality::<Ix3>() {
                Ok(x) => Some(x),
                Err(_) => return None,
            },
            _ => None,
        };
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
        let mut model_map = BTreeMap::new();
        model_map.insert(
            "GT_prob".to_string(),
            PyValue::ArrayF64(model.gt_prob.clone().into_dyn()),
        );
        model_map.insert(
            "ID_prob".to_string(),
            PyValue::ArrayF64(model.id_prob.clone().into_dyn()),
        );
        model_map.insert(
            "ID_prior".to_string(),
            PyValue::ArrayF64(model.id_prior.clone().into_dyn()),
        );
        model_map.insert(
            "beta_mu".to_string(),
            PyValue::ArrayF64(model.beta_mu.clone().into_dyn()),
        );
        model_map.insert(
            "beta_sum".to_string(),
            PyValue::ArrayF64(model.beta_sum.clone().into_dyn()),
        );
        match vireo_doublet::predict_doublet(&model_map, &ad_arr, &dp_arr, true, true, None) {
            Some((doublet, singlet, llr)) => (
                PyValue::ArrayF64(doublet.into_dyn()),
                PyValue::ArrayF64(singlet.into_dyn()),
                PyValue::F64Vec(llr),
            ),
            _ => return None,
        }
    } else {
        let pairs = n_donor_base * (n_donor_base - 1) / 2;
        (
            PyValue::ArrayF64(Array2::<f64>::zeros((ad_arr.ncols(), pairs)).into_dyn()),
            PyValue::ArrayF64(model.id_prob.clone().into_dyn()),
            PyValue::F64Vec(vec![0.0; ad_arr.ncols()]),
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
    let mut rv = BTreeMap::new();
    rv.insert("ID_prob".to_string(), id_prob);
    rv.insert(
        "GT_prob".to_string(),
        PyValue::ArrayF64(model.gt_prob.clone().into_dyn()),
    );
    rv.insert("doublet_LLR".to_string(), doublet_llr);
    rv.insert("doublet_prob".to_string(), doublet_prob);
    rv.insert(
        "theta_shapes".to_string(),
        PyValue::ArrayF64(theta_shapes.into_dyn()),
    );
    rv.insert(
        "theta_mean".to_string(),
        PyValue::ArrayF64(model.beta_mu.clone().into_dyn()),
    );
    rv.insert(
        "theta_sum".to_string(),
        PyValue::ArrayF64(model.beta_sum.clone().into_dyn()),
    );
    rv.insert("ambient_Psi".to_string(), PyValue::None);
    rv.insert("Psi_var".to_string(), PyValue::None);
    rv.insert("Psi_LLRatio".to_string(), PyValue::None);
    rv.insert("LB_list".to_string(), PyValue::F64Vec(elbo_all));
    rv.insert(
        "LB_doublet".to_string(),
        PyValue::F64(*model.elbo_.last().unwrap_or(&0.0)),
    );
    Some(rv)
}
