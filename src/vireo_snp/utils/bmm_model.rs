use crate::vireo_snp::utils::vireo_base;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use statrs::function::gamma::digamma;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BinomMixtureVB {
    pub elbo_inits: Vec<f64>,
    pub elbo_iters: Vec<f64>,
    pub id_prior: Array2<f64>,
    pub id_prob: Array2<f64>,
    pub id_prob_init: Option<Array2<f64>>,
    pub beta_mu: Array2<f64>,
    pub beta_mu_init: Option<Array2<f64>>,
    pub beta_sum: Array2<f64>,
    pub beta_sum_init: Option<Array2<f64>>,
    pub fix_beta_sum: bool,
    pub n_cell: usize,
    pub n_donor: usize,
    pub n_var: usize,
    pub theta_s1_prior: Array2<f64>,
    pub theta_s2_prior: Array2<f64>,
}

impl BinomMixtureVB {
    pub fn __init__(
        &mut self,
        n_cell: usize,
        n_var: usize,
        n_donor: usize,
        fix_beta_sum: bool,
        beta_mu_init: Option<Array2<f64>>,
        beta_sum_init: Option<Array2<f64>>,
        id_prob_init: Option<Array2<f64>>,
    ) -> Option<()> {
        self.n_var = n_var;
        self.n_cell = n_cell;
        self.n_donor = n_donor;
        self.fix_beta_sum = fix_beta_sum;
        self.id_prob_init = id_prob_init.clone();
        self.beta_mu_init = beta_mu_init.clone();
        self.beta_sum_init = beta_sum_init.clone();
        self.set_prior(None, None, None)?;
        self.set_initial(beta_mu_init, beta_sum_init, id_prob_init)
    }

    pub fn set_initial(
        &mut self,
        beta_mu_init: Option<Array2<f64>>,
        beta_sum_init: Option<Array2<f64>>,
        id_prob_init: Option<Array2<f64>>,
    ) -> Option<()> {
        let n_var = self.n_var;
        let n_cell = self.n_cell;
        let n_donor = self.n_donor;
        self.beta_mu = match beta_mu_init {
            None => Array2::from_elem((n_var, n_donor), 0.5),
            Some(value) => value,
        };
        self.beta_sum = match beta_sum_init {
            None => Array2::from_elem((self.beta_mu.shape()[0], self.beta_mu.shape()[1]), 30.0),
            Some(value) => value,
        };
        self.id_prob = match id_prob_init {
            None => {
                let mut rng = StdRng::seed_from_u64(0);
                let mut id = Array2::<f64>::zeros((n_cell, n_donor));
                for v in id.iter_mut() {
                    *v = rng.gen::<f64>();
                }
                match vireo_base::normalize(&id, Some(1)) {
                    Some(x) => x,
                    None => return None,
                }
            }
            Some(x) => match vireo_base::normalize(&x, Some(1)) {
                Some(x) => x,
                None => return None,
            },
        };
        self.elbo_iters.clear();
        Some(())
    }

    pub fn set_prior(
        &mut self,
        id_prior: Option<Array2<f64>>,
        beta_mu_prior: Option<Array2<f64>>,
        beta_sum_prior: Option<Array2<f64>>,
    ) -> Option<()> {
        let n_var = self.n_var;
        let n_cell = self.n_cell;
        let n_donor = self.n_donor;
        let beta_mu_prior = match beta_mu_prior {
            Some(x) => x,
            None => Array2::from_elem((n_var, n_donor), 0.5),
        };
        let beta_sum_prior = match beta_sum_prior {
            Some(x) => x,
            None => Array2::from_elem(beta_mu_prior.raw_dim(), 2.0),
        };
        self.theta_s1_prior = &beta_mu_prior * &beta_sum_prior;
        self.theta_s2_prior = (&beta_mu_prior * -1.0 + 1.0) * &beta_sum_prior;
        self.id_prior = match id_prior {
            Some(x) => x,
            None => match vireo_base::normalize(&Array2::ones((n_cell, n_donor)), Some(1)) {
                Some(x) => x,
                None => return None,
            },
        };
        Some(())
    }

    pub fn theta_s1(&self) -> Option<Array2<f64>> {
        Some(&self.beta_mu * &self.beta_sum)
    }

    pub fn theta_s2(&self) -> Option<Array2<f64>> {
        Some((&self.beta_mu * -1.0 + 1.0) * &self.beta_sum)
    }

    pub fn get_E_logLik(&self, ad: &Array2<f64>, dp: &Array2<f64>) -> Option<Array2<f64>> {
        if ad.raw_dim() != dp.raw_dim() {
            return None;
        }
        let bd = dp - ad;
        let theta_s1 = match self.theta_s1() {
            Some(x) => x,
            None => return None,
        };
        let theta_s2 = match self.theta_s2() {
            Some(x) => x,
            None => return None,
        };
        let dig1 = theta_s1.mapv(digamma);
        let dig2 = theta_s2.mapv(digamma);
        let digsum = (&theta_s1 + &theta_s2).mapv(digamma);
        Some(ad.t().dot(&dig1) + bd.t().dot(&dig2) - dp.t().dot(&digsum))
    }

    pub fn update_theta_size(&mut self, ad: &Array2<f64>, dp: &Array2<f64>) -> Option<()> {
        if ad.raw_dim() != dp.raw_dim() {
            return None;
        }
        let bd = dp - ad;
        let theta_s1 = ad.dot(&self.id_prob) + &self.theta_s1_prior;
        let theta_s2 = bd.dot(&self.id_prob) + &self.theta_s2_prior;
        self.beta_mu = &theta_s1 / (&theta_s1 + &theta_s2);
        if !self.fix_beta_sum {
            self.beta_sum = theta_s1 + theta_s2;
        }
        Some(())
    }

    pub fn update_ID_prob(&mut self, log_lik: &Array2<f64>) -> Option<()> {
        let log_lik_prior = log_lik + &self.id_prior.mapv(f64::ln);
        let amplified = match vireo_base::loglik_amplify(&log_lik_prior, Some(1)) {
            Some(x) => x,
            None => return None,
        };
        self.id_prob = match vireo_base::normalize(&amplified.mapv(f64::exp), Some(1)) {
            Some(x) => x,
            None => return None,
        };
        Some(())
    }

    pub fn get_ELBO(&self, log_lik: &Array2<f64>) -> Option<f64> {
        let lb_p = (log_lik * &self.id_prob).sum();
        let mut kl_id = 0.0;
        for (p, q) in self.id_prob.iter().zip(self.id_prior.iter()) {
            if *p > 0.0 && *q > 0.0 {
                kl_id += p * (p / q).ln();
            }
        }
        let theta_s1 = match self.theta_s1() {
            Some(x) => x,
            None => return None,
        };
        let theta_s2 = match self.theta_s2() {
            Some(x) => x,
            None => return None,
        };
        let mut x = Array2::<f64>::zeros((theta_s1.len(), 2));
        let mut xp = Array2::<f64>::zeros((theta_s1.len(), 2));
        for (i, (((a, b), pa), pb)) in theta_s1
            .iter()
            .zip(theta_s2.iter())
            .zip(self.theta_s1_prior.iter())
            .zip(self.theta_s2_prior.iter())
            .enumerate()
        {
            x[[i, 0]] = *a;
            x[[i, 1]] = *b;
            xp[[i, 0]] = *pa;
            xp[[i, 1]] = *pb;
        }
        let kl_theta = match vireo_base::beta_entropy(&x, Some(&xp), None) {
            Some(v) => v.sum(),
            None => return None,
        };
        Some(lb_p - kl_id - kl_theta)
    }

    pub fn _fit_BV(
        &mut self,
        ad: &Array2<f64>,
        dp: &Array2<f64>,
        max_iter: usize,
        min_iter: usize,
        epsilon_conv: f64,
        _verbose: bool,
    ) -> Option<()> {
        let mut elbo = vec![0.0; max_iter];
        let mut last = 0usize;
        for it in 0..max_iter {
            self.update_theta_size(ad, dp)?;
            let log_lik_id = self.get_E_logLik(ad, dp)?;
            self.update_ID_prob(&log_lik_id)?;
            elbo[it] = self.get_ELBO(&log_lik_id)?;
            last = it;
            if it > min_iter
                && elbo[it] - elbo[it - 1] >= -1e-6
                && elbo[it] - elbo[it - 1] < epsilon_conv
            {
                break;
            }
        }
        self.elbo_iters.extend_from_slice(&elbo[..=last]);
        Some(())
    }

    pub fn fit(
        &mut self,
        ad_arr: &Array2<f64>,
        dp_arr: &Array2<f64>,
        n_init: usize,
        max_iter: usize,
        max_iter_pre: Option<usize>,
        _random_seed: u64,
    ) -> Option<()> {
        let max_iter_pre_use = max_iter_pre.unwrap_or(100);
        let binom_coeff = vireo_base::get_binom_coeff(ad_arr, dp_arr, 700.0)
            .iter()
            .sum::<f64>();
        let mut elbo_inits = Vec::new();
        let mut best: Option<(Array2<f64>, Array2<f64>, Array2<f64>, f64, Vec<f64>)> = None;
        for i in 0..n_init {
            let beta_mu_init = self.beta_mu_init.clone();
            let beta_sum_init = self.beta_sum_init.clone();
            let id_prob_init = self.id_prob_init.clone();
            self.set_initial(beta_mu_init, beta_sum_init, id_prob_init)?;
            self._fit_BV(ad_arr, dp_arr, max_iter_pre_use, 20, 1e-2, true)?;
            let last_elbo = *self.elbo_iters.last().unwrap_or(&f64::NEG_INFINITY);
            elbo_inits.push(last_elbo);
            if i == 0
                || last_elbo
                    > best
                        .as_ref()
                        .map(|(_, _, _, e, _)| *e)
                        .unwrap_or(f64::NEG_INFINITY)
            {
                best = Some((
                    self.id_prob.clone(),
                    self.beta_mu.clone(),
                    self.beta_sum.clone(),
                    last_elbo,
                    self.elbo_iters.clone(),
                ));
            }
        }
        if let Some((id, mu, sum, _, elbo_iters)) = best {
            self.set_initial(Some(mu), Some(sum), Some(id))?;
            self.elbo_iters = elbo_iters;
        }
        self._fit_BV(ad_arr, dp_arr, max_iter, 20, 1e-2, true)?;
        for v in &mut self.elbo_iters {
            *v += binom_coeff;
        }
        self.elbo_inits = elbo_inits.into_iter().map(|v| v + binom_coeff).collect();
        Some(())
    }
}
