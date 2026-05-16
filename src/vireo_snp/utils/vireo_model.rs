//! Vireo model: Variational Inference for reconstruction of ensemble origin.
//!
//! The prior can be set via [`Vireo::set_prior`] before fitting the model.

use crate::vireo_snp::utils::vireo_base;
use ndarray::{Array2, Array3, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use statrs::function::gamma::digamma;

/// Vireo model state: variational inference for reconstructing donor-of-origin
/// assignments from single-cell allele count data.
///
/// Key fields:
/// - `beta_mu`, `beta_sum`: Beta mean and concentration parameters of theta's posterior,
///   shape `(1, n_gt)` or `(n_var, n_gt)` depending on ASE mode.
/// - `id_prob`: Posterior cell assignment probability to each donor, shape `(n_cell, n_donor)`.
/// - `gt_prob`: Posterior genotype probability per variant per donor,
///   shape `(n_var, n_donor, n_gt)`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Vireo {
    pub ase_mode: bool,
    pub elbo_: Vec<f64>,
    pub gt_prior: Array3<f64>,
    pub gt_prob: Array3<f64>,
    pub id_prior: Array2<f64>,
    pub id_prob: Array2<f64>,
    pub beta_mu: Array2<f64>,
    pub beta_sum: Array2<f64>,
    pub fix_beta_sum: bool,
    pub learn_gt: bool,
    pub learn_theta: bool,
    pub n_gt: usize,
    pub n_cell: usize,
    pub n_donor: usize,
    pub n_var: usize,
    pub rng_seed: u64,
    pub theta_s1_prior: Array2<f64>,
    pub theta_s2_prior: Array2<f64>,
}

impl Vireo {
    /// Initialise a Vireo model instance.
    ///
    /// Multiple initializations are highly recommended to avoid local optima.
    ///
    /// # Arguments
    /// * `n_cell` - Number of cells.
    /// * `n_var` - Number of variants.
    /// * `n_donor` - Number of donors.
    /// * `n_gt` - Number of genotype categories.
    /// * `learn_gt` - Whether to update `gt_prob`; otherwise the initial value is used.
    /// * `learn_theta` - Whether to update theta's posterior parameters.
    /// * `ase_mode` - Whether to set the allelic ratio `theta` to be variant specific.
    /// * `fix_beta_sum` - Whether to fix the concentration parameter of theta's posterior.
    /// * `beta_mu_init` - Optional initial value of `beta_mu`.
    /// * `beta_sum_init` - Optional initial value of `beta_sum`.
    /// * `id_prob_init` - Optional initial value of `id_prob`.
    /// * `gt_prob_init` - Optional initial value of `gt_prob`.
    pub fn __init__(
        &mut self,
        n_cell: usize,
        n_var: usize,
        n_donor: usize,
        n_gt: usize,
        learn_gt: bool,
        learn_theta: bool,
        ase_mode: bool,
        fix_beta_sum: bool,
        beta_mu_init: Option<Array2<f64>>,
        beta_sum_init: Option<Array2<f64>>,
        id_prob_init: Option<Array2<f64>>,
        gt_prob_init: Option<Array3<f64>>,
    ) -> Option<()> {
        self.n_gt = n_gt;
        self.n_var = n_var;
        self.n_cell = n_cell;
        self.n_donor = n_donor;
        self.learn_gt = learn_gt;
        self.learn_theta = learn_theta;
        self.ase_mode = ase_mode;
        self.fix_beta_sum = fix_beta_sum;
        self.elbo_.clear();
        self.set_initial(beta_mu_init, beta_sum_init, id_prob_init, gt_prob_init)?;
        self.set_prior(None, None, None, None, None)
    }

    /// Builder-style setter for the RNG seed used during random initialisation.
    pub fn with_rng_seed(mut self, seed: u64) -> Self {
        self.rng_seed = seed;
        self
    }

    /// Set the RNG seed used during random initialisation.
    pub fn set_rng_seed(&mut self, seed: u64) {
        self.rng_seed = seed;
    }

    /// Set initial values for `beta_mu`, `beta_sum`, `id_prob`, and `gt_prob`.
    ///
    /// When an initialisation argument is `None`, a default is used:
    /// `beta_mu` is set to a linear ramp from 0.01 to 0.99 across genotype categories,
    /// `beta_sum` is filled with 50, while `id_prob` and `gt_prob` are drawn from
    /// the seeded RNG and row-normalised.
    pub fn set_initial(
        &mut self,
        beta_mu_init: Option<Array2<f64>>,
        beta_sum_init: Option<Array2<f64>>,
        id_prob_init: Option<Array2<f64>>,
        gt_prob_init: Option<Array3<f64>>,
    ) -> Option<()> {
        let n_gt = self.n_gt;
        let n_var = self.n_var;
        let n_cell = self.n_cell;
        let n_donor = self.n_donor;
        let theta_len = if self.ase_mode { n_var } else { 1 };
        self.beta_mu = match beta_mu_init {
            None => {
                let mut mu = Array2::<f64>::zeros((theta_len, n_gt));
                for i in 0..theta_len {
                    for g in 0..n_gt {
                        mu[[i, g]] = if n_gt == 1 {
                            0.01
                        } else {
                            0.01 + (0.99 - 0.01) * g as f64 / (n_gt - 1) as f64
                        };
                    }
                }
                mu
            }
            Some(value) => value,
        };
        self.beta_sum = match beta_sum_init {
            None => Array2::from_elem((theta_len, n_gt), 50.0),
            Some(value) => value,
        };
        let mut rng = StdRng::seed_from_u64(self.rng_seed);
        self.id_prob = match id_prob_init {
            None => {
                let mut id = Array2::<f64>::zeros((n_cell, n_donor));
                for v in id.iter_mut() {
                    *v = rng.gen::<f64>();
                }
                vireo_base::normalize(&id, Some(1))?
            }
            Some(x) => vireo_base::normalize(&x, Some(1))?,
        };
        self.gt_prob = match gt_prob_init {
            None => {
                let mut gt = Array3::<f64>::zeros((n_var, n_donor, n_gt));
                for v in gt.iter_mut() {
                    *v = rng.gen::<f64>();
                }
                vireo_base::normalize(&gt, Some(2))?
            }
            Some(x) => vireo_base::normalize(&x, None)?,
        };
        Some(())
    }

    /// Set priors for key variables: theta, `gt_prob`, and `id_prob`.
    ///
    /// The priors are in the same shape as their corresponding variables.
    /// `min_gp` (default `0.00001`) clamps the minimum (and `1 - min_gp` the maximum)
    /// genotype probability in the genotype prior.
    pub fn set_prior(
        &mut self,
        gt_prior: Option<Array3<f64>>,
        id_prior: Option<Array2<f64>>,
        beta_mu_prior: Option<Array2<f64>>,
        beta_sum_prior: Option<Array2<f64>>,
        min_gp: Option<f64>,
    ) -> Option<()> {
        let beta_mu_prior = match beta_mu_prior {
            Some(x) => x,
            None => {
                let mut mu = Array2::<f64>::zeros((1, self.beta_mu.ncols()));
                for g in 0..self.beta_mu.ncols() {
                    mu[[0, g]] = if self.beta_mu.ncols() == 1 {
                        0.01
                    } else {
                        0.01 + (0.99 - 0.01) * g as f64 / (self.beta_mu.ncols() - 1) as f64
                    };
                }
                mu
            }
        };
        let beta_sum_prior = match beta_sum_prior {
            Some(x) => x,
            None => Array2::from_elem(beta_mu_prior.raw_dim(), 50.0),
        };
        self.theta_s1_prior = &beta_mu_prior * &beta_sum_prior;
        self.theta_s2_prior = (&beta_mu_prior * -1.0 + 1.0) * &beta_sum_prior;
        self.id_prior = match id_prior {
            Some(x) => vireo_base::normalize(&x, Some(1))?,
            None => Array2::from_elem(
                (self.id_prob.shape()[0], self.id_prob.shape()[1]),
                1.0 / self.id_prob.shape()[1] as f64,
            ),
        };
        let min_gp = min_gp.unwrap_or(0.00001);
        self.gt_prior = match gt_prior {
            Some(mut x) => {
                x.mapv_inplace(|v| v.max(min_gp).min(1.0 - min_gp));
                vireo_base::normalize(&x, Some(2))?
            }
            None => {
                let n_gt = self.gt_prob.shape()[2];
                Array3::from_elem(
                    (self.gt_prob.shape()[0], self.gt_prob.shape()[1], n_gt),
                    1.0 / n_gt as f64,
                )
            }
        };
        Some(())
    }

    /// Beta concentration1 parameter for theta's posterior (`beta_mu * beta_sum`).
    pub fn theta_s1(&self) -> Option<Array2<f64>> {
        Some(&self.beta_mu * &self.beta_sum)
    }

    /// Beta concentration2 parameter for theta's posterior (`(1 - beta_mu) * beta_sum`).
    pub fn theta_s2(&self) -> Option<Array2<f64>> {
        Some((&self.beta_mu * -1.0 + 1.0) * &self.beta_sum)
    }

    /// Digamma of the Beta concentration1 parameter, with an extra axis inserted at position 1.
    pub fn digamma1_(&self) -> Option<Array3<f64>> {
        self.theta_s1()
            .map(|x| x.mapv(digamma).insert_axis(Axis(1)))
    }

    /// Digamma of the Beta concentration2 parameter, with an extra axis inserted at position 1.
    pub fn digamma2_(&self) -> Option<Array3<f64>> {
        self.theta_s2()
            .map(|x| x.mapv(digamma).insert_axis(Axis(1)))
    }

    /// Digamma of the Beta concentration sum (`theta_s1 + theta_s2`), with an extra axis at position 1.
    pub fn digammas_(&self) -> Option<Array3<f64>> {
        match (self.theta_s1(), self.theta_s2()) {
            (Some(a), Some(b)) => Some((&a + &b).mapv(digamma).insert_axis(Axis(1))),
            _ => None,
        }
    }

    /// Coordinate-ascent update for theta's posterior parameters (`beta_mu`, `beta_sum`).
    ///
    /// `ad` carries alternative-allele counts and `dp` total depths; they must share shape.
    pub fn update_theta_size(&mut self, ad: &Array2<f64>, dp: &Array2<f64>) -> Option<()> {
        if ad.raw_dim() != dp.raw_dim() {
            return None;
        }
        let id_prob = self.id_prob.clone();
        let gt_prob = self.gt_prob.clone();
        let s1_gt = ad.dot(&id_prob);
        let s2_gt = (dp - ad).dot(&id_prob);
        let mut theta_s1 = self.theta_s1_prior.clone();
        let mut theta_s2 = self.theta_s2_prior.clone();
        let ase = self.ase_mode;
        for g in 0..gt_prob.shape()[2] {
            if ase {
                for v in 0..gt_prob.shape()[0] {
                    let mut a = 0.0;
                    let mut b = 0.0;
                    for d in 0..gt_prob.shape()[1] {
                        a += s1_gt[[v, d]] * gt_prob[[v, d, g]];
                        b += s2_gt[[v, d]] * gt_prob[[v, d, g]];
                    }
                    theta_s1[[v, g]] += a;
                    theta_s2[[v, g]] += b;
                }
            } else {
                let mut a = 0.0;
                let mut b = 0.0;
                for v in 0..gt_prob.shape()[0] {
                    for d in 0..gt_prob.shape()[1] {
                        a += s1_gt[[v, d]] * gt_prob[[v, d, g]];
                        b += s2_gt[[v, d]] * gt_prob[[v, d, g]];
                    }
                }
                theta_s1[[0, g]] += a;
                theta_s2[[0, g]] += b;
            }
        }
        self.beta_mu = &theta_s1 / (&theta_s1 + &theta_s2);
        if !self.fix_beta_sum {
            self.beta_sum = theta_s1 + theta_s2;
        }
        Some(())
    }

    /// Coordinate-ascent update for the cell-to-donor assignment probability `id_prob`.
    ///
    /// Returns the per-cell, per-donor log-likelihood matrix used in the update.
    pub fn update_ID_prob(&mut self, ad: &Array2<f64>, dp: &Array2<f64>) -> Option<Array2<f64>> {
        if ad.raw_dim() != dp.raw_dim() {
            return None;
        }
        let bd = dp - ad;
        let gt_prob = self.gt_prob.clone();
        let d1 = self.digamma1_()?;
        let d2 = self.digamma2_()?;
        let ds = self.digammas_()?;
        let mut log_lik = Array2::<f64>::zeros((ad.ncols(), gt_prob.shape()[1]));
        for g in 0..gt_prob.shape()[2] {
            let mut weighted1 = Array2::<f64>::zeros((gt_prob.shape()[0], gt_prob.shape()[1]));
            let mut weighted2 = weighted1.clone();
            let mut weighteds = weighted1.clone();
            for v in 0..gt_prob.shape()[0] {
                let theta_row = if d1.shape()[0] == 1 { 0 } else { v };
                for d in 0..gt_prob.shape()[1] {
                    weighted1[[v, d]] = gt_prob[[v, d, g]] * d1[[theta_row, 0, g]];
                    weighted2[[v, d]] = gt_prob[[v, d, g]] * d2[[theta_row, 0, g]];
                    weighteds[[v, d]] = gt_prob[[v, d, g]] * ds[[theta_row, 0, g]];
                }
            }
            log_lik =
                log_lik + ad.t().dot(&weighted1) + bd.t().dot(&weighted2) - dp.t().dot(&weighteds);
        }
        let log_lik_prior = &log_lik + &self.id_prior.mapv(f64::ln);
        let amplified = vireo_base::loglik_amplify(&log_lik_prior, None)?;
        self.id_prob = vireo_base::normalize(&amplified.mapv(f64::exp), None)?;
        Some(log_lik)
    }

    /// Coordinate-ascent update for the per-variant per-donor genotype probability `gt_prob`.
    pub fn update_GT_prob(&mut self, ad: &Array2<f64>, dp: &Array2<f64>) -> Option<()> {
        if ad.raw_dim() != dp.raw_dim() {
            return None;
        }
        let id_prob = self.id_prob.clone();
        let gt_prior = self.gt_prior.clone();
        let s1_gt = ad.dot(&id_prob);
        let ss_gt = dp.dot(&id_prob);
        let s2_gt = &ss_gt - &s1_gt;
        let d1 = self.digamma1_()?;
        let d2 = self.digamma2_()?;
        let ds = self.digammas_()?;
        let mut log_lik = Array3::<f64>::zeros(gt_prior.raw_dim());
        for v in 0..gt_prior.shape()[0] {
            let theta_row = if d1.shape()[0] == 1 { 0 } else { v };
            for d in 0..gt_prior.shape()[1] {
                for g in 0..gt_prior.shape()[2] {
                    log_lik[[v, d, g]] = s1_gt[[v, d]] * d1[[theta_row, 0, g]]
                        + s2_gt[[v, d]] * d2[[theta_row, 0, g]]
                        - ss_gt[[v, d]] * ds[[theta_row, 0, g]];
                }
            }
        }
        let log_lik_prior = &log_lik + &gt_prior.mapv(f64::ln);
        let amplified = vireo_base::loglik_amplify(&log_lik_prior, None)?;
        self.gt_prob = vireo_base::normalize(&amplified.mapv(f64::exp), None)?;
        Some(())
    }

    /// Compute the variational evidence lower bound (ELBO) with the current parameters.
    ///
    /// `log_lik_id` is the per-cell, per-donor log-likelihood matrix produced by
    /// [`Vireo::update_ID_prob`].
    pub fn get_ELBO(&self, log_lik_id: &Array2<f64>) -> Option<f64> {
        let lb_p = (log_lik_id * &self.id_prob).sum();
        let kl_id: f64 = self
            .id_prob
            .iter()
            .zip(self.id_prior.iter())
            .filter(|(p, q)| **p > 0.0 && **q > 0.0)
            .map(|(p, q)| p * (p / q).ln())
            .sum();
        let kl_gt: f64 = self
            .gt_prob
            .iter()
            .zip(self.gt_prior.iter())
            .filter(|(p, q)| **p > 0.0 && **q > 0.0)
            .map(|(p, q)| p * (p / q).ln())
            .sum();
        let theta_s1 = self.theta_s1()?;
        let theta_s2 = self.theta_s2()?;
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
        Some(lb_p - kl_id - kl_gt - kl_theta)
    }

    /// Run coordinate-ascent variational Bayes for at most `max_iter` iterations.
    ///
    /// Stops early once the ELBO change between successive iterations falls below
    /// `epsilon_conv` (after at least `min_iter` iterations). `delay_fit_theta`
    /// suppresses theta updates for the first N iterations. Returns the ELBO trace
    /// (one entry per iteration actually run).
    pub fn _fit_VB(
        &mut self,
        ad: &Array2<f64>,
        dp: &Array2<f64>,
        max_iter: usize,
        min_iter: usize,
        epsilon_conv: f64,
        delay_fit_theta: usize,
        _verbose: bool,
    ) -> Option<Vec<f64>> {
        let mut elbo = vec![0.0; max_iter];
        let mut last = 0usize;
        for it in 0..max_iter {
            if self.learn_theta && it >= delay_fit_theta {
                self.update_theta_size(ad, dp)?;
            }
            if self.learn_gt {
                self.update_GT_prob(ad, dp)?;
            }
            let log_lik_id = self.update_ID_prob(ad, dp)?;
            elbo[it] = self.get_ELBO(&log_lik_id)?;
            last = it;
            if it > min_iter
                && elbo[it] >= elbo[it - 1] - 1e-6
                && elbo[it] - elbo[it - 1] < epsilon_conv
            {
                break;
            }
        }
        Some(elbo[..=last].to_vec())
    }

    /// Fit the Vireo model with coordinate ascent and append the resulting ELBO
    /// trace (corrected with the binomial coefficient term) to `self.elbo_`.
    ///
    /// # Arguments
    /// * `ad` - Count matrix for the alternative allele (`n_var` x `n_cell`).
    /// * `dp` - Count matrix for total depths (alternative + reference).
    /// * `max_iter` - Maximum number of variational iterations.
    /// * `min_iter` - Minimum number of iterations before convergence checks fire.
    /// * `epsilon_conv` - Convergence threshold on the ELBO increment (default `1e-2`).
    /// * `delay_fit_theta` - Number of steps to delay updating theta. Useful when there
    ///   is a strong prior on the allelic ratio.
    /// * `verbose` - Whether to print log information.
    pub fn fit(
        &mut self,
        ad: &Array2<f64>,
        dp: &Array2<f64>,
        max_iter: usize,
        min_iter: usize,
        epsilon_conv: Option<f64>,
        delay_fit_theta: usize,
        verbose: bool,
        _n_inits: Option<usize>,
        _nproc: usize,
    ) -> Option<()> {
        let epsilon_conv = epsilon_conv.unwrap_or(1e-2);
        let mut elbo = self._fit_VB(
            ad,
            dp,
            max_iter,
            min_iter,
            epsilon_conv,
            delay_fit_theta,
            verbose,
        )?;
        let binom_coeff = vireo_base::get_binom_coeff(ad, dp, 700.0)
            .iter()
            .sum::<f64>();
        for v in &mut elbo {
            *v += binom_coeff;
        }
        self.elbo_.extend(elbo);
        Some(())
    }
}
