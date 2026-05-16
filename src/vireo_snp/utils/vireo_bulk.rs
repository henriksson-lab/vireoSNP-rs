//! Identification of donor abundance in a multiplexed bulk sample.

use ndarray::{Array1, Array3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Estimator of donor abundance in a multiplexed bulk sample.
///
/// Inferred variables:
/// - `psi`: fractional abundance of each donor in the mixture (length `n_donor`).
/// - `theta`: alternative allele rate per genotype category (length `n_gt`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VireoBulk {
    pub log_lik: f64,
    pub log_lik_all: Vec<f64>,
    pub n_gt: usize,
    pub n_donor: usize,
    pub psi: Array1<f64>,
    pub theta: Array1<f64>,
}

impl VireoBulk {
    /// Initialise the model with `n_donor` donors and `n_gt` genotype categories.
    ///
    /// `psi` is drawn from a Dirichlet-like distribution unless `psi_init` is provided.
    /// `theta` defaults to `[0.01, 0.5, 0.99]` when `n_gt == 3`, otherwise it is
    /// randomly initialised, unless `theta_init` is provided. Returns `None` if
    /// `psi_init` or `theta_init` have lengths that do not match `n_donor` / `n_gt`.
    pub fn __init__(
        &mut self,
        n_donor: usize,
        n_gt: usize,
        psi_init: Option<Array1<f64>>,
        theta_init: Option<Array1<f64>>,
    ) -> Option<()> {
        self.n_gt = n_gt;
        self.n_donor = n_donor;
        let mut rng = StdRng::seed_from_u64(0);
        let mut psi = Array1::<f64>::zeros(n_donor);
        for v in psi.iter_mut() {
            *v = -rng.gen::<f64>().ln();
        }
        let psi_sum = psi.sum();
        psi /= psi_sum;
        self.psi = match psi_init {
            Some(x) if x.len() == n_donor => x,
            None => psi,
            _ => return None,
        };
        self.theta = match theta_init {
            Some(x) if x.len() == n_gt => x,
            None if n_gt == 3 => Array1::from(vec![0.01, 0.5, 0.99]),
            None => {
                let mut theta = Array1::<f64>::zeros(n_gt);
                for v in theta.iter_mut() {
                    *v = rng.gen::<f64>();
                }
                theta
            }
            _ => return None,
        };
        Some(())
    }

    /// Fit `psi` and `theta` using the Expectation-Maximisation algorithm.
    ///
    /// # Arguments
    /// * `ad` - per-variant alternative allele counts (length `n_variant`).
    /// * `dp` - per-variant total depths over both alleles (length `n_variant`).
    /// * `gt_prob` - per-genotype probability tensor of shape
    ///   `(n_variant, n_donor, n_gt)`.
    /// * `max_iter` / `min_iter` - upper and lower bounds on EM iterations.
    /// * `epsilon_conv` - convergence threshold on the log-likelihood increment.
    /// * `learn_theta` - if `true`, also update `theta`; otherwise keep the
    ///   initial value.
    /// * `delay_fit_theta` - number of initial iterations during which `theta`
    ///   is not updated.
    ///
    /// Returns `None` if input shapes are inconsistent.
    pub fn fit(
        &mut self,
        ad: &Array1<f64>,
        dp: &Array1<f64>,
        gt_prob: &Array3<f64>,
        max_iter: usize,
        min_iter: usize,
        epsilon_conv: f64,
        learn_theta: bool,
        delay_fit_theta: usize,
        _model: Option<&str>,
        _verbose: bool,
    ) -> Option<()> {
        if ad.len() != dp.len() || gt_prob.shape()[0] != ad.len() {
            return None;
        }
        let bd = dp - ad;
        let mut psi = self.psi.clone();
        let mut theta = self.theta.clone();
        let mut log_lik = vec![0.0; max_iter];
        let mut last = 0usize;
        for it in 0..max_iter {
            let mut theta_mat =
                ndarray::Array2::<f64>::zeros((gt_prob.shape()[0], gt_prob.shape()[1]));
            for v in 0..gt_prob.shape()[0] {
                for d in 0..gt_prob.shape()[1] {
                    for g in 0..gt_prob.shape()[2] {
                        theta_mat[[v, d]] += gt_prob[[v, d, g]] * theta[g];
                    }
                }
            }
            let mut z1 = theta_mat.clone();
            let mut z0 = theta_mat.mapv(|v| 1.0 - v);
            for v in 0..theta_mat.nrows() {
                let mut s1 = 0.0;
                let mut s0 = 0.0;
                for d in 0..theta_mat.ncols() {
                    z1[[v, d]] *= psi[d];
                    z0[[v, d]] *= psi[d];
                    s1 += z1[[v, d]];
                    s0 += z0[[v, d]];
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
            if learn_theta && it >= delay_fit_theta {
                let mut theta_s1 = Array1::<f64>::zeros(theta.len());
                let mut theta_s2 = Array1::<f64>::zeros(theta.len());
                for g in 0..theta.len() {
                    for v in 0..ad.len() {
                        let mut gt_z1 = 0.0;
                        let mut gt_z0 = 0.0;
                        for d in 0..psi.len() {
                            gt_z1 += gt_prob[[v, d, g]] * z1[[v, d]];
                            gt_z0 += gt_prob[[v, d, g]] * z0[[v, d]];
                        }
                        theta_s1[g] += ad[v] * gt_z1;
                        theta_s2[g] += bd[v] * gt_z0;
                    }
                }
                theta = &theta_s1 / (&theta_s1 + &theta_s2);
            }
            let mut theta_vct = Array1::<f64>::zeros(ad.len());
            for v in 0..ad.len() {
                for d in 0..psi.len() {
                    let mut donor_theta = 0.0;
                    for g in 0..theta.len() {
                        donor_theta += gt_prob[[v, d, g]] * theta[g];
                    }
                    theta_vct[v] += donor_theta * psi[d];
                }
            }
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
        self.psi = psi;
        self.theta = theta;
        self.log_lik = log_lik[last];
        self.log_lik_all = log_lik[..=last].to_vec();
        Some(())
    }

    /// Likelihood-ratio test of the fitted `psi` against a null hypothesis.
    ///
    /// Delegates to [`LikRatio_test`] using the model's current `psi` and
    /// `theta`. `ad`, `dp`, and `gt_prob` should match those used in [`fit`].
    pub fn LR_test(
        &self,
        psi_null: &Array1<f64>,
        ad: &Array1<f64>,
        dp: &Array1<f64>,
        gt_prob: &ndarray::Array3<f64>,
        log: bool,
    ) -> Option<(f64, f64)> {
        LikRatio_test(&self.psi, psi_null, ad, dp, gt_prob, &self.theta, log)
    }
}

/// Likelihood-ratio test for a `psi` vector against a null hypothesis.
///
/// `ad`, `dp`, and `gt_prob` should be the same arrays used to fit the model.
///
/// # Arguments
/// * `psi` - alternative-hypothesis donor abundances (length `n_donor`).
/// * `psi_null` - null-hypothesis donor abundances (length `n_donor`).
/// * `ad` - alternative-allele counts per variant.
/// * `dp` - total depths per variant.
/// * `gt_prob` - genotype probability tensor of shape
///   `(n_variant, n_donor, n_gt)`.
/// * `theta` - alternative-allele rate per genotype category.
/// * `log` - if `true`, return the p-value on the log scale.
///
/// # Returns
/// `Some((statistic, pvalue))` with the chi-square statistic and single-tailed
/// p-value, or `None` if input shapes are inconsistent or the chi-square
/// distribution cannot be constructed.
pub fn LikRatio_test(
    psi: &Array1<f64>,
    psi_null: &Array1<f64>,
    ad: &Array1<f64>,
    dp: &Array1<f64>,
    gt_prob: &ndarray::Array3<f64>,
    theta: &Array1<f64>,
    log: bool,
) -> Option<(f64, f64)> {
    if psi.len() != psi_null.len()
        || ad.len() != dp.len()
        || gt_prob.shape()[0] != ad.len()
        || gt_prob.shape()[1] != psi.len()
        || gt_prob.shape()[2] != theta.len()
    {
        return None;
    }
    let bd = dp - ad;
    let mut log_lik = [0.0, 0.0];
    for (which, psi_use) in [&psi, &psi_null].iter().enumerate() {
        for v in 0..ad.len() {
            let mut theta_vct = 0.0;
            for d in 0..psi_use.len() {
                let mut donor_theta = 0.0;
                for g in 0..theta.len() {
                    donor_theta += gt_prob[[v, d, g]] * theta[g];
                }
                theta_vct += donor_theta * psi_use[d];
            }
            log_lik[which] += ad[v] * theta_vct.ln() + bd[v] * (1.0 - theta_vct).ln();
        }
    }
    let lr = 2.0 * (log_lik[0] - log_lik[1]);
    let df = psi_null.len().saturating_sub(1) as f64;
    let chi2 = match ChiSquared::new(df) {
        Ok(chi2) => chi2,
        Err(_) => return None,
    };
    let p = 1.0 - chi2.cdf(lr);
    let p = if log { p.ln() } else { p };
    Some((lr, p))
}
