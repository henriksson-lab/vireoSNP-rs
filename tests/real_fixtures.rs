use ndarray::{s, Array1, Array2, Array3};
use std::process::Command;
use vireosnp_rs::fit;
use vireosnp_rs::vireo_snp::plot::base_plot;
use vireosnp_rs::vireo_snp::utils::vireo_model::Vireo;
use vireosnp_rs::vireo_snp::utils::{
    base_utils,
    bmm_model::BinomMixtureVB,
    io_utils, variant_select, vcf_utils, vireo_base,
    vireo_bulk::{self, VireoBulk},
    vireo_doublet, vireo_wrap,
};
#[cfg(feature = "cli")]
use vireosnp_rs::vireo_snp::vireo::{self, VireoError};

fn dense(mat: &io_utils::CountMatrix) -> Array2<f64> {
    match mat {
        io_utils::CountMatrix::Dense(x) => x.clone(),
        io_utils::CountMatrix::DenseU32(x) => x.mapv(|v| v as f64),
        io_utils::CountMatrix::SparseCsc {
            nrows,
            ncols,
            indptr,
            indices,
            data,
        } => {
            let mut out = Array2::<f64>::zeros((*nrows, *ncols));
            for col in 0..*ncols {
                for p in indptr[col]..indptr[col + 1] {
                    out[[indices[p], col]] = data[p];
                }
            }
            out
        }
    }
}

fn run_python_probe(script: &str) -> Option<String> {
    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .env("PYTHONPATH", "vireo")
        .output()
        .ok()?;
    if !output.status.success() {
        eprintln!(
            "skipping Python parity probe: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

fn probe_line<'a>(output: &'a str, prefix: &str) -> &'a str {
    output
        .lines()
        .find_map(|line| line.strip_prefix(prefix))
        .unwrap_or_else(|| panic!("missing Python probe line {prefix:?} in:\n{output}"))
}

fn parse_probe_f64s(output: &str, prefix: &str) -> Vec<f64> {
    probe_line(output, prefix)
        .split(',')
        .filter(|x| !x.is_empty())
        .map(|x| x.parse::<f64>().unwrap())
        .collect()
}

fn assert_slice_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "value {idx} differs: actual={a:?} expected={e:?} tol={tol}"
        );
    }
}

#[test]
fn reads_cellsnp_fixture() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    );
    let dat = dat.unwrap();
    let samples = &dat.samples;
    let variants = &dat.variants;
    let ad = dense(dat.layers.get("AD").expect("missing AD"));
    let dp = dense(dat.layers.get("DP").expect("missing DP"));
    assert!(!samples.is_empty());
    assert!(!variants.is_empty());
    assert_eq!(ad.shape(), dp.shape());
    assert_eq!(ad.shape()[0], variants.len());
    assert_eq!(ad.shape()[1], samples.len());
}

#[test]
fn reads_representative_sparse_cellsnp_matrixmarket_layers() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    )
    .unwrap();
    let ad = dat.layers.get("AD").expect("missing AD");
    let dp = dat.layers.get("DP").expect("missing DP");
    for layer in [ad, dp] {
        match layer {
            io_utils::CountMatrix::SparseCsc {
                nrows, ncols, data, ..
            } => {
                assert_eq!((*nrows, *ncols), (dat.variants.len(), dat.samples.len()));
                let density = data.len() as f64 / (*nrows as f64 * *ncols as f64);
                assert!(
                    density < 0.05,
                    "expected sparse representative fixture, observed density {density}"
                );
            }
            _ => panic!("expected MatrixMarket layer to remain sparse before fitting"),
        }
    }
}

#[test]
fn computes_binomial_coefficients_on_real_count_slice() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    );
    let dat = dat.unwrap();
    let ad = dense(dat.layers.get("AD").expect("missing AD"));
    let dp = dense(dat.layers.get("DP").expect("missing DP"));
    let ad = ad.slice(s![0..10, 0..10]).to_owned();
    let dp = dp.slice(s![0..10, 0..10]).to_owned();
    let coeff = vireo_base::get_binom_coeff(&ad, &dp, 700.0);
    assert!(!coeff.is_empty());
    assert!(coeff.iter().all(|v| v.is_finite()));
}

#[test]
fn matches_python_binomial_coefficients_on_real_count_slice() {
    let Some(py) = run_python_probe(
        r#"
import numpy as np
from vireoSNP.utils.io_utils import read_cellSNP
from vireoSNP.utils.vireo_base import get_binom_coeff
dat = read_cellSNP('vireo/data/cellSNP_mat', layers=['AD', 'DP'])
coeff = np.asarray(get_binom_coeff(dat['AD'][:100, :100], dat['DP'][:100, :100], max_val=700)).reshape(-1)
print('coeff_len\t%d' % coeff.size)
print('coeff_sum\t%.17g' % float(np.sum(coeff)))
print('coeff_max\t%.17g' % float(np.max(coeff) if coeff.size else 0.0))
"#,
    ) else {
        return;
    };

    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    )
    .unwrap();
    let ad = dense(dat.layers.get("AD").expect("missing AD"));
    let dp = dense(dat.layers.get("DP").expect("missing DP"));
    let ad = ad.slice(s![0..100, 0..100]).to_owned();
    let dp = dp.slice(s![0..100, 0..100]).to_owned();
    let coeff = vireo_base::get_binom_coeff(&ad, &dp, 700.0);

    let py_len = probe_line(&py, "coeff_len\t").parse::<usize>().unwrap();
    let py_sum = probe_line(&py, "coeff_sum\t").parse::<f64>().unwrap();
    let py_max = probe_line(&py, "coeff_max\t").parse::<f64>().unwrap();
    assert_eq!(coeff.len(), py_len);
    assert!((coeff.iter().sum::<f64>() - py_sum).abs() < 1e-9);
    assert!((coeff.iter().copied().fold(0.0, f64::max) - py_max).abs() < 1e-9);
}

#[test]
fn amplifies_log_likelihood_array_without_pyvalue() {
    let x = Array2::<f64>::from_shape_vec(
        (3, 3),
        vec![-5.0, -2.0, -3.0, 10.0, 7.0, 8.0, 0.5, 0.25, 0.0],
    )
    .unwrap();
    let amplified = vireo_base::loglik_amplify(&x, Some(1)).unwrap();
    assert_eq!(amplified[[0, 1]], 0.0);
    assert_eq!(amplified[[1, 0]], 0.0);
    assert_eq!(amplified[[2, 0]], 0.0);
    assert!(amplified.iter().all(|v| *v <= 0.0));
}

#[test]
fn normalizes_arrays_without_pyvalue() {
    let x = Array2::<f64>::from_shape_vec((2, 3), vec![1.0, 1.0, 2.0, 2.0, 3.0, 5.0]).unwrap();
    let normalized = vireo_base::normalize(&x, Some(1)).unwrap();
    assert!((normalized.row(0).sum() - 1.0).abs() < 1e-12);
    assert!((normalized.row(1).sum() - 1.0).abs() < 1e-12);
    let normalized = vireo_base::tensor_normalize(&x, Some(1)).unwrap();
    assert!((normalized.row(0).sum() - 1.0).abs() < 1e-12);
}

#[test]
fn computes_logbincoeff_without_pyvalue() {
    let n = Array2::<f64>::from_shape_vec((1, 3), vec![4.0, 5.0, 6.0]).unwrap();
    let k = Array2::<f64>::from_shape_vec((1, 3), vec![2.0, 2.0, 3.0]).unwrap();
    let out = vireo_base::logbincoeff(&n, &k).unwrap();
    assert_eq!(out.shape(), &[1, 3]);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn selects_genotype_barcode_variants_without_pyvalue() {
    let gt = Array2::<f64>::from_shape_vec(
        (4, 3),
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0, 0.0],
    )
    .unwrap();
    let counts = vec![10.0, 20.0, 30.0, 40.0];
    let (entropy, barcode, variants) =
        variant_select::variant_select(&gt, Some(&counts), 0).unwrap();
    assert!(entropy > 0.0);
    assert_eq!(barcode.len(), 3);
    assert!(!variants.is_empty());
}

#[test]
fn reads_vartrix_style_matrixmarket_with_optional_vcf() {
    let dat = io_utils::read_vartrix(
        "vireo/data/cellSNP_mat/cellSNP.tag.AD.mtx",
        "vireo/data/cellSNP_mat/cellSNP.tag.OTH.mtx",
        "vireo/data/cellSNP_mat/cellSNP.samples.tsv",
        Some("vireo/data/cellSNP_mat/cellSNP.base.vcf.gz"),
    );
    let dat = dat.unwrap();
    let samples = &dat.samples;
    let variants = &dat.variants;
    let ad = dense(dat.layers.get("AD").expect("missing AD"));
    let dp = dense(dat.layers.get("DP").expect("missing DP"));
    assert_eq!(ad.shape(), dp.shape());
    assert_eq!(ad.shape()[0], variants.len());
    assert_eq!(ad.shape()[1], samples.len());
    assert!(dp.sum() >= ad.sum());
}

#[test]
fn loads_cells_vcf_sparse_and_materializes_read_layers() {
    let vcf =
        vcf_utils::load_VCF("vireo/data/cells.cellSNP.vcf.gz", false, true, true, None).unwrap();
    let variants = &vcf.variants;
    let samples = &vcf.samples;
    let geno_info = vcf.geno_info.as_ref().expect("missing GenoINFO");
    assert!(geno_info.i64_vecs.contains_key("indices"));
    assert!(geno_info.i64_vecs.contains_key("indptr"));
    assert!(geno_info.i64_vecs.contains_key("shape"));

    let dense = vcf_utils::read_sparse_GeneINFO(
        geno_info,
        Some(&["AD".to_string(), "DP".to_string()]),
        Some(&[-1, -1]),
    )
    .unwrap();
    let ad = dense.get("AD").unwrap();
    let dp = dense.get("DP").unwrap();
    assert_eq!(ad.shape(), &[variants.len(), samples.len()]);
    assert_eq!(dp.shape(), &[variants.len(), samples.len()]);
    assert!(dp.sum() >= ad.sum());
}

#[test]
fn parses_real_donor_vcf_genotypes_to_probabilities() {
    let donor_vcf = vcf_utils::load_VCF(
        "vireo/data/donors.cellSNP.vcf.gz",
        true,
        true,
        false,
        Some(&["GT".to_string(), "PL".to_string()]),
    )
    .unwrap();
    let variants = &donor_vcf.variants;
    let samples = &donor_vcf.samples;
    let geno_info = donor_vcf.geno_info.as_ref().expect("missing GenoINFO");
    let gt = geno_info.string_matrices.get("GT").expect("missing GT");
    let gpb = vcf_utils::parse_donor_GPb(gt, "GT", 0.0).unwrap();
    assert_eq!(gpb.shape(), &[variants.len(), samples.len(), 3]);
    for lane in gpb.outer_iter().take(10) {
        for donor in lane.outer_iter() {
            assert!((donor.sum() - 1.0).abs() < 1e-12);
        }
    }
}

#[test]
fn makes_and_writes_vcf_genoinfo_roundtrip() {
    let mut vcf = vcf_utils::load_VCF(
        "vireo/data/cellSNP_mat/cellSNP.base.vcf.gz",
        false,
        false,
        true,
        None,
    )
    .unwrap();
    vcf.variants.truncate(5);
    for values in vcf.fixed_info.values_mut() {
        values.truncate(5);
    }
    vcf.samples = vec!["donor0".to_string(), "donor1".to_string()];

    let mut gt_prob = Array3::<f64>::zeros((5, 2, 3));
    for i in 0..5 {
        gt_prob[[i, 0, i % 3]] = 1.0;
        gt_prob[[i, 1, (i + 1) % 3]] = 1.0;
    }
    let ad_reads = Array2::<f64>::from_shape_fn((5, 2), |(i, j)| (i + j + 1) as f64);
    let dp_reads = Array2::<f64>::from_shape_fn((5, 2), |(i, j)| (i + j + 6) as f64);
    let geno_info = vcf_utils::GenoINFO_maker(&gt_prob, &ad_reads, &dp_reads).unwrap();
    assert_eq!(geno_info.get("GT").unwrap().len(), 5);
    assert_eq!(geno_info.get("PL").unwrap()[0].len(), 2);
    vcf.geno_info = Some(vcf_utils::VcfGenoInfo {
        string_matrices: geno_info,
        ..Default::default()
    });

    let out_file = std::env::temp_dir().join(format!(
        "vireo-rs-write-vcf-{}-{}.vcf.gz",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let wrote = vcf_utils::write_VCF(&out_file.to_string_lossy(), &vcf, None);
    assert_eq!(wrote, Some(()));

    let loaded = vcf_utils::load_VCF(
        &out_file.to_string_lossy(),
        false,
        true,
        false,
        Some(&["GT".to_string(), "AD".to_string(), "DP".to_string()]),
    )
    .unwrap();
    let geno_info = loaded
        .geno_info
        .as_ref()
        .expect("missing reloaded GenoINFO");
    assert_eq!(
        loaded.samples,
        vec!["donor0".to_string(), "donor1".to_string()]
    );
    assert_eq!(loaded.variants.len(), 5);
    assert!(matches!(
        geno_info.string_matrices.get("GT"),
        Some(rows) if rows.len() == 5 && rows[0].len() == 2
    ));

    let h5_file = std::env::temp_dir().join(format!(
        "vireo-rs-write-vcf-{}-{}.h5",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    assert_eq!(
        vcf_utils::write_VCF_to_hdf5(&vcf, &h5_file.to_string_lossy()),
        Some(())
    );
    let h5 = hdf5::File::open(&h5_file).unwrap();
    assert_eq!(h5.dataset("variants").unwrap().shape(), vec![5]);
    assert_eq!(h5.dataset("samples").unwrap().shape(), vec![2]);
    assert_eq!(h5.dataset("GenoINFO/GT").unwrap().shape(), vec![10]);
    assert_eq!(
        h5.dataset("GenoINFO/GT_shape")
            .unwrap()
            .read_1d::<i64>()
            .unwrap()
            .to_vec(),
        vec![5, 2]
    );
    std::fs::remove_file(out_file).unwrap();
    std::fs::remove_file(h5_file).unwrap();
}

#[test]
fn matches_cellsnp_fixture_to_real_donor_vcf() {
    let cell_dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    );
    let cell_dat = cell_dat.unwrap();
    let donor_vcf = vcf_utils::load_VCF(
        "vireo/data/donors.two.cellSNP.vcf.gz",
        false,
        true,
        false,
        Some(&["GT".to_string(), "PL".to_string()]),
    )
    .unwrap();
    let (cell_dat, donor_vcf) = io_utils::match_donor_VCF(cell_dat, donor_vcf).unwrap();
    let cell_variants = &cell_dat.variants;
    let donor_variants = &donor_vcf.variants;
    let ad = dense(cell_dat.layers.get("AD").expect("missing matched AD"));
    assert!(!cell_variants.is_empty());
    assert_eq!(cell_variants.len(), donor_variants.len());
    assert_eq!(ad.shape()[0], cell_variants.len());
}

#[test]
fn matches_snp_ids_with_chr_prefix_fallback() {
    let left = vec!["1_10_A_G".to_string(), "2_20_C_T".to_string()];
    let right = vec!["chr2_20_C_T".to_string(), "chr1_10_A_G".to_string()];
    assert_eq!(vcf_utils::match_SNPs(&left, &right), vec![Some(1), Some(0)]);
}

#[test]
fn matches_real_donor_vcf_samples() {
    let matched = vcf_utils::match_VCF_samples(
        "vireo/data/donors.cellSNP.vcf.gz",
        "vireo/data/donors.two.cellSNP.vcf.gz",
        "GT",
        "GT",
    )
    .unwrap();
    assert!(matched.matched_n_var > 0);
    assert_eq!(matched.matched_donors1.len(), matched.matched_donors2.len());
    assert_eq!(
        matched.matched_gpb_diff.shape(),
        &[matched.matched_donors1.len(), matched.matched_donors2.len()]
    );
}

#[test]
fn matches_python_real_donor_vcf_sample_alignment() {
    let Some(py) = run_python_probe(
        r#"
import numpy as np
from contextlib import redirect_stdout
from vireoSNP.utils.vcf_utils import match_VCF_samples
with open('/dev/null', 'w') as devnull, redirect_stdout(devnull):
    m = match_VCF_samples(
        'vireo/data/donors.cellSNP.vcf.gz',
        'vireo/data/donors.two.cellSNP.vcf.gz',
        'GT',
        'GT',
    )
print('matched_n_var\t%d' % m['matched_n_var'])
print('donors1\t%s' % ','.join(m['matched_donors1']))
print('donors2\t%s' % ','.join(m['matched_donors2']))
print('diff\t%s' % ','.join('%.17g' % x for x in m['matched_GPb_diff'].reshape(-1)))
"#,
    ) else {
        return;
    };

    let matched = vcf_utils::match_VCF_samples(
        "vireo/data/donors.cellSNP.vcf.gz",
        "vireo/data/donors.two.cellSNP.vcf.gz",
        "GT",
        "GT",
    )
    .unwrap();
    assert_eq!(
        matched.matched_n_var,
        probe_line(&py, "matched_n_var\t").parse::<usize>().unwrap()
    );
    assert_eq!(
        matched.matched_donors1.join(","),
        probe_line(&py, "donors1\t")
    );
    assert_eq!(
        matched.matched_donors2.join(","),
        probe_line(&py, "donors2\t")
    );
    let expected = parse_probe_f64s(&py, "diff\t");
    let actual: Vec<f64> = matched.matched_gpb_diff.iter().copied().collect();
    assert_slice_close(&actual, &expected, 1e-12);
}

#[test]
fn fits_vireo_model_on_real_cellsnp_slice() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    );
    let dat = dat.unwrap();
    let ad = dense(dat.layers.get("AD").expect("missing AD"));
    let dp = dense(dat.layers.get("DP").expect("missing DP"));
    let ad = ad.slice(s![0..80, 0..60]).to_owned();
    let dp = dp.slice(s![0..80, 0..60]).to_owned();

    let mut model = Vireo::default();
    model
        .__init__(
            60, 80, 2, 3, true, true, false, false, None, None, None, None,
        )
        .unwrap();
    model
        .fit(&ad, &dp, 4, 1, Some(1e-2), 1, false, None, 1)
        .unwrap();

    let id_prob = &model.id_prob;
    let gt_prob = &model.gt_prob;
    let beta_mu = &model.beta_mu;
    let elbo = &model.elbo_;
    assert_eq!(id_prob.shape(), &[60, 2]);
    assert_eq!(gt_prob.shape(), &[80, 2, 3]);
    assert_eq!(beta_mu.shape(), &[1, 3]);
    assert!(!elbo.is_empty());
    assert!(elbo.iter().all(|v| v.is_finite()));
}

#[test]
fn vireo_initialization_uses_configured_seed() {
    let mut a = Vireo::default();
    a.set_rng_seed(11);
    a.__init__(4, 3, 2, 3, true, true, false, false, None, None, None, None)
        .unwrap();

    let mut b = Vireo::default();
    b.set_rng_seed(11);
    b.__init__(4, 3, 2, 3, true, true, false, false, None, None, None, None)
        .unwrap();

    let mut c = Vireo::default();
    c.set_rng_seed(12);
    c.__init__(4, 3, 2, 3, true, true, false, false, None, None, None, None)
        .unwrap();

    assert_eq!(a.id_prob, b.id_prob);
    assert_eq!(a.gt_prob, b.gt_prob);
    assert_ne!(a.id_prob, c.id_prob);
    assert_ne!(a.gt_prob, c.gt_prob);
}

#[test]
fn fits_binom_mixture_on_real_cellsnp_slice_without_pyvalue_updates() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    )
    .unwrap();
    let ad = dense(dat.layers.get("AD").expect("missing AD"));
    let dp = dense(dat.layers.get("DP").expect("missing DP"));
    let ad = ad.clone().slice(s![0..40, 0..30]).to_owned();
    let dp = dp.clone().slice(s![0..40, 0..30]).to_owned();
    let mut model = BinomMixtureVB::default();
    model.__init__(30, 40, 2, false, None, None, None).unwrap();
    model.fit(&ad, &dp, 2, 5, Some(3), 0).unwrap();
    let id_prob = &model.id_prob;
    let elbo = &model.elbo_iters;
    assert_eq!(id_prob.shape(), &[30, 2]);
    assert!(!elbo.is_empty());
    assert!(elbo.iter().all(|v| v.is_finite()));
}

#[test]
fn wraps_real_cellsnp_slice_and_writes_donor_outputs() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    );
    let dat = dat.unwrap();
    let samples = &dat.samples;
    let ad = dense(dat.layers.get("AD").expect("missing AD"));
    let dp = dense(dat.layers.get("DP").expect("missing DP"));
    let ad = ad.slice(s![0..50, 0..24]).to_owned();
    let dp = dp.slice(s![0..50, 0..24]).to_owned();

    let res = vireo_wrap::vireo_wrap(
        &ad,
        &dp,
        None,
        Some(2),
        true,
        1,
        Some(0),
        false,
        3,
        1,
        0,
        Some("distance"),
        false,
        1,
        false,
        false,
        3,
    )
    .unwrap();
    let id_prob = &res.id_prob;
    let gt_prob = &res.gt_prob;
    let doublet_prob = &res.doublet_prob;
    assert_eq!(id_prob.shape(), &[24, 2]);
    assert_eq!(gt_prob.shape(), &[50, 2, 3]);
    assert_eq!(doublet_prob.shape(), &[24, 1]);

    let out_dir = std::env::temp_dir().join(format!(
        "vireo-rs-donor-output-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&out_dir).unwrap();
    let n_vars: Vec<f64> = dp
        .mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
        .sum_axis(ndarray::Axis(0))
        .iter()
        .copied()
        .collect();
    io_utils::write_donor_id(
        &out_dir.to_string_lossy(),
        &["donor0".to_string(), "donor1".to_string()],
        &samples[0..24],
        &n_vars,
        &res,
    )
    .unwrap();

    let donor_ids = std::fs::read_to_string(out_dir.join("donor_ids.tsv")).unwrap();
    let prob_singlet = std::fs::read_to_string(out_dir.join("prob_singlet.tsv")).unwrap();
    let prob_doublet = std::fs::read_to_string(out_dir.join("prob_doublet.tsv")).unwrap();
    assert!(donor_ids.starts_with(
        "cell\tdonor_id\tprob_max\tprob_doublet\tn_vars\tbest_singlet\tbest_doublet\tdoublet_logLikRatio\n"
    ));
    assert_eq!(donor_ids.lines().count(), 25);
    assert!(prob_singlet.starts_with("cell\tdonor0\tdonor1\n"));
    assert!(prob_doublet.starts_with("cell\tdonor0,donor1\n"));
    std::fs::remove_dir_all(out_dir).unwrap();
}

#[test]
fn matches_python_vireo_wrap_with_donor_prior_on_real_slice() {
    let Some(py) = run_python_probe(
        r#"
import numpy as np
from contextlib import redirect_stdout
from vireoSNP.utils.io_utils import match_donor_VCF, read_cellSNP
from vireoSNP.utils.vcf_utils import load_VCF, parse_donor_GPb
from vireoSNP.utils.vireo_wrap import vireo_wrap
with open('/dev/null', 'w') as devnull, redirect_stdout(devnull):
    cell = read_cellSNP('vireo/data/cellSNP_mat', layers=['AD', 'DP'])
    donor = load_VCF(
        'vireo/data/donors.cellSNP.vcf.gz',
        biallelic_only=True,
        sparse=False,
        format_list=['GT'],
    )
    cell, donor = match_donor_VCF(cell, donor)
    gt = parse_donor_GPb(donor['GenoINFO']['GT'], 'GT')
    res = vireo_wrap(
        cell['AD'][:100, :20],
        cell['DP'][:100, :20],
        GT_prior=gt[:100, :, :],
        n_donor=gt.shape[1],
        learn_GT=False,
        n_init=1,
        random_seed=3,
        check_doublet=False,
        max_iter_init=3,
        delay_fit_theta=1,
        n_extra_donor=0,
        extra_donor_mode='distance',
        check_ambient=False,
        nproc=1,
        n_GT=3,
    )
print('id_shape\t%d,%d' % res['ID_prob'].shape)
print('gt_shape\t%d,%d,%d' % res['GT_prob'].shape)
print('id\t%s' % ','.join('%.17g' % x for x in res['ID_prob'].reshape(-1)))
print('theta_mean\t%s' % ','.join('%.17g' % x for x in res['theta_mean'].reshape(-1)))
print('theta_sum\t%s' % ','.join('%.17g' % x for x in res['theta_sum'].reshape(-1)))
print('lb\t%.17g' % float(res['LB_doublet']))
"#,
    ) else {
        return;
    };

    let cell_dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    )
    .unwrap();
    let donor_tags = ["GT".to_string()];
    let donor_vcf = vcf_utils::load_VCF(
        "vireo/data/donors.cellSNP.vcf.gz",
        true,
        true,
        false,
        Some(&donor_tags),
    )
    .unwrap();
    let (cell_dat, donor_vcf) = io_utils::match_donor_VCF(cell_dat, donor_vcf).unwrap();
    let geno = donor_vcf
        .geno_info
        .as_ref()
        .unwrap()
        .string_matrices
        .get("GT")
        .unwrap();
    let gt = vcf_utils::parse_donor_GPb(geno, "GT", 0.0).unwrap();
    let ad = dense(cell_dat.layers.get("AD").expect("missing AD"))
        .slice(s![0..100, 0..20])
        .to_owned();
    let dp = dense(cell_dat.layers.get("DP").expect("missing DP"))
        .slice(s![0..100, 0..20])
        .to_owned();
    let gt = gt.slice(s![0..100, .., ..]).to_owned();
    let res = vireo_wrap::vireo_wrap(
        &ad,
        &dp,
        Some(&gt),
        Some(gt.shape()[1]),
        false,
        1,
        Some(3),
        false,
        3,
        1,
        0,
        Some("distance"),
        false,
        1,
        false,
        false,
        3,
    )
    .unwrap();

    let id_shape = format!("{},{}", res.id_prob.nrows(), res.id_prob.ncols());
    let gt_shape = format!(
        "{},{},{}",
        res.gt_prob.shape()[0],
        res.gt_prob.shape()[1],
        res.gt_prob.shape()[2]
    );
    assert_eq!(id_shape, probe_line(&py, "id_shape\t"));
    assert_eq!(gt_shape, probe_line(&py, "gt_shape\t"));
    assert_slice_close(
        res.id_prob.as_slice().unwrap(),
        &parse_probe_f64s(&py, "id\t"),
        1e-8,
    );
    assert_slice_close(
        res.theta_mean.as_slice().unwrap(),
        &parse_probe_f64s(&py, "theta_mean\t"),
        1e-8,
    );
    assert_slice_close(
        res.theta_sum.as_slice().unwrap(),
        &parse_probe_f64s(&py, "theta_sum\t"),
        1e-7,
    );
    let py_lb = probe_line(&py, "lb\t").parse::<f64>().unwrap();
    assert!((res.lb_doublet - py_lb).abs() < 1e-6);
}

#[test]
fn parses_gt_codes() {
    assert_eq!(vcf_utils::parse_GT_code("0/1", "GT"), Some([0.0, 1.0, 0.0]));
    assert_eq!(
        vcf_utils::parse_GT_code("./.", "GT"),
        Some([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    );
}

#[test]
fn scores_variant_elbo_gain_on_real_count_slice() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    );
    let dat = dat.unwrap();
    let ad = dense(dat.layers.get("AD").expect("expected AD"));
    let dp = dense(dat.layers.get("DP").expect("expected DP"));
    let ad = ad.slice(s![0..20, 0..12]).to_owned();
    let dp = dp.slice(s![0..20, 0..12]).to_owned();
    let mut id_prob = Array2::<f64>::zeros((12, 2));
    for i in 0..12 {
        id_prob[[i, i % 2]] = 1.0;
    }
    let gain = variant_select::variant_ELBO_gain(&id_prob, &ad, &dp, 0.5).unwrap();
    assert_eq!(gain.len(), 20);
    assert!(gain.iter().all(|v| v.is_finite()));
    assert!(gain.iter().any(|v| *v > 0.0));
}

#[test]
fn selects_donors_from_real_vireo_model_arrays_without_pyvalue() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    );
    let dat = dat.unwrap();
    let ad = dense(dat.layers.get("AD").expect("expected AD"));
    let dp = dense(dat.layers.get("DP").expect("expected DP"));
    let ad = ad.clone().slice(s![0..50, 0..24]).to_owned();
    let dp = dp.clone().slice(s![0..50, 0..24]).to_owned();

    let mut model = Vireo::default();
    model
        .__init__(
            24, 50, 3, 3, true, true, false, false, None, None, None, None,
        )
        .unwrap();
    model
        .fit(&ad, &dp, 4, 1, Some(1e-2), 1, false, None, 1)
        .unwrap();
    let gt_prob = model.gt_prob.clone();
    let id_prob = model.id_prob.clone();
    let selected = vireo_base::donor_select(&gt_prob, &id_prob, 2, Some("distance")).unwrap();
    assert_eq!(selected.dim(), (24, 2));
    assert!(selected.iter().all(|v| v.is_finite() && *v >= 1e-10));
}

#[test]
fn computes_beta_entropy_from_real_vireo_theta_arrays_without_pyvalue() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    );
    let dat = dat.unwrap();
    let ad = dense(dat.layers.get("AD").expect("expected AD"));
    let dp = dense(dat.layers.get("DP").expect("expected DP"));
    let ad = ad.clone().slice(s![0..50, 0..24]).to_owned();
    let dp = dp.clone().slice(s![0..50, 0..24]).to_owned();
    let mut model = Vireo::default();
    model
        .__init__(
            24, 50, 2, 3, true, true, false, false, None, None, None, None,
        )
        .unwrap();
    model
        .fit(&ad, &dp, 4, 1, Some(1e-2), 1, false, None, 1)
        .unwrap();
    let theta_s1 = model.theta_s1().expect("expected theta_s1");
    let theta_s2 = model.theta_s2().expect("expected theta_s2");
    let s1_prior = &model.theta_s1_prior;
    let s2_prior = &model.theta_s2_prior;
    let mut x = Array2::<f64>::zeros((theta_s1.len(), 2));
    let mut xp = Array2::<f64>::zeros((theta_s1.len(), 2));
    for (i, (((a, b), pa), pb)) in theta_s1
        .iter()
        .zip(theta_s2.iter())
        .zip(s1_prior.iter())
        .zip(s2_prior.iter())
        .enumerate()
    {
        x[[i, 0]] = *a;
        x[[i, 1]] = *b;
        xp[[i, 0]] = *pa;
        xp[[i, 1]] = *pb;
    }
    let entropy = vireo_base::beta_entropy(&x, Some(&xp), None).unwrap();
    let cross = vireo_base::_beta_cross_entropy(&x, &xp).unwrap();
    assert_eq!(entropy.len(), theta_s1.len());
    assert_eq!(cross.len(), theta_s1.len());
    assert!(entropy.iter().all(|v| v.is_finite()));
    assert!(cross.iter().all(|v| v.is_finite()));
}

#[test]
fn matches_real_variant_ids_without_pyvalue() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    );
    let dat = dat.unwrap();
    let variants = &dat.variants;
    let ref_ids = variants[0..10].to_vec();
    let new_ids = variants[3..13].to_vec();
    let matched = vireo_base::r#match(&ref_ids, &new_ids, true);
    assert_eq!(matched.len(), 10);
    assert_eq!(matched[0], None);
    assert_eq!(matched[3], Some(0));

    let x = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.9, 0.8, 0.7, 0.1]).unwrap();
    let z = Array2::from_shape_vec((2, 3), vec![0.9, 0.1, 0.2, 0.1, 0.8, 0.7]).unwrap();
    let (_, idx, delta) = vireo_base::optimal_match(&x, &z, Some(1), true).unwrap();
    assert_eq!(idx.len(), 3);
    assert_eq!(delta.unwrap().dim(), (3, 3));
    assert_eq!(vireo_base::greed_match(&x, &z, Some(1)).unwrap().len(), 3);
}

#[test]
fn optimal_match_uses_global_assignment_not_row_greedy() {
    let x = Array2::from_shape_vec((1, 3), vec![0.0, 2.0, 3.0]).unwrap();
    let z = Array2::from_shape_vec((1, 3), vec![2.0, 0.0, 3.0]).unwrap();
    let (idx0, idx1, delta) = vireo_base::optimal_match(&x, &z, Some(1), true).unwrap();
    assert_eq!(idx0, vec![0, 1, 2]);
    assert_eq!(idx1, vec![1, 0, 2]);
    assert_eq!(
        delta.unwrap(),
        Array2::from_shape_vec((3, 3), vec![2.0, 0.0, 3.0, 0.0, 2.0, 1.0, 1.0, 3.0, 0.0]).unwrap()
    );
}

#[test]
fn computes_confusion_matrix_without_pyvalue() {
    let ids1 = vec![
        "donor0".to_string(),
        "donor0".to_string(),
        "donor1".to_string(),
        "donor1".to_string(),
    ];
    let ids2 = vec![
        "A".to_string(),
        "A".to_string(),
        "A".to_string(),
        "B".to_string(),
    ];
    let (mat, rows, cols) = base_utils::get_confusion(&ids1, &ids2).unwrap();
    assert_eq!(rows, vec!["donor0".to_string(), "donor1".to_string()]);
    assert_eq!(cols, vec!["A".to_string(), "B".to_string()]);
    assert_eq!(mat.dim(), (2, 2));
    assert_eq!(mat[[0, 0]], 2.0);
    assert_eq!(mat[[1, 1]], 1.0);
}

#[test]
fn runs_doublet_leaf_transforms_without_pyvalue() {
    let beta_mu = Array2::from_shape_vec((1, 3), vec![0.05, 0.5, 0.95]).unwrap();
    let beta_sum = Array2::from_shape_vec((1, 3), vec![50.0, 50.0, 50.0]).unwrap();
    let (mu_both, sum_both) = vireo_doublet::add_doublet_theta(&beta_mu, &beta_sum).unwrap();
    assert_eq!(mu_both.dim(), (1, 6));
    assert_eq!(sum_both.dim(), (1, 6));

    let mut gt_prob = Array3::<f64>::zeros((4, 3, 3));
    for v in 0..4 {
        for d in 0..3 {
            gt_prob[[v, d, (v + d) % 3]] = 1.0;
        }
    }
    let gt_both = vireo_doublet::add_doublet_GT(&gt_prob).unwrap();
    assert_eq!(gt_both.dim(), (4, 6, 6));

    let ad = Array1::from(vec![1.0, 4.0, 8.0, 2.0]);
    let dp = Array1::from(vec![10.0, 10.0, 12.0, 8.0]);
    let theta = Array2::from_shape_vec(
        (4, 3),
        vec![0.1, 0.5, 0.9, 0.2, 0.6, 0.8, 0.7, 0.4, 0.2, 0.3, 0.5, 0.7],
    )
    .unwrap();
    let (psi, var, llr) = vireo_doublet::_fit_EM_ambient(
        &ad,
        &dp,
        &theta,
        None,
        Some(20),
        Some(3),
        None,
        true,
        false,
    )
    .unwrap();
    assert_eq!(psi.len(), 3);
    assert_eq!(var.len(), 3);
    assert!(psi.iter().all(|v| v.is_finite()));
    assert!(llr.is_finite());
}

#[test]
fn wraps_real_cellsnp_slice_with_ambient_outputs() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    )
    .unwrap();
    let ad = dense(dat.layers.get("AD").expect("missing AD"));
    let dp = dense(dat.layers.get("DP").expect("missing DP"));
    let ad = ad.slice(s![0..40, 0..6]).to_owned();
    let dp = dp.slice(s![0..40, 0..6]).to_owned();

    let res = vireo_wrap::vireo_wrap(
        &ad,
        &dp,
        None,
        Some(2),
        true,
        1,
        Some(5),
        false,
        3,
        1,
        0,
        Some("distance"),
        true,
        1,
        false,
        false,
        3,
    )
    .unwrap();
    assert_eq!(res.ambient_psi.as_ref().unwrap().nrows(), 6);
    assert_eq!(res.psi_var.as_ref().unwrap().nrows(), 6);
    assert_eq!(res.psi_llratio.as_ref().unwrap().len(), 6);
}

#[test]
fn ambient_prediction_is_stable_with_multiple_workers() {
    let mut gt_prob = Array3::<f64>::zeros((4, 3, 3));
    for v in 0..4 {
        for d in 0..3 {
            gt_prob[[v, d, (v + d) % 3]] = 1.0;
        }
    }
    let beta_mu = Array2::from_shape_vec((1, 3), vec![0.05, 0.5, 0.95]).unwrap();
    let id_prob =
        Array2::from_shape_vec((3, 3), vec![0.8, 0.1, 0.1, 0.2, 0.7, 0.1, 0.1, 0.2, 0.7]).unwrap();
    let ad = Array2::from_shape_vec(
        (4, 3),
        vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 8.0, 7.0, 6.0, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let dp = Array2::from_shape_vec(
        (4, 3),
        vec![
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0, 8.0, 8.0, 8.0,
        ],
    )
    .unwrap();
    let serial =
        vireo_doublet::predit_ambient(&gt_prob, &beta_mu, &id_prob, &ad, &dp, 1, Some(0.0))
            .unwrap();
    let parallel =
        vireo_doublet::predit_ambient(&gt_prob, &beta_mu, &id_prob, &ad, &dp, 4, Some(0.0))
            .unwrap();
    assert_eq!(serial, parallel);
}

#[test]
fn matches_python_predict_doublet_on_numeric_probe() {
    let Some(py) = run_python_probe(
        r#"
import numpy as np
from vireoSNP.utils.vireo_doublet import predict_doublet
from vireoSNP.utils.vireo_model import Vireo
v = Vireo(
    n_cell=3,
    n_var=4,
    n_donor=3,
    n_GT=3,
    learn_GT=False,
    learn_theta=False,
    ASE_mode=False,
    fix_beta_sum=False,
)
v.GT_prob = np.zeros((4, 3, 3))
for i in range(4):
    for j in range(3):
        v.GT_prob[i, j, (i + j) % 3] = 1.0
v.beta_mu = np.array([[0.05, 0.5, 0.95]])
v.beta_sum = np.array([[30.0, 6.0, 30.0]])
v.ID_prior = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
v.ID_prob = v.ID_prior.copy()
ad = np.array([[1, 2, 3], [4, 3, 2], [8, 7, 6], [2, 3, 4]], dtype=float)
dp = np.array([[10, 10, 10], [10, 10, 10], [12, 12, 12], [8, 8, 8]], dtype=float)
doublet, singlet, llr = predict_doublet(
    v, ad, dp, update_GT=False, update_ID=False, doublet_rate_prior=0.0
)
print('doublet\t%s' % ','.join('%.17g' % x for x in doublet.reshape(-1)))
print('singlet\t%s' % ','.join('%.17g' % x for x in singlet.reshape(-1)))
print('llr\t%s' % ','.join('%.17g' % x for x in llr.reshape(-1)))
"#,
    ) else {
        return;
    };

    let mut gt_prob = Array3::<f64>::zeros((4, 3, 3));
    for v in 0..4 {
        for d in 0..3 {
            gt_prob[[v, d, (v + d) % 3]] = 1.0;
        }
    }
    let beta_mu = Array2::from_shape_vec((1, 3), vec![0.05, 0.5, 0.95]).unwrap();
    let beta_sum = Array2::from_shape_vec((1, 3), vec![30.0, 6.0, 30.0]).unwrap();
    let id_prior =
        Array2::from_shape_vec((3, 3), vec![0.8, 0.1, 0.1, 0.2, 0.7, 0.1, 0.1, 0.2, 0.7]).unwrap();
    let ad = Array2::from_shape_vec(
        (4, 3),
        vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 8.0, 7.0, 6.0, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let dp = Array2::from_shape_vec(
        (4, 3),
        vec![
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0, 8.0, 8.0, 8.0,
        ],
    )
    .unwrap();
    let (doublet, singlet, llr) = vireo_doublet::predict_doublet(
        &gt_prob,
        &beta_mu,
        &beta_sum,
        Some(&id_prior),
        &ad,
        &dp,
        false,
        false,
        Some(0.0),
    )
    .unwrap();
    assert_slice_close(
        doublet.as_slice().unwrap(),
        &parse_probe_f64s(&py, "doublet\t"),
        1e-10,
    );
    assert_slice_close(
        singlet.as_slice().unwrap(),
        &parse_probe_f64s(&py, "singlet\t"),
        1e-10,
    );
    assert_slice_close(&llr, &parse_probe_f64s(&py, "llr\t"), 1e-10);
}

#[test]
fn runs_bulk_likelihood_ratio_without_pyvalue() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    )
    .unwrap();
    let ad = dense(dat.layers.get("AD").expect("missing AD"));
    let dp = dense(dat.layers.get("DP").expect("missing DP"));
    let ad = ad.clone().slice(s![0..12, 0]).to_owned();
    let dp = dp.clone().slice(s![0..12, 0]).to_owned();
    let mut gt_prob = Array3::<f64>::zeros((12, 2, 3));
    for v in 0..12 {
        gt_prob[[v, 0, v % 3]] = 1.0;
        gt_prob[[v, 1, (v + 1) % 3]] = 1.0;
    }
    let psi = Array1::from(vec![0.7, 0.3]);
    let psi_null = Array1::from(vec![0.5, 0.5]);
    let theta = Array1::from(vec![0.01, 0.5, 0.99]);
    let (lr, p) =
        vireo_bulk::LikRatio_test(&psi, &psi_null, &ad, &dp, &gt_prob, &theta, false).unwrap();
    assert!(lr.is_finite());
    assert!(p.is_finite());
}

#[test]
fn fits_bulk_model_on_real_count_slice_without_pyvalue() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    )
    .unwrap();
    let ad_mat = dense(dat.layers.get("AD").expect("missing AD"));
    let dp_mat = dense(dat.layers.get("DP").expect("missing DP"));
    let idx: Vec<usize> = (0..ad_mat.nrows())
        .filter(|&i| dp_mat[[i, 0]] > 0.0 && ad_mat[[i, 0]] <= dp_mat[[i, 0]])
        .take(20)
        .collect();
    let ad = Array1::from(idx.iter().map(|&i| ad_mat[[i, 0]]).collect::<Vec<_>>());
    let dp = Array1::from(idx.iter().map(|&i| dp_mat[[i, 0]]).collect::<Vec<_>>());
    let mut gt_prob = Array3::<f64>::from_elem((20, 2, 3), 0.01);
    for v in 0..20 {
        gt_prob[[v, 0, v % 3]] = 0.98;
        gt_prob[[v, 1, (v + 1) % 3]] = 0.98;
    }
    let mut model = VireoBulk::default();
    model.__init__(2, 3, None, None).unwrap();
    model
        .fit(&ad, &dp, &gt_prob, 10, 2, 1e-3, true, 0, None, false)
        .unwrap();
    let psi = &model.psi;
    let log_lik_all = &model.log_lik_all;
    assert_eq!(psi.len(), 2);
    assert!(!log_lik_all.is_empty());
    assert!(log_lik_all.iter().all(|v| v.is_finite()));
}

#[test]
fn matches_snps_to_genes_without_pyvalue() {
    let vcf = vcf_utils::load_VCF(
        "vireo/data/cellSNP_mat/cellSNP.base.vcf.gz",
        false,
        false,
        true,
        None,
    )
    .unwrap();
    let genes = vcf_utils::GeneData {
        chrom: vec!["1".to_string(), "2".to_string(), "chr1".to_string()],
        start: vec![0, 0, 0],
        stop: vec![1_000_000_000, 1_000_000_000, 1_000_000_000],
        gene: vec![
            "gene_chr1".to_string(),
            "gene_chr2".to_string(),
            "gene_chr1_prefixed".to_string(),
        ],
    };
    let (gene_list, flags) =
        vcf_utils::snp_gene_match(&vcf.fixed_info, &genes, None, true, Some(&[0, 1000]), false)
            .unwrap();
    assert_eq!(gene_list.len(), vcf.variants.len());
    assert_eq!(flags.len(), vcf.variants.len());
}

#[test]
fn makes_whitelists_without_pyvalue() {
    let out_dir = std::env::temp_dir().join(format!(
        "vireo-rs-whitelist-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&out_dir).unwrap();
    let donor_ids = out_dir.join("donor_ids.tsv");
    std::fs::write(
        &donor_ids,
        "cell\tdonor_id\ncellA-1\tdonor0\ncellB-1\tdoublet\ncellC-2\tdonor1\n",
    )
    .unwrap();
    let prefix = out_dir.join("wl");
    io_utils::make_whitelists(&donor_ids.to_string_lossy(), &prefix.to_string_lossy()).unwrap();
    assert_eq!(
        std::fs::read_to_string(out_dir.join("wl_donor0.txt")).unwrap(),
        "cellA\n"
    );
    assert_eq!(
        std::fs::read_to_string(out_dir.join("wl_donor1.txt")).unwrap(),
        "cellC\n"
    );
    std::fs::remove_dir_all(out_dir).unwrap();
}

#[test]
fn materializes_minicode_plot_matrix_without_pyvalue() {
    let barcode_set = vec!["#012".to_string(), "#120".to_string(), "#201".to_string()];
    let mat = base_plot::minicode_plot(&barcode_set, None, None, "Set3", "none").unwrap();
    assert_eq!(mat.dim(), (3, 3));
    assert_eq!(mat[[0, 0]], 0.0);
    assert_eq!(mat[[1, 1]], 2.0);
    assert_eq!(mat[[2, 2]], 1.0);
}

#[cfg(feature = "plotting-pdf")]
#[test]
fn renders_plot_files_with_plotters() {
    let out_dir = std::env::temp_dir().join(format!(
        "vireo-rs-plotting-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&out_dir).unwrap();

    let barcode_set = vec!["#012".to_string(), "#120".to_string(), "#201".to_string()];
    let out_tsv = out_dir.join("GTbarcode.tsv");
    base_plot::save_minicode_plot(
        &out_tsv.to_string_lossy(),
        &barcode_set,
        None,
        None,
        (4.0, 2.0),
        "pdf",
    )
    .unwrap();
    assert!(out_dir.join("GTbarcode.pdf").metadata().unwrap().len() > 0);

    let mut gpb = Array3::<f64>::zeros((2, 2, 3));
    gpb[[0, 0, 0]] = 1.0;
    gpb[[0, 1, 1]] = 1.0;
    gpb[[1, 0, 2]] = 1.0;
    gpb[[1, 1, 0]] = 1.0;
    let donor_names = vec!["donor0".to_string(), "donor1".to_string()];
    base_plot::plot_GT(&out_dir.to_string_lossy(), &gpb, &donor_names, None, None).unwrap();
    assert!(
        out_dir
            .join("fig_GT_distance_estimated.pdf")
            .metadata()
            .unwrap()
            .len()
            > 0
    );

    let heat =
        Array2::from_shape_vec((3, 3), vec![0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.4, 0.5, 0.6]).unwrap();
    let row_anno = vec!["B".to_string(), "A".to_string(), "B".to_string()];
    let col_anno = vec!["Y".to_string(), "X".to_string(), "X".to_string()];
    assert!(base_plot::anno_heat(
        &heat,
        Some(&row_anno),
        Some(&col_anno),
        None,
        None,
        true,
        true,
        false,
        false,
    )
    .is_some());
    let anno_out = out_dir.join("anno.tsv");
    base_plot::save_anno_heat(
        &anno_out.to_string_lossy(),
        &heat,
        Some(&row_anno),
        Some(&col_anno),
        None,
        None,
        true,
        true,
        (4.0, 3.0),
        "pdf",
    )
    .unwrap();
    assert!(out_dir.join("anno.pdf").metadata().unwrap().len() > 0);

    std::fs::remove_dir_all(out_dir).unwrap();
}

#[test]
fn runs_high_level_fit_api_and_writes_outputs() {
    let out_dir = std::env::temp_dir().join(format!(
        "vireo-rs-high-level-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let result = fit("vireo/data/cellSNP_mat")
        .with_donors("vireo/data/donors.cellSNP.vcf.gz")
        .genotype_tag("GT")
        .seed(1)
        .run()
        .unwrap();
    assert_eq!(result.cell_names.len(), result.result.id_prob.nrows());
    assert!(!result.donor_names.is_empty());
    result.write_outputs(out_dir.to_string_lossy()).unwrap();
    assert!(out_dir.join("donor_ids.tsv").exists());
    assert!(out_dir.join("summary.tsv").exists());
    std::fs::remove_dir_all(out_dir).unwrap();
}

#[test]
fn runs_high_level_fit_api_with_vartrix_and_cell_range() {
    let result = fit("ignored-cell-data-when-vartrix-is-set")
        .vartrix_data(concat!(
            "vireo/data/cellSNP_mat/cellSNP.tag.AD.mtx,",
            "vireo/data/cellSNP_mat/cellSNP.tag.OTH.mtx,",
            "vireo/data/cellSNP_mat/cellSNP.samples.tsv,",
            "vireo/data/cellSNP_mat/cellSNP.base.vcf.gz"
        ))
        .with_donors("vireo/data/donors.cellSNP.vcf.gz")
        .genotype_tag("GT")
        .cell_range(0, 1)
        .no_plot(true)
        .seed(3)
        .run()
        .unwrap();
    assert_eq!(result.cell_names.len(), 1);
    assert_eq!(result.result.id_prob.nrows(), 1);
    assert_eq!(result.n_vars.len(), 1);
}

#[test]
fn runs_high_level_fit_with_donor_count_different_from_vcf() {
    let fewer = fit("vireo/data/cellSNP_mat")
        .with_donors("vireo/data/donors.cellSNP.vcf.gz")
        .genotype_tag("GT")
        .infer_donors(1)
        .cell_range(0, 2)
        .no_plot(true)
        .seed(7)
        .run()
        .unwrap();
    assert_eq!(fewer.result.gt_prob.shape()[1], 1);
    assert_eq!(fewer.result.id_prob.ncols(), 1);

    let extra = fit("vireo/data/cellSNP_mat")
        .with_donors("vireo/data/donors.cellSNP.vcf.gz")
        .genotype_tag("GT")
        .infer_donors(3)
        .cell_range(0, 2)
        .no_plot(true)
        .seed(7)
        .run()
        .unwrap();
    assert_eq!(extra.result.gt_prob.shape()[1], 3);
    assert_eq!(extra.result.id_prob.ncols(), 3);
}

#[cfg(feature = "cli")]
#[test]
fn vireo_snp_cli_nproc_parallel_matches_serial_outputs() {
    let Some(bin) = option_env!("CARGO_BIN_EXE_vireoSNP") else {
        return;
    };
    let root = std::env::temp_dir().join(format!(
        "vireo-rs-cli-nproc-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let serial_dir = root.join("serial");
    let parallel_dir = root.join("parallel");
    std::fs::create_dir_all(&serial_dir).unwrap();
    std::fs::create_dir_all(&parallel_dir).unwrap();

    for (nproc, out_dir) in [(1usize, &serial_dir), (2usize, &parallel_dir)] {
        let output = Command::new(bin)
            .args([
                "--cellData",
                "vireo/data/cellSNP_mat",
                "--nDonor",
                "2",
                "--outDir",
                &out_dir.to_string_lossy(),
                "--cellRange",
                "0-10",
                "--nInit",
                "3",
                "--randSeed",
                "19",
                "--noDoublet",
                "--noPlot",
                "--nproc",
                &nproc.to_string(),
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "vireoSNP --nproc {nproc} failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    for file in ["donor_ids.tsv", "prob_singlet.tsv", "summary.tsv"] {
        let serial = std::fs::read_to_string(serial_dir.join(file)).unwrap();
        let parallel = std::fs::read_to_string(parallel_dir.join(file)).unwrap();
        assert_eq!(
            serial, parallel,
            "{file} differs between nproc=1 and nproc=2"
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(feature = "cli")]
#[test]
fn vireo_snp_cli_rejects_optparse_style_edge_cases() {
    let args = vec!["--cellData".to_string()];
    assert!(matches!(
        vireo::builder_from_cli_args(&args),
        Err(VireoError::CliMissingValue(option)) if option == "--cellData"
    ));

    let args = vec!["--nDonor".to_string(), "two".to_string()];
    assert!(matches!(
        vireo::builder_from_cli_args(&args),
        Err(VireoError::CliInvalidValue { option, value }) if option == "--nDonor" && value == "two"
    ));

    let args = vec!["--cellRange".to_string(), "0:10".to_string()];
    assert!(matches!(
        vireo::builder_from_cli_args(&args),
        Err(VireoError::InvalidCellRange(range)) if range == "0:10"
    ));

    let args = vec!["--unknown".to_string()];
    assert!(matches!(
        vireo::builder_from_cli_args(&args),
        Err(VireoError::CliUnknownOption(option)) if option == "--unknown"
    ));
}
