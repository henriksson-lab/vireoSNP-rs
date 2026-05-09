use ndarray::{s, Array1, Array2, Array3, Ix2};
use vireo_rs::vireo_snp::plot::base_plot;
use vireo_rs::vireo_snp::utils::vireo_model::Vireo;
use vireo_rs::vireo_snp::utils::{
    base_utils,
    bmm_model::BinomMixtureVB,
    io_utils, variant_select, vcf_utils, vireo_base,
    vireo_bulk::{self, VireoBulk},
    vireo_doublet, vireo_wrap,
};
use vireo_rs::PyValue;

#[test]
fn reads_cellsnp_fixture() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    );
    let dat = dat.unwrap();
    let PyValue::StringVec(samples) = dat.get("samples").unwrap() else {
        panic!("missing samples");
    };
    let PyValue::StringVec(variants) = dat.get("variants").unwrap() else {
        panic!("missing variants");
    };
    let PyValue::ArrayF64(ad) = dat.get("AD").unwrap() else {
        panic!("missing AD");
    };
    let PyValue::ArrayF64(dp) = dat.get("DP").unwrap() else {
        panic!("missing DP");
    };
    assert!(!samples.is_empty());
    assert!(!variants.is_empty());
    assert_eq!(ad.shape(), dp.shape());
    assert_eq!(ad.shape()[0], variants.len());
    assert_eq!(ad.shape()[1], samples.len());
}

#[test]
fn computes_binomial_coefficients_on_real_count_slice() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    );
    let dat = dat.unwrap();
    let PyValue::ArrayF64(ad) = dat.get("AD").unwrap() else {
        panic!("missing AD");
    };
    let PyValue::ArrayF64(dp) = dat.get("DP").unwrap() else {
        panic!("missing DP");
    };
    let ad = ad.clone().into_dimensionality::<Ix2>().unwrap();
    let dp = dp.clone().into_dimensionality::<Ix2>().unwrap();
    let ad = ad.slice(s![0..10, 0..10]).to_owned().into_dyn();
    let dp = dp.slice(s![0..10, 0..10]).to_owned().into_dyn();
    let coeff = vireo_base::get_binom_coeff(&ad, &dp, 700.0);
    assert!(!coeff.is_empty());
    assert!(coeff.iter().all(|v| v.is_finite()));
}

#[test]
fn amplifies_log_likelihood_array_without_pyvalue() {
    let x = Array2::<f64>::from_shape_vec(
        (3, 3),
        vec![-5.0, -2.0, -3.0, 10.0, 7.0, 8.0, 0.5, 0.25, 0.0],
    )
    .unwrap()
    .into_dyn();
    let amplified = vireo_base::loglik_amplify(&x, Some(1)).unwrap();
    let amplified = amplified.into_dimensionality::<Ix2>().unwrap();
    assert_eq!(amplified[[0, 1]], 0.0);
    assert_eq!(amplified[[1, 0]], 0.0);
    assert_eq!(amplified[[2, 0]], 0.0);
    assert!(amplified.iter().all(|v| *v <= 0.0));
}

#[test]
fn normalizes_arrays_without_pyvalue() {
    let x = Array2::<f64>::from_shape_vec((2, 3), vec![1.0, 1.0, 2.0, 2.0, 3.0, 5.0])
        .unwrap()
        .into_dyn();
    let normalized = vireo_base::normalize(&x, Some(1)).unwrap();
    let normalized = normalized.into_dimensionality::<Ix2>().unwrap();
    assert!((normalized.row(0).sum() - 1.0).abs() < 1e-12);
    assert!((normalized.row(1).sum() - 1.0).abs() < 1e-12);
    let normalized = vireo_base::tensor_normalize(&x, Some(1)).unwrap();
    let normalized = normalized.into_dimensionality::<Ix2>().unwrap();
    assert!((normalized.row(0).sum() - 1.0).abs() < 1e-12);
}

#[test]
fn computes_logbincoeff_without_pyvalue() {
    let n = Array2::<f64>::from_shape_vec((1, 3), vec![4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let k = Array2::<f64>::from_shape_vec((1, 3), vec![2.0, 2.0, 3.0])
        .unwrap()
        .into_dyn();
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
    let PyValue::StringVec(samples) = dat.get("samples").unwrap() else {
        panic!("missing samples");
    };
    let PyValue::StringVec(variants) = dat.get("variants").unwrap() else {
        panic!("missing variants");
    };
    let PyValue::ArrayF64(ad) = dat.get("AD").unwrap() else {
        panic!("missing AD");
    };
    let PyValue::ArrayF64(dp) = dat.get("DP").unwrap() else {
        panic!("missing DP");
    };
    assert_eq!(ad.shape(), dp.shape());
    assert_eq!(ad.shape()[0], variants.len());
    assert_eq!(ad.shape()[1], samples.len());
    assert!(dp.sum() >= ad.sum());
}

#[test]
fn loads_cells_vcf_sparse_and_materializes_read_layers() {
    let vcf =
        vcf_utils::load_VCF("vireo/data/cells.cellSNP.vcf.gz", false, true, true, None).unwrap();
    let PyValue::StringVec(variants) = vcf.get("variants").unwrap() else {
        panic!("missing variants");
    };
    let PyValue::StringVec(samples) = vcf.get("samples").unwrap() else {
        panic!("missing samples");
    };
    let PyValue::Map(geno_info) = vcf.get("GenoINFO").unwrap() else {
        panic!("missing GenoINFO");
    };
    assert!(geno_info.contains_key("indices"));
    assert!(geno_info.contains_key("indptr"));
    assert!(geno_info.contains_key("shape"));

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
    let PyValue::StringVec(variants) = donor_vcf.get("variants").unwrap() else {
        panic!("missing variants");
    };
    let PyValue::StringVec(samples) = donor_vcf.get("samples").unwrap() else {
        panic!("missing samples");
    };
    let PyValue::Map(geno_info) = donor_vcf.get("GenoINFO").unwrap() else {
        panic!("missing GenoINFO");
    };
    let PyValue::StringMatrix(gt) = geno_info.get("GT").unwrap() else {
        panic!("missing GT");
    };
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
    let PyValue::StringVec(variants) = vcf.get_mut("variants").unwrap() else {
        panic!("missing variants");
    };
    variants.truncate(5);
    let PyValue::Map(fixed_info) = vcf.get_mut("FixedINFO").unwrap() else {
        panic!("missing FixedINFO");
    };
    for value in fixed_info.values_mut() {
        let PyValue::StringVec(values) = value else {
            panic!("unexpected FixedINFO value");
        };
        values.truncate(5);
    }
    vcf.insert(
        "samples".to_string(),
        PyValue::StringVec(vec!["donor0".to_string(), "donor1".to_string()]),
    );

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
    vcf.insert(
        "GenoINFO".to_string(),
        PyValue::Map(
            geno_info
                .into_iter()
                .map(|(k, v)| (k, PyValue::StringMatrix(v)))
                .collect(),
        ),
    );

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
    let PyValue::StringVec(samples) = loaded.get("samples").unwrap() else {
        panic!("missing reloaded samples");
    };
    let PyValue::StringVec(variants) = loaded.get("variants").unwrap() else {
        panic!("missing reloaded variants");
    };
    let PyValue::Map(geno_info) = loaded.get("GenoINFO").unwrap() else {
        panic!("missing reloaded GenoINFO");
    };
    assert_eq!(samples, &vec!["donor0".to_string(), "donor1".to_string()]);
    assert_eq!(variants.len(), 5);
    assert!(matches!(
        geno_info.get("GT"),
        Some(PyValue::StringMatrix(rows)) if rows.len() == 5 && rows[0].len() == 2
    ));
    std::fs::remove_file(out_file).unwrap();
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
    let PyValue::StringVec(cell_variants) = cell_dat.get("variants").unwrap() else {
        panic!("missing matched cell variants");
    };
    let PyValue::StringVec(donor_variants) = donor_vcf.get("variants").unwrap() else {
        panic!("missing matched donor variants");
    };
    let PyValue::ArrayF64(ad) = cell_dat.get("AD").unwrap() else {
        panic!("missing matched AD");
    };
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
    let PyValue::I64(n_var) = matched.get("matched_n_var").unwrap() else {
        panic!("missing matched_n_var");
    };
    let PyValue::StringVec(donors1) = matched.get("matched_donors1").unwrap() else {
        panic!("missing matched_donors1");
    };
    let PyValue::StringVec(donors2) = matched.get("matched_donors2").unwrap() else {
        panic!("missing matched_donors2");
    };
    let PyValue::ArrayF64(diff) = matched.get("matched_GPb_diff").unwrap() else {
        panic!("missing matched_GPb_diff");
    };
    assert!(*n_var > 0);
    assert_eq!(donors1.len(), donors2.len());
    assert_eq!(diff.shape(), &[donors1.len(), donors2.len()]);
}

#[test]
fn fits_vireo_model_on_real_cellsnp_slice() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    );
    let dat = dat.unwrap();
    let PyValue::ArrayF64(ad) = dat.get("AD").unwrap() else {
        panic!("missing AD");
    };
    let PyValue::ArrayF64(dp) = dat.get("DP").unwrap() else {
        panic!("missing DP");
    };
    let ad = ad.clone().into_dimensionality::<Ix2>().unwrap();
    let dp = dp.clone().into_dimensionality::<Ix2>().unwrap();
    let ad = ad.slice(s![0..80, 0..60]).to_owned().into_dyn();
    let dp = dp.slice(s![0..80, 0..60]).to_owned().into_dyn();

    let mut model = Vireo::default();
    model
        .__init__(
            60, 80, 2, 3, true, true, false, false, None, None, None, None,
        )
        .unwrap();
    model
        .fit(
            &ad.into_dimensionality::<Ix2>().unwrap(),
            &dp.into_dimensionality::<Ix2>().unwrap(),
            4,
            1,
            Some(1e-2),
            1,
            false,
            None,
            1,
        )
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
fn fits_binom_mixture_on_real_cellsnp_slice_without_pyvalue_updates() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    )
    .unwrap();
    let PyValue::ArrayF64(ad) = dat.get("AD").unwrap() else {
        panic!("missing AD");
    };
    let PyValue::ArrayF64(dp) = dat.get("DP").unwrap() else {
        panic!("missing DP");
    };
    let ad = ad
        .clone()
        .into_dimensionality::<Ix2>()
        .unwrap()
        .slice(s![0..40, 0..30])
        .to_owned();
    let dp = dp
        .clone()
        .into_dimensionality::<Ix2>()
        .unwrap()
        .slice(s![0..40, 0..30])
        .to_owned();
    let mut model = BinomMixtureVB::default();
    model.__init__(30, 40, 2, false, None, None, None).unwrap();
    model.fit(&ad, &dp, 2, 5, Some(3), 0, None).unwrap();
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
    let PyValue::StringVec(samples) = dat.get("samples").unwrap() else {
        panic!("missing samples");
    };
    let PyValue::ArrayF64(ad) = dat.get("AD").unwrap() else {
        panic!("missing AD");
    };
    let PyValue::ArrayF64(dp) = dat.get("DP").unwrap() else {
        panic!("missing DP");
    };
    let ad = ad.clone().into_dimensionality::<Ix2>().unwrap();
    let dp = dp.clone().into_dimensionality::<Ix2>().unwrap();
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
    let res_map = &res;
    let PyValue::ArrayF64(id_prob) = res_map.get("ID_prob").unwrap() else {
        panic!("missing ID_prob");
    };
    let PyValue::ArrayF64(gt_prob) = res_map.get("GT_prob").unwrap() else {
        panic!("missing GT_prob");
    };
    let PyValue::ArrayF64(doublet_prob) = res_map.get("doublet_prob").unwrap() else {
        panic!("missing doublet_prob");
    };
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
    let PyValue::ArrayF64(ad) = dat.get("AD").unwrap() else {
        panic!("expected AD");
    };
    let PyValue::ArrayF64(dp) = dat.get("DP").unwrap() else {
        panic!("expected DP");
    };
    let ad = ad.clone().into_dimensionality::<Ix2>().unwrap();
    let dp = dp.clone().into_dimensionality::<Ix2>().unwrap();
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
    let PyValue::ArrayF64(ad) = dat.get("AD").unwrap() else {
        panic!("expected AD");
    };
    let PyValue::ArrayF64(dp) = dat.get("DP").unwrap() else {
        panic!("expected DP");
    };
    let ad = ad
        .clone()
        .into_dimensionality::<Ix2>()
        .unwrap()
        .slice(s![0..50, 0..24])
        .to_owned();
    let dp = dp
        .clone()
        .into_dimensionality::<Ix2>()
        .unwrap()
        .slice(s![0..50, 0..24])
        .to_owned();

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
    let PyValue::ArrayF64(ad) = dat.get("AD").unwrap() else {
        panic!("expected AD");
    };
    let PyValue::ArrayF64(dp) = dat.get("DP").unwrap() else {
        panic!("expected DP");
    };
    let ad = ad
        .clone()
        .into_dimensionality::<Ix2>()
        .unwrap()
        .slice(s![0..50, 0..24])
        .to_owned();
    let dp = dp
        .clone()
        .into_dimensionality::<Ix2>()
        .unwrap()
        .slice(s![0..50, 0..24])
        .to_owned();
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
    let PyValue::StringVec(variants) = dat.get("variants").unwrap() else {
        panic!("expected variants");
    };
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
fn runs_bulk_likelihood_ratio_without_pyvalue() {
    let dat = io_utils::read_cellSNP(
        "vireo/data/cellSNP_mat",
        Some(&["AD".to_string(), "DP".to_string()]),
    )
    .unwrap();
    let PyValue::ArrayF64(ad) = dat.get("AD").unwrap() else {
        panic!("missing AD");
    };
    let PyValue::ArrayF64(dp) = dat.get("DP").unwrap() else {
        panic!("missing DP");
    };
    let ad = ad
        .clone()
        .into_dimensionality::<Ix2>()
        .unwrap()
        .slice(s![0..12, 0])
        .to_owned();
    let dp = dp
        .clone()
        .into_dimensionality::<Ix2>()
        .unwrap()
        .slice(s![0..12, 0])
        .to_owned();
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
    let PyValue::ArrayF64(ad) = dat.get("AD").unwrap() else {
        panic!("missing AD");
    };
    let PyValue::ArrayF64(dp) = dat.get("DP").unwrap() else {
        panic!("missing DP");
    };
    let ad_mat = ad.clone().into_dimensionality::<Ix2>().unwrap();
    let dp_mat = dp.clone().into_dimensionality::<Ix2>().unwrap();
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
    let PyValue::Map(fixed_info) = vcf.get("FixedINFO").unwrap() else {
        panic!("missing FixedINFO");
    };
    let mut genes = std::collections::BTreeMap::new();
    genes.insert(
        "chrom".to_string(),
        PyValue::StringVec(vec!["1".to_string(), "2".to_string(), "chr1".to_string()]),
    );
    genes.insert("start".to_string(), PyValue::I64Vec(vec![0, 0, 0]));
    genes.insert(
        "stop".to_string(),
        PyValue::I64Vec(vec![1_000_000_000, 1_000_000_000, 1_000_000_000]),
    );
    genes.insert(
        "gene".to_string(),
        PyValue::StringVec(vec![
            "gene_chr1".to_string(),
            "gene_chr2".to_string(),
            "gene_chr1_prefixed".to_string(),
        ]),
    );
    let (gene_list, flags) =
        vcf_utils::snp_gene_match(fixed_info, &genes, None, true, Some(&[0, 1000]), false).unwrap();
    let PyValue::StringVec(variants) = vcf.get("variants").unwrap() else {
        panic!("missing variants");
    };
    assert_eq!(gene_list.len(), variants.len());
    assert_eq!(flags.len(), variants.len());
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
    let mat = base_plot::minicode_plot(&barcode_set, None, None, "Set3", "none", None).unwrap();
    assert_eq!(mat.dim(), (3, 3));
    assert_eq!(mat[[0, 0]], 0.0);
    assert_eq!(mat[[1, 1]], 2.0);
    assert_eq!(mat[[2, 2]], 1.0);
}
