use crate::vireo_snp::utils::io_utils;
use crate::vireo_snp::utils::vcf_utils;
use crate::vireo_snp::utils::vireo_wrap;
use crate::PyValue;
use ndarray::{Axis, Ix2, Ix3};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

pub fn show_progress<T>(rv: T) -> T {
    rv
}

pub fn main() -> PyValue {
    let args: Vec<String> = env::args().collect();
    if args.len() <= 1 {
        return PyValue::None;
    }
    let mut cell_data = None;
    let mut donor_file = None;
    let mut n_donor = None;
    let mut out_dir = None;
    let mut geno_tag = "PL".to_string();
    let mut no_doublet = false;
    let mut n_init = 50i64;
    let mut n_extra_donor = 0i64;
    let mut extra_donor_mode = "distance".to_string();
    let mut force_learn_gt = false;
    let mut ase_mode = false;
    let mut check_ambient = false;
    let mut rand_seed = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--cellData" | "-c" => {
                i += 1;
                cell_data = args.get(i).cloned();
            }
            "--donorFile" | "-d" => {
                i += 1;
                donor_file = args.get(i).cloned();
            }
            "--nDonor" | "-N" => {
                i += 1;
                n_donor = args.get(i).and_then(|v| v.parse::<i64>().ok());
            }
            "--outDir" | "-o" => {
                i += 1;
                out_dir = args.get(i).cloned();
            }
            "--genoTag" | "-t" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    geno_tag = v.clone();
                }
            }
            "--noDoublet" => no_doublet = true,
            "--nInit" | "-M" => {
                i += 1;
                n_init = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(n_init);
            }
            "--extraDonor" => {
                i += 1;
                n_extra_donor = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(0);
            }
            "--extraDonorMode" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    extra_donor_mode = v.clone();
                }
            }
            "--forceLearnGT" => force_learn_gt = true,
            "--ASEmode" => ase_mode = true,
            "--callAmbientRNAs" => check_ambient = true,
            "--randSeed" => {
                i += 1;
                rand_seed = args.get(i).and_then(|v| v.parse::<u64>().ok());
            }
            _ => {}
        }
        i += 1;
    }
    let Some(cell_data_path) = cell_data else {
        return PyValue::None;
    };
    let out_dir = out_dir.unwrap_or_else(|| {
        let input = fs::canonicalize(&cell_data_path)
            .unwrap_or_else(|_| Path::new(&cell_data_path).to_path_buf());
        input
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("vireo")
            .to_string_lossy()
            .into_owned()
    });
    if fs::create_dir_all(&out_dir).is_err() {
        return PyValue::None;
    }
    let mut cell_dat = if Path::new(&cell_data_path).is_dir() {
        match io_utils::read_cellSNP(&cell_data_path, None) {
            Some(m) => m,
            None => return PyValue::None,
        }
    } else {
        let cell_vcf = match vcf_utils::load_VCF(&cell_data_path, true, true, true, None) {
            Some(m) => m,
            None => return PyValue::None,
        };
        let mut dat: BTreeMap<String, PyValue> = match cell_vcf.get("GenoINFO") {
            Some(PyValue::Map(value)) => {
                let keys = ["AD".to_string(), "DP".to_string()];
                match vcf_utils::read_sparse_GeneINFO(value, Some(&keys), None) {
                    Some(m) => m
                        .into_iter()
                        .map(|(k, v)| (k, PyValue::ArrayF64(v.into_dyn())))
                        .collect::<BTreeMap<String, PyValue>>(),
                    None => return PyValue::None,
                }
            }
            _ => return PyValue::None,
        };
        for key in ["samples", "variants", "FixedINFO", "contigs", "comments"] {
            if let Some(value) = cell_vcf.get(key) {
                dat.insert(key.to_string(), value.clone());
            }
        }
        dat
    };
    let mut donor_gpb = PyValue::None;
    let donor_names;
    let mut learn_gt = true;
    if let Some(donor_file) = donor_file {
        let geno_tags = [geno_tag.clone()];
        let donor_vcf = match vcf_utils::load_VCF(&donor_file, true, true, false, Some(&geno_tags))
        {
            Some(m) => m,
            None => return PyValue::None,
        };
        let (cell_matched, donor_vcf) = match io_utils::match_donor_VCF(cell_dat, donor_vcf) {
            Some(v) => v,
            None => return PyValue::None,
        };
        cell_dat = cell_matched;
        let geno = match donor_vcf.get("GenoINFO") {
            Some(PyValue::Map(m)) => match m.get(&geno_tag) {
                Some(PyValue::StringMatrix(v)) => v.clone(),
                _ => return PyValue::None,
            },
            _ => return PyValue::None,
        };
        let donor_gpb_arr = match vcf_utils::parse_donor_GPb(&geno, &geno_tag, 0.0) {
            Some(x) => x,
            None => return PyValue::None,
        };
        let donor_count = donor_gpb_arr.shape()[1] as i64;
        donor_gpb = PyValue::ArrayF64(donor_gpb_arr.into_dyn());
        match n_donor {
            None => {
                n_donor = Some(donor_count);
                donor_names = match donor_vcf.get("samples") {
                    Some(PyValue::StringVec(v)) => v.clone(),
                    _ => (0..donor_count).map(|x| format!("donor{x}")).collect(),
                };
                learn_gt = false;
            }
            Some(n) if n == donor_count => {
                donor_names = match donor_vcf.get("samples") {
                    Some(PyValue::StringVec(v)) => v.clone(),
                    _ => (0..n).map(|x| format!("donor{x}")).collect(),
                };
                learn_gt = false;
            }
            Some(n) if n < donor_count => {
                donor_names = (0..n).map(|x| format!("donor{x}")).collect();
                learn_gt = false;
            }
            Some(n) => {
                let mut names = match donor_vcf.get("samples") {
                    Some(PyValue::StringVec(v)) => v.clone(),
                    _ => (0..donor_count).map(|x| format!("donor{x}")).collect(),
                };
                names.extend((donor_count..n).map(|x| format!("donor{x}")));
                donor_names = names;
                learn_gt = true;
            }
        }
    } else if let Some(n) = n_donor {
        donor_names = (0..n).map(|x| format!("donor{x}")).collect();
    } else {
        return PyValue::None;
    }
    if force_learn_gt {
        learn_gt = true;
    }
    let n_donor = n_donor.unwrap_or(donor_names.len() as i64);
    let n_extra_donor = if learn_gt && n_extra_donor == 0 {
        (n_donor as f64).sqrt().round() as i64
    } else if learn_gt {
        n_extra_donor
    } else {
        0
    };
    let n_init = if learn_gt { n_init } else { 1 };
    let ad = match cell_dat.get("AD") {
        Some(v) => v.clone(),
        _ => return PyValue::None,
    };
    let dp = match cell_dat.get("DP") {
        Some(v) => v.clone(),
        _ => return PyValue::None,
    };
    let ad_arr = match &ad {
        PyValue::ArrayF64(x) => match x.clone().into_dimensionality::<Ix2>() {
            Ok(x) => x,
            Err(_) => return PyValue::None,
        },
        _ => return PyValue::None,
    };
    let dp_arr = match &dp {
        PyValue::ArrayF64(x) => match x.clone().into_dimensionality::<Ix2>() {
            Ok(x) => x,
            Err(_) => return PyValue::None,
        },
        _ => return PyValue::None,
    };
    let n_vars_vec: Vec<f64> = dp_arr
        .mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
        .sum_axis(Axis(0))
        .iter()
        .copied()
        .collect();
    let donor_gpb_arr = match &donor_gpb {
        PyValue::ArrayF64(x) => match x.clone().into_dimensionality::<Ix3>() {
            Ok(x) => Some(x),
            Err(_) => return PyValue::None,
        },
        PyValue::None => None,
        _ => return PyValue::None,
    };
    let res = match vireo_wrap::vireo_wrap(
        &ad_arr,
        &dp_arr,
        donor_gpb_arr.as_ref(),
        Some(n_donor as usize),
        learn_gt,
        n_init as usize,
        rand_seed,
        !no_doublet,
        20,
        3,
        n_extra_donor as usize,
        Some(&extra_donor_mode),
        check_ambient,
        1,
        ase_mode,
        false,
        3,
    ) {
        Some(v) => v,
        None => return PyValue::None,
    };
    let cell_names = match cell_dat.get("samples") {
        Some(PyValue::StringVec(v)) => v.clone(),
        _ => (0..dp_arr.ncols()).map(|i| format!("cell{i}")).collect(),
    };
    io_utils::write_donor_id(&out_dir, &donor_names, &cell_names, &n_vars_vec, &res);
    if learn_gt && cell_dat.contains_key("variants") {
        if let (Some(gt_prob), Some(id_prob)) = (res.get("GT_prob"), res.get("ID_prob")) {
            let id = match id_prob {
                PyValue::ArrayF64(x) => match x.clone().into_dimensionality::<Ix2>() {
                    Ok(x) => x,
                    Err(_) => return PyValue::Map(res),
                },
                _ => return PyValue::Map(res),
            };
            let gt_prob = match gt_prob {
                PyValue::ArrayF64(x) => match x.clone().into_dimensionality::<ndarray::Ix3>() {
                    Ok(x) => x,
                    Err(_) => return PyValue::Map(res),
                },
                _ => return PyValue::Map(res),
            };
            let geno = match vcf_utils::GenoINFO_maker(&gt_prob, &ad_arr.dot(&id), &dp_arr.dot(&id))
            {
                Some(m) => PyValue::Map(
                    m.into_iter()
                        .map(|(k, v)| (k, PyValue::StringMatrix(v)))
                        .collect(),
                ),
                None => return PyValue::Map(res),
            };
            cell_dat.insert("samples".to_string(), PyValue::StringVec(donor_names));
            cell_dat.insert("GenoINFO".to_string(), geno);
            let out_vcf = format!("{out_dir}/GT_donors.vireo.vcf.gz");
            vcf_utils::write_VCF(&out_vcf, &cell_dat, None);
        }
    }
    PyValue::Map(res)
}
