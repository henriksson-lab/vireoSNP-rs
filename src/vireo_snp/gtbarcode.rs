use crate::vireo_snp::utils::variant_select;
use crate::vireo_snp::utils::vcf_utils;
use crate::PyValue;
use ndarray::Array2;
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

pub fn main() -> PyValue {
    let args: Vec<String> = env::args().collect();
    if args.len() <= 1 {
        return PyValue::None;
    }
    let mut vcf_file = None;
    let mut out_file = None;
    let mut geno_tag = "GT".to_string();
    let mut no_homo_alt = false;
    let mut rand_seed = PyValue::None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vcfFile" | "-i" => {
                i += 1;
                vcf_file = args.get(i).cloned();
            }
            "--outFile" | "-o" => {
                i += 1;
                out_file = args.get(i).cloned();
            }
            "--genoTag" | "-t" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    geno_tag = v.clone();
                }
            }
            "--noHomoAlt" => no_homo_alt = true,
            "--randSeed" => {
                i += 1;
                rand_seed = args
                    .get(i)
                    .and_then(|v| v.parse::<i64>().ok())
                    .map(PyValue::I64)
                    .unwrap_or(PyValue::None);
            }
            _ => {}
        }
        i += 1;
    }
    let Some(vcf_file) = vcf_file else {
        return PyValue::None;
    };
    let out_file = out_file.unwrap_or_else(|| {
        let parent = Path::new(&vcf_file)
            .parent()
            .unwrap_or_else(|| Path::new("."));
        parent.join("GTbarcode.tsv").to_string_lossy().into_owned()
    });
    if let Some(parent) = Path::new(&out_file).parent() {
        if !parent.as_os_str().is_empty() && fs::create_dir_all(parent).is_err() {
            return PyValue::None;
        }
    }
    let donor_vcf = match vcf_utils::load_VCF(&vcf_file, true, true, false, None) {
        Some(m) => m,
        None => return PyValue::None,
    };
    let geno = match donor_vcf.get("GenoINFO") {
        Some(PyValue::Map(m)) => match m.get(&geno_tag) {
            Some(PyValue::StringMatrix(v)) => v.clone(),
            _ => return PyValue::None,
        },
        _ => return PyValue::None,
    };
    let donor_gpb = match vcf_utils::parse_donor_GPb(&geno, &geno_tag, 0.0) {
        Some(x) => x,
        None => return PyValue::None,
    };
    let var_ids = match donor_vcf.get("variants") {
        Some(PyValue::StringVec(v)) => v.clone(),
        _ => return PyValue::None,
    };
    let sample_ids = match donor_vcf.get("samples") {
        Some(PyValue::StringVec(v)) => v.clone(),
        _ => return PyValue::None,
    };
    let info = match donor_vcf.get("FixedINFO") {
        Some(PyValue::Map(m)) => match m.get("INFO") {
            Some(PyValue::StringVec(v)) => v.clone(),
            _ => vec![String::new(); var_ids.len()],
        },
        _ => vec![String::new(); var_ids.len()],
    };
    let mut gt_vals = Array2::<f64>::zeros((donor_gpb.shape()[0], donor_gpb.shape()[1]));
    for v in 0..donor_gpb.shape()[0] {
        for d in 0..donor_gpb.shape()[1] {
            let best = (0..donor_gpb.shape()[2])
                .max_by(|&a, &b| donor_gpb[[v, d, a]].total_cmp(&donor_gpb[[v, d, b]]))
                .unwrap_or(0);
            gt_vals[[v, d]] = best as f64;
        }
    }
    let mut keep = Vec::new();
    let mut dp_values = Vec::new();
    for (idx, text) in info.iter().enumerate() {
        let mut dp = 0.0;
        let mut oth = 0.0;
        for field in text.split(';') {
            if let Some(v) = field.strip_prefix("DP=") {
                dp = v.parse().unwrap_or(0.0);
            } else if let Some(v) = field.strip_prefix("OTH=") {
                oth = v.parse().unwrap_or(0.0);
            }
        }
        let max_gt = gt_vals
            .row(idx)
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if dp > 20.0 && (dp == 0.0 || oth / dp < 0.05) && (!no_homo_alt || max_gt < 2.0) {
            keep.push(idx);
            dp_values.push(dp);
        }
    }
    let mut gt_use = Array2::<f64>::zeros((keep.len(), gt_vals.ncols()));
    let mut var_use = Vec::new();
    for (new_i, &old_i) in keep.iter().enumerate() {
        gt_use.row_mut(new_i).assign(&gt_vals.row(old_i));
        var_use.push(var_ids[old_i].clone());
    }
    let seed = match rand_seed {
        PyValue::I64(v) => v as u64,
        PyValue::None => 0,
        _ => return PyValue::None,
    };
    let variant_set = match variant_select::variant_select(&gt_use, Some(&dp_values), seed) {
        Some((_, _, v)) => v,
        None => return PyValue::None,
    };
    let mut f = match File::create(out_file) {
        Ok(f) => f,
        Err(_) => return PyValue::None,
    };
    let mut header = vec!["variants".to_string()];
    header.extend(sample_ids);
    if writeln!(f, "{}", header.join("\t")).is_err() {
        return PyValue::None;
    }
    for i in variant_set {
        let values: Vec<String> = gt_use.row(i).iter().map(|v| format!("{v:.0}")).collect();
        if writeln!(f, "{}\t{}", var_use[i], values.join("\t")).is_err() {
            return PyValue::None;
        }
    }
    PyValue::None
}
