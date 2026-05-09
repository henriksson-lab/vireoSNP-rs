use crate::vireo_snp::utils::vcf_utils;
use crate::PyValue;
use ndarray::{Array2, Ix2};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

pub fn match_donor_VCF(
    mut cell_dat: BTreeMap<String, PyValue>,
    mut donor_vcf: BTreeMap<String, PyValue>,
) -> Option<(BTreeMap<String, PyValue>, BTreeMap<String, PyValue>)> {
    let cell_variants = match cell_dat.get("variants") {
        Some(PyValue::StringVec(v)) => v.clone(),
        _ => return None,
    };
    let donor_variants = match donor_vcf.get("variants") {
        Some(PyValue::StringVec(v)) => v.clone(),
        _ => return None,
    };
    let pairs: Vec<(usize, usize)> = vcf_utils::match_SNPs(&cell_variants, &donor_variants)
        .into_iter()
        .enumerate()
        .filter_map(|(i, v)| v.map(|j| (i, j)))
        .collect();
    let idx1: Vec<usize> = pairs.iter().map(|(i, _)| *i).collect();
    let idx2: Vec<usize> = pairs.iter().map(|(_, j)| *j).collect();
    let subset_string_vec = |values: &[String], idx: &[usize]| -> Vec<String> {
        idx.iter().map(|&i| values[i].clone()).collect()
    };
    cell_dat.insert(
        "variants".to_string(),
        PyValue::StringVec(subset_string_vec(&cell_variants, &idx1)),
    );
    donor_vcf.insert(
        "variants".to_string(),
        PyValue::StringVec(subset_string_vec(&donor_variants, &idx2)),
    );
    for key in ["AD", "DP"] {
        if let Some(PyValue::ArrayF64(arr)) = cell_dat.get(key).cloned() {
            if arr.ndim() == 2 {
                let arr2 = match arr.into_dimensionality::<ndarray::Ix2>() {
                    Ok(a) => a,
                    Err(_) => return None,
                };
                let mut out = ndarray::Array2::<f64>::zeros((idx1.len(), arr2.ncols()));
                for (new_i, &old_i) in idx1.iter().enumerate() {
                    out.row_mut(new_i).assign(&arr2.row(old_i));
                }
                cell_dat.insert(key.to_string(), PyValue::ArrayF64(out.into_dyn()));
            }
        }
    }
    for (dat, idx) in [(&mut cell_dat, &idx1), (&mut donor_vcf, &idx2)] {
        if let Some(PyValue::Map(mut fixed)) = dat.get("FixedINFO").cloned() {
            for value in fixed.values_mut() {
                if let PyValue::StringVec(values) = value {
                    *values = subset_string_vec(values, idx);
                }
            }
            dat.insert("FixedINFO".to_string(), PyValue::Map(fixed));
        }
    }
    if let Some(PyValue::Map(mut geno)) = donor_vcf.get("GenoINFO").cloned() {
        for value in geno.values_mut() {
            if let PyValue::StringMatrix(rows) = value {
                *rows = idx2.iter().map(|&i| rows[i].clone()).collect();
            }
        }
        donor_vcf.insert("GenoINFO".to_string(), PyValue::Map(geno));
    }
    Some((cell_dat, donor_vcf))
}

pub fn read_cellSNP(
    dir_name: &str,
    layers: Option<&[String]>,
) -> Option<BTreeMap<String, PyValue>> {
    let layers = layers
        .map(|x| x.to_vec())
        .unwrap_or_else(|| vec!["AD".to_string(), "DP".to_string()]);
    let vcf_path = format!("{dir_name}/cellSNP.base.vcf.gz");
    let mut cell_dat = vcf_utils::load_VCF(&vcf_path, false, false, true, None)?;
    for layer in layers {
        let path = format!("{dir_name}/cellSNP.tag.{layer}.mtx");
        let file = match File::open(path) {
            Ok(file) => file,
            Err(_) => return None,
        };
        let mut dims = None;
        let mut entries = Vec::new();
        for line in BufReader::new(file).lines().map_while(Result::ok) {
            let line = line.trim();
            if line.is_empty() || line.starts_with('%') {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if dims.is_none() {
                if parts.len() < 3 {
                    return None;
                }
                let rows = match parts[0].parse::<usize>() {
                    Ok(v) => v,
                    Err(_) => return None,
                };
                let cols = match parts[1].parse::<usize>() {
                    Ok(v) => v,
                    Err(_) => return None,
                };
                dims = Some((rows, cols));
            } else {
                if parts.len() < 3 {
                    return None;
                }
                let row = match parts[0].parse::<usize>() {
                    Ok(v) => v - 1,
                    Err(_) => return None,
                };
                let col = match parts[1].parse::<usize>() {
                    Ok(v) => v - 1,
                    Err(_) => return None,
                };
                let val = match parts[2].parse::<f64>() {
                    Ok(v) => v,
                    Err(_) => return None,
                };
                entries.push((row, col, val));
            }
        }
        let (rows, cols) = match dims {
            Some(v) => v,
            None => return None,
        };
        let mut mat = Array2::<f64>::zeros((rows, cols));
        for (row, col, val) in entries {
            mat[[row, col]] = val;
        }
        cell_dat.insert(layer, PyValue::ArrayF64(mat.into_dyn()));
    }
    let samples_file = match File::open(format!("{dir_name}/cellSNP.samples.tsv")) {
        Ok(file) => file,
        Err(_) => return None,
    };
    let samples: Vec<String> = BufReader::new(samples_file)
        .lines()
        .map_while(Result::ok)
        .collect();
    cell_dat.insert("samples".to_string(), PyValue::StringVec(samples));
    Some(cell_dat)
}

pub fn read_vartrix(
    alt_mtx: &str,
    ref_mtx: &str,
    cell_file: &str,
    vcf_file: Option<&str>,
) -> Option<BTreeMap<String, PyValue>> {
    let mut cell_dat = match vcf_file {
        Some(v) => vcf_utils::load_VCF(v, false, false, true, None)?,
        None => BTreeMap::new(),
    };
    let file = match File::open(alt_mtx) {
        Ok(file) => file,
        Err(_) => return None,
    };
    let mut dims = None;
    let mut entries = Vec::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let line = line.trim();
        if line.is_empty() || line.starts_with('%') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if dims.is_none() {
            if parts.len() < 3 {
                return None;
            }
            let rows = match parts[0].parse::<usize>() {
                Ok(v) => v,
                Err(_) => return None,
            };
            let cols = match parts[1].parse::<usize>() {
                Ok(v) => v,
                Err(_) => return None,
            };
            dims = Some((rows, cols));
        } else {
            if parts.len() < 3 {
                return None;
            }
            let row = match parts[0].parse::<usize>() {
                Ok(v) => v - 1,
                Err(_) => return None,
            };
            let col = match parts[1].parse::<usize>() {
                Ok(v) => v - 1,
                Err(_) => return None,
            };
            let val = match parts[2].parse::<f64>() {
                Ok(v) => v,
                Err(_) => return None,
            };
            entries.push((row, col, val));
        }
    }
    let (rows, cols) = match dims {
        Some(v) => v,
        None => return None,
    };
    let mut ad = Array2::<f64>::zeros((rows, cols));
    for (row, col, val) in entries {
        ad[[row, col]] = val;
    }
    let file = match File::open(ref_mtx) {
        Ok(file) => file,
        Err(_) => return None,
    };
    let mut dims = None;
    let mut entries = Vec::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let line = line.trim();
        if line.is_empty() || line.starts_with('%') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if dims.is_none() {
            if parts.len() < 3 {
                return None;
            }
            let rows = match parts[0].parse::<usize>() {
                Ok(v) => v,
                Err(_) => return None,
            };
            let cols = match parts[1].parse::<usize>() {
                Ok(v) => v,
                Err(_) => return None,
            };
            dims = Some((rows, cols));
        } else {
            if parts.len() < 3 {
                return None;
            }
            let row = match parts[0].parse::<usize>() {
                Ok(v) => v - 1,
                Err(_) => return None,
            };
            let col = match parts[1].parse::<usize>() {
                Ok(v) => v - 1,
                Err(_) => return None,
            };
            let val = match parts[2].parse::<f64>() {
                Ok(v) => v,
                Err(_) => return None,
            };
            entries.push((row, col, val));
        }
    }
    let (ref_rows, ref_cols) = match dims {
        Some(v) => v,
        None => return None,
    };
    if ref_rows != rows || ref_cols != cols {
        return None;
    }
    let mut dp = ad.clone();
    for (row, col, val) in entries {
        dp[[row, col]] += val;
    }
    let samples_file = match File::open(cell_file) {
        Ok(file) => file,
        Err(_) => return None,
    };
    let samples: Vec<String> = BufReader::new(samples_file)
        .lines()
        .map_while(Result::ok)
        .collect();
    cell_dat.insert("AD".to_string(), PyValue::ArrayF64(ad.into_dyn()));
    cell_dat.insert("DP".to_string(), PyValue::ArrayF64(dp.into_dyn()));
    cell_dat.insert("samples".to_string(), PyValue::StringVec(samples));
    Some(cell_dat)
}

pub fn write_donor_id(
    out_dir: &str,
    donor_names: &[String],
    cell_names: &[String],
    n_vars: &[f64],
    res: &BTreeMap<String, PyValue>,
) -> Option<()> {
    let id_prob = match res.get("ID_prob") {
        Some(PyValue::ArrayF64(x)) => match x.clone().into_dimensionality::<Ix2>() {
            Ok(x) => x,
            Err(_) => return None,
        },
        _ => return None,
    };
    let doublet_prob = match res.get("doublet_prob") {
        Some(PyValue::ArrayF64(x)) => match x.clone().into_dimensionality::<Ix2>() {
            Ok(x) => x,
            Err(_) => return None,
        },
        _ => return None,
    };
    let doublet_llr: Vec<f64> = match res.get("doublet_LLR") {
        Some(PyValue::ArrayF64(x)) => x.iter().copied().collect(),
        Some(PyValue::F64Vec(v)) => v.clone(),
        _ => vec![0.0; id_prob.nrows()],
    };
    let lb_doublet = match res.get("LB_doublet") {
        Some(PyValue::F64(v)) => *v,
        Some(PyValue::F64Vec(v)) => *v.last().unwrap_or(&0.0),
        _ => 0.0,
    };
    let theta_shapes = match res.get("theta_shapes") {
        Some(PyValue::ArrayF64(x)) => format!("{:?}", x),
        other => format!("{:?}", other),
    };
    let mut doublet_names = Vec::new();
    for i in 0..donor_names.len() {
        for j in (i + 1)..donor_names.len() {
            doublet_names.push(format!("{},{}", donor_names[i], donor_names[j]));
        }
    }
    let mut donor_singlet = Vec::new();
    let mut donor_doublet = Vec::new();
    let mut prob_max = Vec::new();
    let mut prob_doublet_out = Vec::new();
    let mut donor_ids = Vec::new();
    for i in 0..id_prob.nrows() {
        let best_s = (0..id_prob.ncols())
            .max_by(|&a, &b| id_prob[[i, a]].total_cmp(&id_prob[[i, b]]))
            .unwrap_or(0);
        let best_d = if doublet_prob.ncols() > 0 {
            (0..doublet_prob.ncols())
                .max_by(|&a, &b| doublet_prob[[i, a]].total_cmp(&doublet_prob[[i, b]]))
                .unwrap_or(0)
        } else {
            0
        };
        let pmax = id_prob[[i, best_s]];
        let pdbl = if doublet_prob.ncols() > 0 {
            doublet_prob[[i, best_d]]
        } else {
            0.0
        };
        prob_max.push(pmax);
        prob_doublet_out.push(pdbl);
        donor_singlet.push(donor_names[best_s].clone());
        donor_doublet.push(doublet_names.get(best_d).cloned().unwrap_or_default());
        let mut donor_id = donor_names[best_s].clone();
        if pmax < 0.9 {
            donor_id = "unassigned".to_string();
        }
        if pdbl >= 0.9 {
            donor_id = "doublet".to_string();
        }
        if n_vars.get(i).copied().unwrap_or(0.0) < 10.0 {
            donor_id = "unassigned".to_string();
        }
        donor_ids.push(donor_id);
    }
    let mut f = match File::create(format!("{out_dir}/_log.txt")) {
        Ok(f) => f,
        Err(_) => return None,
    };
    if writeln!(f, "logLik: {:.3e}", lb_doublet).is_err()
        || writeln!(f, "thetas: \n{}", theta_shapes).is_err()
    {
        return None;
    }
    let mut counts = BTreeMap::<String, usize>::new();
    for id in &donor_ids {
        *counts.entry(id.clone()).or_insert(0) += 1;
    }
    let mut f = match File::create(format!("{out_dir}/summary.tsv")) {
        Ok(f) => f,
        Err(_) => return None,
    };
    if writeln!(f, "Var1\tFreq").is_err() {
        return None;
    }
    for (id, count) in counts {
        if writeln!(f, "{id}\t{count}").is_err() {
            return None;
        }
    }
    let mut f = match File::create(format!("{out_dir}/donor_ids.tsv")) {
        Ok(f) => f,
        Err(_) => return None,
    };
    let header = [
        "cell",
        "donor_id",
        "prob_max",
        "prob_doublet",
        "n_vars",
        "best_singlet",
        "best_doublet",
        "doublet_logLikRatio",
    ];
    if writeln!(f, "{}", header.join("\t")).is_err() {
        return None;
    }
    for i in 0..cell_names.len() {
        if writeln!(
            f,
            "{}\t{}\t{:.2e}\t{:.2e}\t{:.0}\t{}\t{}\t{:.3}",
            cell_names[i],
            donor_ids[i],
            prob_max[i],
            prob_doublet_out[i],
            n_vars.get(i).copied().unwrap_or(0.0),
            donor_singlet[i],
            donor_doublet[i],
            doublet_llr.get(i).copied().unwrap_or(0.0)
        )
        .is_err()
        {
            return None;
        }
    }
    let mut f = match File::create(format!("{out_dir}/prob_singlet.tsv")) {
        Ok(f) => f,
        Err(_) => return None,
    };
    if writeln!(
        f,
        "{}",
        ["cell"]
            .into_iter()
            .chain(donor_names.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join("\t")
    )
    .is_err()
    {
        return None;
    }
    for i in 0..cell_names.len() {
        let vals: Vec<String> = (0..id_prob.ncols())
            .map(|j| format!("{:.2e}", id_prob[[i, j]]))
            .collect();
        if writeln!(f, "{}\t{}", cell_names[i], vals.join("\t")).is_err() {
            return None;
        }
    }
    let mut f = match File::create(format!("{out_dir}/prob_doublet.tsv")) {
        Ok(f) => f,
        Err(_) => return None,
    };
    if writeln!(
        f,
        "{}",
        ["cell"]
            .into_iter()
            .chain(doublet_names.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join("\t")
    )
    .is_err()
    {
        return None;
    }
    for i in 0..cell_names.len() {
        let vals: Vec<String> = (0..doublet_prob.ncols())
            .map(|j| format!("{:.2e}", doublet_prob[[i, j]]))
            .collect();
        if writeln!(f, "{}\t{}", cell_names[i], vals.join("\t")).is_err() {
            return None;
        }
    }
    if let Some(PyValue::ArrayF64(ambient)) = res.get("ambient_Psi") {
        if let Ok(ambient) = ambient.clone().into_dimensionality::<Ix2>() {
            let llr: Vec<f64> = match res.get("Psi_LLRatio") {
                Some(PyValue::ArrayF64(x)) => x.iter().copied().collect(),
                Some(PyValue::F64Vec(v)) => v.clone(),
                _ => vec![0.0; ambient.nrows()],
            };
            let mut f = match File::create(format!("{out_dir}/prop_ambient.tsv")) {
                Ok(f) => f,
                Err(_) => return None,
            };
            let mut header = vec!["cell".to_string()];
            header.extend(donor_names.iter().cloned());
            header.push("logLik_ratio".to_string());
            if writeln!(f, "{}", header.join("\t")).is_err() {
                return None;
            }
            for i in 0..cell_names.len() {
                let mut vals: Vec<String> = (0..ambient.ncols())
                    .map(|j| format!("{:.4e}", ambient[[i, j]]))
                    .collect();
                vals.push(format!("{:.2}", llr.get(i).copied().unwrap_or(0.0)));
                if writeln!(f, "{}\t{}", cell_names[i], vals.join("\t")).is_err() {
                    return None;
                }
            }
        }
    }
    Some(())
}

pub fn make_whitelists(donor_id_file: &str, out_prefix: &str) -> Option<()> {
    let file = match File::open(donor_id_file) {
        Ok(file) => file,
        Err(_) => return None,
    };
    let mut by_donor: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (i, line) in BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .enumerate()
    {
        if i == 0 {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 2 || cols[1] == "unassigned" || cols[1] == "doublet" {
            continue;
        }
        by_donor
            .entry(cols[1].to_string())
            .or_default()
            .push(cols[0].to_string());
    }
    for (donor, barcodes) in by_donor {
        let mut out = match File::create(format!("{out_prefix}_{donor}.txt")) {
            Ok(file) => file,
            Err(_) => return None,
        };
        for barcode in barcodes {
            let prefix = barcode.split('-').next().unwrap_or(&barcode);
            if writeln!(out, "{prefix}").is_err() {
                return None;
            }
        }
    }
    Some(())
}
