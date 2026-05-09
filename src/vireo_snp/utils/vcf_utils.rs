use crate::PyValue;
use flate2::read::MultiGzDecoder;
use ndarray::{Array2, Array3};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};

pub fn parse_sample_info(
    sample_dat: &[Vec<String>],
    sparse: bool,
    format_list: Option<&[String]>,
) -> Option<(BTreeMap<String, PyValue>, Vec<i64>)> {
    if sample_dat.is_empty() {
        return None;
    }
    let format_all: Vec<Vec<String>> = sample_dat
        .iter()
        .map(|x| x[0].split(':').map(|s| s.to_string()).collect())
        .collect();
    let format_list = format_list
        .map(|x| x.to_vec())
        .unwrap_or_else(|| format_all[0].clone());
    let mut rv = BTreeMap::new();
    let mut n_snp_tagged = vec![0i64; format_list.len()];
    for key in &format_list {
        rv.insert(key.clone(), PyValue::StringVec(Vec::new()));
    }
    if sparse {
        if format_all.iter().any(|x| x != &format_list) {
            return None;
        }
        let mut indices = Vec::new();
        let mut indptr = vec![0i64];
        let mut cnt = 0i64;
        let missing_val = vec!["."; format_list.len()].join(":");
        for line in sample_dat {
            for (i, cell) in line.iter().skip(1).enumerate() {
                if cell == &missing_val || cell == "." {
                    continue;
                }
                let line_key: Vec<&str> = cell.split(':').collect();
                for (k, key) in format_list.iter().enumerate() {
                    if let Some(PyValue::StringVec(values)) = rv.get_mut(key) {
                        values.push(line_key[k].to_string());
                    }
                    n_snp_tagged[k] += 1;
                }
                cnt += 1;
                indices.push(i as i64);
            }
            indptr.push(cnt);
        }
        rv.insert("indices".to_string(), PyValue::I64Vec(indices));
        rv.insert("indptr".to_string(), PyValue::I64Vec(indptr));
        rv.insert(
            "shape".to_string(),
            PyValue::I64Vec(vec![
                (sample_dat[0].len() - 1) as i64,
                sample_dat.len() as i64,
            ]),
        );
    } else {
        for (j, line) in sample_dat.iter().enumerate() {
            let line_split: Vec<Vec<&str>> = line
                .iter()
                .skip(1)
                .map(|x| x.split(':').collect())
                .collect();
            for (il, key) in format_list.iter().enumerate() {
                let values = if let Some(k) = format_all[j].iter().position(|x| x == key) {
                    n_snp_tagged[il] += 1;
                    line_split.iter().map(|x| x[k].to_string()).collect()
                } else {
                    vec![".".to_string(); line_split.len()]
                };
                if let Some(PyValue::StringMatrix(rows)) = rv.get_mut(key) {
                    rows.push(values);
                } else {
                    rv.insert(key.clone(), PyValue::StringMatrix(vec![values]));
                }
            }
        }
    }
    Some((rv, n_snp_tagged))
}

pub fn load_VCF(
    path: &str,
    biallelic_only: bool,
    load_sample: bool,
    sparse: bool,
    format_list: Option<&[String]>,
) -> Option<BTreeMap<String, PyValue>> {
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(_) => return None,
    };
    let reader: Box<dyn Read> = if path.ends_with(".gz") || path.ends_with(".bgz") {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut fixed_info: BTreeMap<String, PyValue> = BTreeMap::new();
    let mut contig_lines = Vec::new();
    let mut comment_lines = Vec::new();
    let mut var_ids = Vec::new();
    let mut obs_ids = Vec::new();
    let mut obs_dat = Vec::new();
    let mut key_ids = Vec::<String>::new();
    for line in BufReader::new(reader).lines().map_while(Result::ok) {
        if line.starts_with('#') {
            if line.starts_with("##contig=") {
                contig_lines.push(line.clone());
            }
            if line.starts_with("#CHROM") {
                let parts: Vec<String> =
                    line.trim_end().split('\t').map(|s| s.to_string()).collect();
                if load_sample {
                    obs_ids = parts.iter().skip(9).cloned().collect();
                }
                key_ids = parts
                    .iter()
                    .take(8)
                    .map(|s| s.trim_start_matches('#').to_string())
                    .collect();
                for key in &key_ids {
                    fixed_info.insert(key.clone(), PyValue::StringVec(Vec::new()));
                }
            } else {
                comment_lines.push(line);
            }
        } else {
            let list_val: Vec<String> =
                line.trim_end().split('\t').map(|s| s.to_string()).collect();
            if biallelic_only && (list_val[3].len() > 1 || list_val[4].len() > 1) {
                continue;
            }
            if load_sample {
                obs_dat.push(list_val.iter().skip(8).cloned().collect());
            }
            for (i, key) in key_ids.iter().enumerate() {
                if let Some(PyValue::StringVec(values)) = fixed_info.get_mut(key) {
                    values.push(list_val[i].clone());
                }
            }
            var_ids.push(
                [0usize, 1, 3, 4]
                    .iter()
                    .map(|&i| list_val[i].clone())
                    .collect::<Vec<_>>()
                    .join("_"),
            );
        }
    }
    let mut rv = BTreeMap::new();
    rv.insert("variants".to_string(), PyValue::StringVec(var_ids));
    rv.insert("FixedINFO".to_string(), PyValue::Map(fixed_info));
    rv.insert("contigs".to_string(), PyValue::StringVec(contig_lines));
    rv.insert("comments".to_string(), PyValue::StringVec(comment_lines));
    if load_sample {
        rv.insert("samples".to_string(), PyValue::StringVec(obs_ids));
        let (geno_info, n_snp_tagged) = parse_sample_info(&obs_dat, sparse, format_list)?;
        rv.insert("GenoINFO".to_string(), PyValue::Map(geno_info));
        rv.insert("n_SNP_tagged".to_string(), PyValue::I64Vec(n_snp_tagged));
    }
    Some(rv)
}

pub fn write_VCF_to_hdf5(vcf_dat: &BTreeMap<String, PyValue>, out_file: &str) -> Option<()> {
    let file = match hdf5::File::create(out_file) {
        Ok(f) => f,
        Err(_) => return None,
    };
    for key in ["contigs", "samples", "variants", "comments"] {
        if let Some(PyValue::StringVec(values)) = vcf_dat.get(key) {
            let data: Vec<hdf5::types::VarLenUnicode> = values
                .iter()
                .filter_map(|s| s.parse::<hdf5::types::VarLenUnicode>().ok())
                .collect();
            if file
                .new_dataset_builder()
                .with_data(&data)
                .create(key)
                .is_err()
            {
                return None;
            }
        }
    }
    if let Some(PyValue::Map(fixed_info)) = vcf_dat.get("FixedINFO") {
        let group = match file.create_group("FixedINFO") {
            Ok(g) => g,
            Err(_) => return None,
        };
        for (key, value) in fixed_info {
            if let PyValue::StringVec(values) = value {
                let data: Vec<hdf5::types::VarLenUnicode> = values
                    .iter()
                    .filter_map(|s| s.parse::<hdf5::types::VarLenUnicode>().ok())
                    .collect();
                if group
                    .new_dataset_builder()
                    .with_data(&data)
                    .create(key.as_str())
                    .is_err()
                {
                    return None;
                }
            }
        }
    }
    if let Some(PyValue::Map(geno_info)) = vcf_dat.get("GenoINFO") {
        let group = match file.create_group("GenoINFO") {
            Ok(g) => g,
            Err(_) => return None,
        };
        for (key, value) in geno_info {
            match value {
                PyValue::StringVec(values) => {
                    let data: Vec<hdf5::types::VarLenUnicode> = values
                        .iter()
                        .filter_map(|s| s.parse::<hdf5::types::VarLenUnicode>().ok())
                        .collect();
                    if group
                        .new_dataset_builder()
                        .with_data(&data)
                        .create(key.as_str())
                        .is_err()
                    {
                        return None;
                    }
                }
                PyValue::StringMatrix(rows) => {
                    let flat: Vec<hdf5::types::VarLenUnicode> = rows
                        .iter()
                        .flatten()
                        .filter_map(|s| s.parse::<hdf5::types::VarLenUnicode>().ok())
                        .collect();
                    let shape = (rows.len(), rows.first().map_or(0, Vec::len));
                    if group
                        .new_dataset_builder()
                        .with_data(&flat)
                        .create(key.as_str())
                        .is_err()
                    {
                        return None;
                    }
                    let shape_key = format!("{key}_shape");
                    let shape_data = [shape.0 as i64, shape.1 as i64];
                    if group
                        .new_dataset_builder()
                        .with_data(&shape_data)
                        .create(shape_key.as_str())
                        .is_err()
                    {
                        return None;
                    }
                }
                PyValue::I64Vec(values) => {
                    if group
                        .new_dataset_builder()
                        .with_data(values)
                        .create(key.as_str())
                        .is_err()
                    {
                        return None;
                    }
                }
                _ => {}
            }
        }
    }
    Some(())
}

pub fn read_sparse_GeneINFO(
    geno_info: &BTreeMap<String, PyValue>,
    keys: Option<&[String]>,
    axes: Option<&[i64]>,
) -> Option<BTreeMap<String, Array2<f64>>> {
    let keys = keys
        .map(|x| x.to_vec())
        .unwrap_or_else(|| vec!["AD".to_string(), "DP".to_string()]);
    let axes = axes
        .map(|x| x.to_vec())
        .unwrap_or_else(|| vec![-1; keys.len()]);
    let shape = match geno_info.get("shape") {
        Some(PyValue::I64Vec(v)) if v.len() == 2 => (v[0] as usize, v[1] as usize),
        _ => return None,
    };
    let indptr = match geno_info.get("indptr") {
        Some(PyValue::I64Vec(v)) => v,
        _ => return None,
    };
    let indices = match geno_info.get("indices") {
        Some(PyValue::I64Vec(v)) => v,
        _ => return None,
    };
    let mut rv = BTreeMap::new();
    for (ki, key) in keys.iter().enumerate() {
        let values = match geno_info.get(key) {
            Some(PyValue::StringVec(v)) => v,
            _ => return None,
        };
        let mut mat = ndarray::Array2::<f64>::zeros((shape.1, shape.0));
        for row in 0..shape.1 {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            for p in start..end {
                let col = indices[p] as usize;
                let parts: Vec<&str> = values[p].split(',').collect();
                let axis = axes.get(ki).copied().unwrap_or(-1);
                let idx = if axis < 0 {
                    parts.len().saturating_sub((-axis) as usize)
                } else {
                    axis as usize
                };
                mat[[row, col]] = parts
                    .get(idx)
                    .copied()
                    .unwrap_or("0")
                    .parse::<f64>()
                    .unwrap_or(0.0);
            }
        }
        rv.insert(key.clone(), mat);
    }
    Some(rv)
}

pub fn GenoINFO_maker(
    gt_prob: &Array3<f64>,
    ad_reads: &ndarray::Array2<f64>,
    dp_reads: &ndarray::Array2<f64>,
) -> Option<BTreeMap<String, Vec<Vec<String>>>> {
    if gt_prob.shape()[0] != ad_reads.nrows()
        || gt_prob.shape()[1] != ad_reads.ncols()
        || ad_reads.raw_dim() != dp_reads.raw_dim()
    {
        return None;
    }
    let mut gt_prob = gt_prob.clone();
    gt_prob.mapv_inplace(|v| v.max(1e-10));
    let labels = ["0/0", "1/0", "1/1"];
    let mut gt = Vec::new();
    let mut pl = Vec::new();
    let mut ad = Vec::new();
    let mut dp = Vec::new();
    for i in 0..gt_prob.shape()[0] {
        let mut gt_row = Vec::new();
        let mut pl_row = Vec::new();
        let mut ad_row = Vec::new();
        let mut dp_row = Vec::new();
        for j in 0..gt_prob.shape()[1] {
            let mut best = 0usize;
            for g in 1..gt_prob.shape()[2] {
                if gt_prob[[i, j, g]] > gt_prob[[i, j, best]] {
                    best = g;
                }
            }
            gt_row.push(labels[best].to_string());
            pl_row.push(
                (0..gt_prob.shape()[2])
                    .map(|g| (-10.0 * gt_prob[[i, j, g]].log10()).round().to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
            ad_row.push(ad_reads[[i, j]].round().to_string());
            dp_row.push(dp_reads[[i, j]].round().to_string());
        }
        gt.push(gt_row);
        pl.push(pl_row);
        ad.push(ad_row);
        dp.push(dp_row);
    }
    let mut rv = BTreeMap::new();
    rv.insert("GT".to_string(), gt);
    rv.insert("AD".to_string(), ad);
    rv.insert("DP".to_string(), dp);
    rv.insert("PL".to_string(), pl);
    Some(rv)
}

pub fn write_VCF(
    out_file: &str,
    vcf_dat: &BTreeMap<String, PyValue>,
    geno_tags: Option<&[String]>,
) -> Option<()> {
    let geno_tags = geno_tags
        .map(|x| x.to_vec())
        .unwrap_or_else(|| vec!["GT".into(), "AD".into(), "DP".into(), "PL".into()]);
    let out_file_use = out_file
        .strip_suffix(".gz")
        .unwrap_or(&out_file)
        .to_string();
    let mut f = match File::create(&out_file_use) {
        Ok(f) => f,
        Err(_) => return None,
    };
    if let Some(PyValue::StringVec(comments)) = vcf_dat.get("comments") {
        for line in comments {
            let mut tag_found = false;
            if line.starts_with("##FORMAT=<ID=") {
                for tag in &geno_tags {
                    if line.starts_with(&format!("##FORMAT=<ID={tag}")) {
                        tag_found = true;
                    }
                }
            }
            if !tag_found && writeln!(f, "{line}").is_err() {
                return None;
            }
        }
    }
    for tag in &geno_tags {
        let line = match tag.as_str() {
            "GT" => "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
            "AD" => "##FORMAT=<ID=AD,Number=1,Type=Integer,Description=\"Read depth for each allele\">",
            "DP" => "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">",
            "PL" => "##FORMAT=<ID=PL,Number=G,Type=Integer,Description=\"Phred-scaled genotype likelihoods\">",
            _ => continue,
        };
        if writeln!(f, "{line}").is_err() {
            return None;
        }
    }
    let samples = match vcf_dat.get("samples") {
        Some(PyValue::StringVec(v)) => v.clone(),
        None => Vec::new(),
        _ => return None,
    };
    let geno_tags = if samples.is_empty() {
        Vec::new()
    } else {
        geno_tags
    };
    let mut columns = vec![
        "CHROM".to_string(),
        "POS".to_string(),
        "ID".to_string(),
        "REF".to_string(),
        "ALT".to_string(),
        "QUAL".to_string(),
        "FILTER".to_string(),
        "INFO".to_string(),
    ];
    if !geno_tags.is_empty() {
        columns.push("FORMAT".to_string());
        columns.extend(samples.iter().cloned());
    }
    if writeln!(f, "#{}", columns.join("\t")).is_err() {
        return None;
    }
    let variants = match vcf_dat.get("variants") {
        Some(PyValue::StringVec(v)) => v,
        _ => return None,
    };
    let fixed_info = match vcf_dat.get("FixedINFO") {
        Some(PyValue::Map(m)) => m,
        _ => return None,
    };
    let geno_info = match vcf_dat.get("GenoINFO") {
        Some(PyValue::Map(m)) => Some(m),
        None => None,
        _ => return None,
    };
    let fixed_cols = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"];
    for i in 0..variants.len() {
        let mut line = Vec::new();
        for col in fixed_cols {
            let values = match fixed_info.get(col) {
                Some(PyValue::StringVec(v)) => v,
                _ => return None,
            };
            let Some(value) = values.get(i) else {
                return None;
            };
            line.push(value.clone());
        }
        if !geno_tags.is_empty() {
            let Some(geno_info) = geno_info else {
                return None;
            };
            line.push(geno_tags.join(":"));
            for s in 0..samples.len() {
                let mut values = Vec::new();
                for tag in &geno_tags {
                    let rows = match geno_info.get(tag) {
                        Some(PyValue::StringMatrix(v)) => v,
                        _ => return None,
                    };
                    let Some(row) = rows.get(i) else {
                        return None;
                    };
                    let Some(value) = row.get(s) else {
                        return None;
                    };
                    values.push(value.clone());
                }
                line.push(values.join(":"));
            }
        }
        if writeln!(f, "{}", line.join("\t")).is_err() {
            return None;
        }
    }
    drop(f);
    if out_file.ends_with(".gz") {
        let gzip_ok = std::process::Command::new("gzip")
            .arg("-f")
            .arg(&out_file_use)
            .status()
            .map(|status| status.success())
            .unwrap_or(false);
        if !gzip_ok {
            return None;
        }
    }
    Some(())
}

pub fn parse_donor_GPb(gt_dat: &[Vec<String>], tag: &str, min_prob: f64) -> Option<Array3<f64>> {
    if !matches!(tag, "GT" | "GP" | "PL") {
        return None;
    }
    if gt_dat.is_empty() {
        return Some(Array3::<f64>::zeros((0, 0, 3)));
    }
    let n_sample = gt_dat[0].len();
    if gt_dat.iter().any(|row| row.len() != n_sample) {
        return None;
    }
    let mut gt_prob = Array3::<f64>::zeros((gt_dat.len(), n_sample, 3));
    for i in 0..gt_dat.len() {
        for j in 0..gt_dat[i].len() {
            let prob = parse_GT_code(&gt_dat[i][j], tag)?;
            for g in 0..3 {
                gt_prob[[i, j, g]] = prob[g] + min_prob;
            }
            let sum: f64 = (0..3).map(|g| gt_prob[[i, j, g]]).sum();
            if sum == 0.0 {
                return None;
            }
            for g in 0..3 {
                gt_prob[[i, j, g]] /= sum;
            }
        }
    }
    Some(gt_prob)
}

pub fn parse_GT_code(code: &str, tag: &str) -> Option<[f64; 3]> {
    if code == "." || code == "./." || code == ".|." {
        return Some([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    }
    match tag {
        "GT" => {
            let a = code
                .chars()
                .next()
                .and_then(|c| c.to_digit(10))
                .unwrap_or(0);
            let b = code
                .chars()
                .last()
                .and_then(|c| c.to_digit(10))
                .unwrap_or(0);
            let mut prob = [0.0, 0.0, 0.0];
            let idx = (a + b) as usize;
            if idx >= prob.len() {
                return None;
            }
            prob[idx] = 1.0;
            Some(prob)
        }
        "GP" => {
            let prob: Vec<f64> = code
                .split(',')
                .filter_map(|x| x.parse::<f64>().ok())
                .collect();
            if prob.len() == 3 {
                Some([prob[0], prob[1], prob[2]])
            } else {
                None
            }
        }
        "PL" => {
            let phred: Vec<f64> = code
                .split(',')
                .filter_map(|x| x.parse::<f64>().ok())
                .collect();
            if phred.len() != 3 {
                return None;
            }
            let min = phred.iter().copied().fold(f64::INFINITY, f64::min);
            Some([
                10f64.powf(-0.1 * (phred[0] - min) - 0.025),
                10f64.powf(-0.1 * (phred[1] - min) - 0.025),
                10f64.powf(-0.1 * (phred[2] - min) - 0.025),
            ])
        }
        _ => None,
    }
}

pub fn match_SNPs(snp_ids1: &[String], snps_ids2: &[String]) -> Vec<Option<usize>> {
    let mut out = Vec::with_capacity(snp_ids1.len());
    for id1 in snp_ids1 {
        out.push(snps_ids2.iter().position(|id2| id2 == id1));
    }
    if out.iter().all(Option::is_none) {
        out.clear();
        for id1 in snp_ids1 {
            let id1_chr = format!("chr{id1}");
            out.push(snps_ids2.iter().position(|id2| id2 == &id1_chr));
        }
    }
    if out.iter().all(Option::is_none) {
        out.clear();
        for id1 in snp_ids1 {
            out.push(snps_ids2.iter().position(|id2| &format!("chr{id2}") == id1));
        }
    }
    out
}

pub fn match_VCF_samples(
    vcf_file1: &str,
    vcf_file2: &str,
    gt_tag1: &str,
    gt_tag2: &str,
) -> Option<BTreeMap<String, PyValue>> {
    let gt_tags1 = [gt_tag1.to_string()];
    let gt_tags2 = [gt_tag2.to_string()];
    let vcf0 = match load_VCF(vcf_file1, true, true, false, Some(&gt_tags1)) {
        Some(m) => m,
        None => return None,
    };
    let vcf1 = match load_VCF(vcf_file2, true, true, false, Some(&gt_tags2)) {
        Some(m) => m,
        None => return None,
    };
    let var0 = match vcf0.get("variants") {
        Some(PyValue::StringVec(v)) => v.clone(),
        _ => return None,
    };
    let var1 = match vcf1.get("variants") {
        Some(PyValue::StringVec(v)) => v.clone(),
        _ => return None,
    };
    let donor0 = match vcf0.get("samples") {
        Some(PyValue::StringVec(v)) => v.clone(),
        _ => return None,
    };
    let donor1 = match vcf1.get("samples") {
        Some(PyValue::StringVec(v)) => v.clone(),
        _ => return None,
    };
    let geno0 = match vcf0.get("GenoINFO") {
        Some(PyValue::Map(m)) => match m.get(gt_tag1) {
            Some(PyValue::StringMatrix(v)) => v.clone(),
            _ => return None,
        },
        _ => return None,
    };
    let geno1 = match vcf1.get("GenoINFO") {
        Some(PyValue::Map(m)) => match m.get(gt_tag2) {
            Some(PyValue::StringMatrix(v)) => v.clone(),
            _ => return None,
        },
        _ => return None,
    };
    let gpb0 = match parse_donor_GPb(&geno0, gt_tag1, 0.0) {
        Some(x) => x,
        None => return None,
    };
    let gpb1 = match parse_donor_GPb(&geno1, gt_tag2, 0.0) {
        Some(x) => x,
        None => return None,
    };
    let pairs: Vec<(usize, usize)> = match_SNPs(&var1, &var0)
        .into_iter()
        .enumerate()
        .filter_map(|(i, v)| v.map(|j| (i, j)))
        .collect();
    let mut diff = ndarray::Array2::<f64>::zeros((donor0.len(), donor1.len()));
    for d0 in 0..donor0.len() {
        for d1 in 0..donor1.len() {
            let mut total = 0.0;
            let mut count = 0usize;
            for &(i1, i0) in &pairs {
                for g in 0..gpb0.shape()[2].min(gpb1.shape()[2]) {
                    total += (gpb0[[i0, d0, g]] - gpb1[[i1, d1, g]]).abs();
                    count += 1;
                }
            }
            diff[[d0, d1]] = if count == 0 {
                f64::NAN
            } else {
                total / count as f64
            };
        }
    }
    let mut assigned = vec![false; donor1.len()];
    let mut idx0 = Vec::new();
    let mut idx1 = Vec::new();
    for d0 in 0..donor0.len() {
        let mut best = None;
        for d1 in 0..donor1.len() {
            if !assigned[d1] && best.is_none_or(|(_, v)| diff[[d0, d1]] < v) {
                best = Some((d1, diff[[d0, d1]]));
            }
        }
        if let Some((d1, _)) = best {
            assigned[d1] = true;
            idx0.push(d0);
            idx1.push(d1);
        }
    }
    let mut matched = ndarray::Array2::<f64>::zeros((idx0.len(), idx1.len()));
    for (i, &d0) in idx0.iter().enumerate() {
        for (j, &d1) in idx1.iter().enumerate() {
            matched[[i, j]] = diff[[d0, d1]];
        }
    }
    let mut rv = BTreeMap::new();
    rv.insert(
        "matched_GPb_diff".to_string(),
        PyValue::ArrayF64(matched.into_dyn()),
    );
    rv.insert(
        "matched_donors1".to_string(),
        PyValue::StringVec(idx0.iter().map(|&i| donor0[i].clone()).collect()),
    );
    rv.insert(
        "matched_donors2".to_string(),
        PyValue::StringVec(idx1.iter().map(|&i| donor1[i].clone()).collect()),
    );
    rv.insert(
        "full_GPb_diff".to_string(),
        PyValue::ArrayF64(diff.into_dyn()),
    );
    rv.insert("full_donors1".to_string(), PyValue::StringVec(donor0));
    rv.insert("full_donors2".to_string(), PyValue::StringVec(donor1));
    rv.insert(
        "matched_n_var".to_string(),
        PyValue::I64(pairs.len() as i64),
    );
    Some(rv)
}

pub fn snp_gene_match(
    var_fixed_info: &BTreeMap<String, PyValue>,
    gene_df: &BTreeMap<String, PyValue>,
    gene_key: Option<&str>,
    multi_gene: bool,
    gaps: Option<&[i64]>,
    _verbose: bool,
) -> Option<(Vec<Vec<String>>, Vec<i64>)> {
    let gene_key = gene_key.unwrap_or("gene");
    let gaps: Vec<i64> = gaps
        .map(|x| x.to_vec())
        .unwrap_or_else(|| vec![0, 1000, 10000, 100000]);
    let chroms = match var_fixed_info.get("CHROM") {
        Some(PyValue::StringVec(v)) => v,
        _ => return None,
    };
    let pos: Vec<i64> = match var_fixed_info.get("POS") {
        Some(PyValue::StringVec(v)) => v.iter().filter_map(|x| x.parse().ok()).collect(),
        Some(PyValue::I64Vec(v)) => v.clone(),
        _ => return None,
    };
    if pos.len() != chroms.len() {
        return None;
    }
    let gene_chrom = match gene_df.get("chrom") {
        Some(PyValue::StringVec(v)) => v,
        _ => return None,
    };
    let gene_start: Vec<i64> = match gene_df.get("start") {
        Some(PyValue::I64Vec(v)) => v.clone(),
        Some(PyValue::F64Vec(v)) => v.iter().map(|x| *x as i64).collect(),
        _ => return None,
    };
    let gene_stop: Vec<i64> = match gene_df.get("stop") {
        Some(PyValue::I64Vec(v)) => v.clone(),
        Some(PyValue::F64Vec(v)) => v.iter().map(|x| *x as i64).collect(),
        _ => return None,
    };
    let gene_names = match gene_df.get(gene_key) {
        Some(PyValue::StringVec(v)) => v,
        _ => return None,
    };
    let mut gene_list = Vec::new();
    let mut flag_list = Vec::new();
    for (i, chrom) in chroms.iter().enumerate() {
        let pos_i = pos[i];
        let gene_use: Vec<usize> = gene_chrom
            .iter()
            .enumerate()
            .filter_map(|(idx, c)| if c == chrom { Some(idx) } else { None })
            .collect();
        let mut idx_chrom = Vec::new();
        let mut flag = gaps.len() as i64;
        for (k, gap) in gaps.iter().enumerate() {
            flag = k as i64;
            let mut candidates = Vec::<(usize, i64)>::new();
            for &gi in &gene_use {
                let dist1 = gene_start[gi] - pos_i;
                let dist2 = gene_stop[gi] - pos_i;
                let sign = dist1.signum() * dist2.signum();
                let dist = sign * dist1.abs().min(dist2.abs());
                if dist < *gap {
                    candidates.push((gi, dist));
                }
            }
            if !candidates.is_empty() {
                if *gap > 0 || !multi_gene {
                    let nearest = candidates
                        .into_iter()
                        .min_by_key(|(_, dist)| *dist)
                        .map(|(gi, _)| gi)
                        .unwrap();
                    idx_chrom = vec![nearest];
                } else {
                    idx_chrom = candidates.into_iter().map(|(gi, _)| gi).collect();
                }
                break;
            }
        }
        if idx_chrom.is_empty() {
            flag = gaps.len() as i64;
        }
        gene_list.push(
            idx_chrom
                .iter()
                .map(|&gi| gene_names[gi].clone())
                .collect::<Vec<_>>(),
        );
        flag_list.push(flag);
    }
    Some((gene_list, flag_list))
}
