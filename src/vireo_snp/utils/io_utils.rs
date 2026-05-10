use crate::vireo_snp::utils::vcf_utils;
use crate::vireo_snp::utils::vireo_wrap::VireoWrapResult;
use ndarray::Array2;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CellData {
    pub variants: Vec<String>,
    pub fixed_info: BTreeMap<String, Vec<String>>,
    pub contigs: Vec<String>,
    pub comments: Vec<String>,
    pub samples: Vec<String>,
    pub layers: BTreeMap<String, CountMatrix>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CountMatrix {
    Dense(Array2<f64>),
    DenseU32(Array2<u32>),
    SparseCsc {
        nrows: usize,
        ncols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
    },
}

pub fn match_donor_VCF(
    mut cell_dat: CellData,
    mut donor_vcf: vcf_utils::VcfData,
) -> Option<(CellData, vcf_utils::VcfData)> {
    let pairs: Vec<(usize, usize)> = vcf_utils::match_SNPs(&cell_dat.variants, &donor_vcf.variants)
        .into_iter()
        .enumerate()
        .filter_map(|(i, v)| v.map(|j| (i, j)))
        .collect();
    let idx1: Vec<usize> = pairs.iter().map(|(i, _)| *i).collect();
    let idx2: Vec<usize> = pairs.iter().map(|(_, j)| *j).collect();
    let subset_string_vec = |values: &[String], idx: &[usize]| -> Vec<String> {
        idx.iter().map(|&i| values[i].clone()).collect()
    };
    let cell_variants = cell_dat.variants.clone();
    let donor_variants = donor_vcf.variants.clone();
    cell_dat.variants = subset_string_vec(&cell_variants, &idx1);
    donor_vcf.variants = subset_string_vec(&donor_variants, &idx2);
    for mat in cell_dat.layers.values_mut() {
        *mat = match mat {
            CountMatrix::Dense(x) => {
                let mut out = Array2::<f64>::zeros((idx1.len(), x.ncols()));
                for (new_i, &old_i) in idx1.iter().enumerate() {
                    out.row_mut(new_i).assign(&x.row(old_i));
                }
                CountMatrix::Dense(out)
            }
            CountMatrix::DenseU32(x) => {
                let mut out = ndarray::Array2::<u32>::zeros((idx1.len(), x.ncols()));
                for (new_i, &old_i) in idx1.iter().enumerate() {
                    out.row_mut(new_i).assign(&x.row(old_i));
                }
                CountMatrix::DenseU32(out)
            }
            CountMatrix::SparseCsc {
                nrows,
                ncols,
                indptr,
                indices,
                data,
            } => {
                let mut row_map = vec![usize::MAX; *nrows];
                for (new_i, &old_i) in idx1.iter().enumerate() {
                    row_map[old_i] = new_i;
                }
                let mut out_indptr = Vec::with_capacity(*ncols + 1);
                let mut out_indices = Vec::new();
                let mut out_data = Vec::new();
                out_indptr.push(0);
                for col in 0..*ncols {
                    for p in indptr[col]..indptr[col + 1] {
                        let new_row = row_map[indices[p]];
                        if new_row != usize::MAX {
                            out_indices.push(new_row);
                            out_data.push(data[p]);
                        }
                    }
                    out_indptr.push(out_indices.len());
                }
                CountMatrix::SparseCsc {
                    nrows: idx1.len(),
                    ncols: *ncols,
                    indptr: out_indptr,
                    indices: out_indices,
                    data: out_data,
                }
            }
        };
    }
    for value in cell_dat.fixed_info.values_mut() {
        *value = subset_string_vec(value, &idx1);
    }
    for value in donor_vcf.fixed_info.values_mut() {
        *value = subset_string_vec(value, &idx2);
    }
    if let Some(geno) = &mut donor_vcf.geno_info {
        for rows in geno.string_matrices.values_mut() {
            *rows = idx2.iter().map(|&i| rows[i].clone()).collect();
        }
        for values in geno.string_vecs.values_mut() {
            *values = idx2.iter().map(|&i| values[i].clone()).collect();
        }
    }
    Some((cell_dat, donor_vcf))
}

pub fn read_cellSNP(dir_name: &str, layers: Option<&[String]>) -> Option<CellData> {
    let layers = layers
        .map(|x| x.to_vec())
        .unwrap_or_else(|| vec!["AD".to_string(), "DP".to_string()]);
    let vcf_path_gz = format!("{dir_name}/cellSNP.base.vcf.gz");
    let vcf_path_plain = format!("{dir_name}/cellSNP.base.vcf");
    let vcf_path = if Path::new(&vcf_path_gz).exists() {
        vcf_path_gz
    } else {
        vcf_path_plain
    };
    let vcf_dat = vcf_utils::load_VCF(&vcf_path, false, false, true, None)?;
    let parsed_layers = std::thread::scope(|scope| {
        let mut layer_handles = Vec::with_capacity(layers.len());
        for layer in layers {
            layer_handles.push(scope.spawn(move || -> Option<(String, CountMatrix)> {
                let path = format!("{dir_name}/cellSNP.tag.{layer}.mtx");
                let file = match File::open(path) {
                    Ok(file) => file,
                    Err(_) => return None,
                };
                let mmap = match unsafe { memmap2::MmapOptions::new().map(&file) } {
                    Ok(mmap) => mmap,
                    Err(_) => return None,
                };
                let bytes = mmap.as_ref();
                let mut pos = 0usize;
                let len = bytes.len();
                while pos < len {
                    while pos < len && bytes[pos] <= b' ' {
                        pos += 1;
                    }
                    if pos < len && bytes[pos] == b'%' {
                        match memchr::memchr(b'\n', &bytes[pos..]) {
                            Some(offset) => pos += offset + 1,
                            None => pos = len,
                        }
                    } else {
                        break;
                    }
                }
                let next_usize = |pos: &mut usize| -> Option<usize> {
                    while *pos < len && bytes[*pos] <= b' ' {
                        *pos += 1;
                    }
                    if *pos >= len || bytes[*pos] < b'0' || bytes[*pos] > b'9' {
                        return None;
                    }
                    let mut value = 0usize;
                    while *pos < len && bytes[*pos] >= b'0' && bytes[*pos] <= b'9' {
                        value = value * 10 + (bytes[*pos] - b'0') as usize;
                        *pos += 1;
                    }
                    Some(value)
                };
                let rows = next_usize(&mut pos)?;
                let cols = next_usize(&mut pos)?;
                let nnz = next_usize(&mut pos)?;
                let body_start = pos;
                let n_threads = std::thread::available_parallelism()
                    .map(|x| x.get())
                    .unwrap_or(1)
                    .clamp(1, 16);
                let body_len = len.saturating_sub(body_start);
                let mut starts = Vec::with_capacity(n_threads + 1);
                starts.push(body_start);
                for t in 1..n_threads {
                    let mut start = body_start + body_len * t / n_threads;
                    if start < len && bytes[start - 1] != b'\n' {
                        match memchr::memchr(b'\n', &bytes[start..]) {
                            Some(offset) => start += offset + 1,
                            None => start = len,
                        }
                    }
                    starts.push(start);
                }
                starts.push(len);
                let bytes_ref = bytes;
                if nnz > rows.saturating_mul(cols) / 2 {
                    let mut dense = ndarray::Array2::<u32>::zeros((rows, cols));
                    let dense_ptr = dense.as_mut_ptr() as usize;
                    let parsed = std::thread::scope(|scope| {
                        let mut handles = Vec::with_capacity(n_threads);
                        for t in 0..n_threads {
                            let start = starts[t];
                            let end = starts[t + 1];
                            handles.push(scope.spawn(move || -> Option<usize> {
                                let mut pos = start;
                                let read_digits = |pos: &mut usize| -> Option<usize> {
                                    if *pos >= end
                                        || bytes_ref[*pos] < b'0'
                                        || bytes_ref[*pos] > b'9'
                                    {
                                        return None;
                                    }
                                    let mut value = 0usize;
                                    while *pos < end
                                        && bytes_ref[*pos] >= b'0'
                                        && bytes_ref[*pos] <= b'9'
                                    {
                                        value = value * 10 + (bytes_ref[*pos] - b'0') as usize;
                                        *pos += 1;
                                    }
                                    Some(value)
                                };
                                let mut count = 0usize;
                                loop {
                                    while pos < end && bytes_ref[pos] <= b' ' {
                                        pos += 1;
                                    }
                                    if pos >= end {
                                        break;
                                    }
                                    let row = read_digits(&mut pos)?;
                                    while pos < end && bytes_ref[pos] <= b' ' {
                                        pos += 1;
                                    }
                                    let col = read_digits(&mut pos)?;
                                    while pos < end && bytes_ref[pos] <= b' ' {
                                        pos += 1;
                                    }
                                    let val = read_digits(&mut pos)?;
                                    if val > u32::MAX as usize {
                                        return None;
                                    }
                                    if row == 0 || row > rows || col == 0 || col > cols {
                                        return None;
                                    }
                                    unsafe {
                                        *((dense_ptr as *mut u32)
                                            .add((row - 1) * cols + col - 1)) = val as u32;
                                    }
                                    count += 1;
                                }
                                Some(count)
                            }));
                        }
                        let mut count = 0usize;
                        for handle in handles {
                            count += handle.join().ok()??;
                        }
                        Some(count)
                    });
                    if parsed? != nnz {
                        return None;
                    }
                    return Some((layer, CountMatrix::DenseU32(dense)));
                }
                let count_result = std::thread::scope(|scope| {
                    let mut handles = Vec::with_capacity(n_threads);
                    for t in 0..n_threads {
                        let start = starts[t];
                        let end = starts[t + 1];
                        handles.push(scope.spawn(move || -> Option<(Vec<usize>, usize)> {
                            let mut pos = start;
                            let mut col_counts = vec![0usize; cols];
                            let next_usize = |pos: &mut usize| -> Option<usize> {
                                while *pos < end && bytes_ref[*pos] <= b' ' {
                                    *pos += 1;
                                }
                                if *pos >= end || bytes_ref[*pos] < b'0' || bytes_ref[*pos] > b'9' {
                                    return None;
                                }
                                let mut value = 0usize;
                                while *pos < end
                                    && bytes_ref[*pos] >= b'0'
                                    && bytes_ref[*pos] <= b'9'
                                {
                                    value = value * 10 + (bytes_ref[*pos] - b'0') as usize;
                                    *pos += 1;
                                }
                                Some(value)
                            };
                            let mut count = 0usize;
                            loop {
                                while pos < end && bytes_ref[pos] <= b' ' {
                                    pos += 1;
                                }
                                if pos >= end {
                                    break;
                                }
                                let row = next_usize(&mut pos)?;
                                let col = next_usize(&mut pos)?;
                                let _ = next_usize(&mut pos)?;
                                if row == 0 || row > rows || col == 0 || col > cols {
                                    return None;
                                }
                                col_counts[col - 1] += 1;
                                count += 1;
                            }
                            Some((col_counts, count))
                        }));
                    }
                    let mut chunk_col_counts = Vec::with_capacity(n_threads);
                    let mut count = 0usize;
                    for handle in handles {
                        let (thread_counts, thread_count) = handle.join().ok()??;
                        count += thread_count;
                        chunk_col_counts.push(thread_counts);
                    }
                    Some((chunk_col_counts, count))
                });
                let (chunk_col_counts, counted_nnz) = count_result?;
                if counted_nnz != nnz {
                    return None;
                }
                let mut indptr = vec![0usize; cols + 1];
                for col in 0..cols {
                    let mut col_total = 0usize;
                    for counts in &chunk_col_counts {
                        col_total += counts[col];
                    }
                    indptr[col + 1] = indptr[col] + col_total;
                }
                let mut chunk_offsets = vec![vec![0usize; cols]; n_threads];
                let mut next_offsets = indptr[..cols].to_vec();
                for t in 0..n_threads {
                    chunk_offsets[t][..cols].copy_from_slice(&next_offsets[..cols]);
                    for col in 0..cols {
                        next_offsets[col] += chunk_col_counts[t][col];
                    }
                }
                let mut indices = vec![0usize; nnz];
                let mut data = vec![0f64; nnz];
                let indices_ptr = indices.as_mut_ptr() as usize;
                let data_ptr = data.as_mut_ptr() as usize;
                let parsed = std::thread::scope(|scope| {
                    let mut handles = Vec::with_capacity(n_threads);
                    for t in 0..n_threads {
                        let start = starts[t];
                        let end = starts[t + 1];
                        let mut offsets = chunk_offsets[t].clone();
                        handles.push(scope.spawn(move || -> Option<usize> {
                            let mut pos = start;
                            let next_usize = |pos: &mut usize| -> Option<usize> {
                                while *pos < end && bytes_ref[*pos] <= b' ' {
                                    *pos += 1;
                                }
                                if *pos >= end || bytes_ref[*pos] < b'0' || bytes_ref[*pos] > b'9' {
                                    return None;
                                }
                                let mut value = 0usize;
                                while *pos < end
                                    && bytes_ref[*pos] >= b'0'
                                    && bytes_ref[*pos] <= b'9'
                                {
                                    value = value * 10 + (bytes_ref[*pos] - b'0') as usize;
                                    *pos += 1;
                                }
                                Some(value)
                            };
                            let mut count = 0usize;
                            loop {
                                while pos < end && bytes_ref[pos] <= b' ' {
                                    pos += 1;
                                }
                                if pos >= end {
                                    break;
                                }
                                let row = next_usize(&mut pos)?;
                                let col = next_usize(&mut pos)?;
                                let val = next_usize(&mut pos)? as f64;
                                if row == 0 || row > rows || col == 0 || col > cols {
                                    return None;
                                }
                                let col0 = col - 1;
                                let p = offsets[col0];
                                offsets[col0] += 1;
                                unsafe {
                                    *((indices_ptr as *mut usize).add(p)) = row - 1;
                                    *((data_ptr as *mut f64).add(p)) = val;
                                }
                                count += 1;
                            }
                            Some(count)
                        }));
                    }
                    let mut count = 0usize;
                    for handle in handles {
                        count += handle.join().ok()??;
                    }
                    Some(count)
                });
                if parsed? != nnz {
                    return None;
                }
                Some((
                    layer,
                    CountMatrix::SparseCsc {
                        nrows: rows,
                        ncols: cols,
                        indptr,
                        indices,
                        data,
                    },
                ))
            }));
        }
        let mut parsed_layers = Vec::with_capacity(layer_handles.len());
        for handle in layer_handles {
            parsed_layers.push(handle.join().ok()??);
        }
        Some(parsed_layers)
    })?;
    let mut cell_dat = CellData {
        variants: vcf_dat.variants,
        fixed_info: vcf_dat.fixed_info,
        contigs: vcf_dat.contigs,
        comments: vcf_dat.comments,
        ..Default::default()
    };
    for (layer, matrix) in parsed_layers {
        cell_dat.layers.insert(layer, matrix);
    }
    let samples_file = match File::open(format!("{dir_name}/cellSNP.samples.tsv")) {
        Ok(file) => file,
        Err(_) => return None,
    };
    let samples: Vec<String> = BufReader::new(samples_file)
        .lines()
        .map_while(Result::ok)
        .collect();
    cell_dat.samples = samples;
    Some(cell_dat)
}

pub fn read_vartrix(
    alt_mtx: &str,
    ref_mtx: &str,
    cell_file: &str,
    vcf_file: Option<&str>,
) -> Option<CellData> {
    let mut cell_dat = match vcf_file {
        Some(v) => {
            let vcf_dat = vcf_utils::load_VCF(v, false, false, true, None)?;
            CellData {
                variants: vcf_dat.variants,
                fixed_info: vcf_dat.fixed_info,
                contigs: vcf_dat.contigs,
                comments: vcf_dat.comments,
                ..Default::default()
            }
        }
        None => CellData::default(),
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
    let (rows, cols) = dims?;
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
    let (ref_rows, ref_cols) = dims?;
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
    cell_dat
        .layers
        .insert("AD".to_string(), CountMatrix::Dense(ad));
    cell_dat
        .layers
        .insert("DP".to_string(), CountMatrix::Dense(dp));
    cell_dat.samples = samples;
    Some(cell_dat)
}

pub fn write_donor_id(
    out_dir: &str,
    donor_names: &[String],
    cell_names: &[String],
    n_vars: &[f64],
    res: &VireoWrapResult,
) -> Option<()> {
    let id_prob = &res.id_prob;
    let doublet_prob = &res.doublet_prob;
    let doublet_llr = &res.doublet_llr;
    let lb_doublet = res.lb_doublet;
    let theta_shapes = format!("{:?}", res.theta_shapes);
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
    if let Some(ambient) = &res.ambient_psi {
        let llr: Vec<f64> = res
            .psi_llratio
            .clone()
            .unwrap_or_else(|| vec![0.0; ambient.nrows()]);
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
