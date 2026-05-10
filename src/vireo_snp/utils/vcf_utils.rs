use crate::vireo_snp::utils::vireo_base;
use flate2::read::MultiGzDecoder;
use ndarray::{Array2, Array3};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VcfGenoInfo {
    pub string_vecs: BTreeMap<String, Vec<String>>,
    pub string_matrices: BTreeMap<String, Vec<Vec<String>>>,
    pub i64_vecs: BTreeMap<String, Vec<i64>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VcfData {
    pub variants: Vec<String>,
    pub fixed_info: BTreeMap<String, Vec<String>>,
    pub contigs: Vec<String>,
    pub comments: Vec<String>,
    pub samples: Vec<String>,
    pub geno_info: Option<VcfGenoInfo>,
    pub n_snp_tagged: Vec<i64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VcfSampleMatchResult {
    pub matched_gpb_diff: Array2<f64>,
    pub matched_donors1: Vec<String>,
    pub matched_donors2: Vec<String>,
    pub full_gpb_diff: Array2<f64>,
    pub full_donors1: Vec<String>,
    pub full_donors2: Vec<String>,
    pub matched_n_var: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeneData {
    pub chrom: Vec<String>,
    pub start: Vec<i64>,
    pub stop: Vec<i64>,
    pub gene: Vec<String>,
}

pub fn parse_sample_info(
    sample_dat: &[Vec<String>],
    sparse: bool,
    format_list: Option<&[String]>,
) -> Option<(VcfGenoInfo, Vec<i64>)> {
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
    let mut rv = VcfGenoInfo::default();
    let mut n_snp_tagged = vec![0i64; format_list.len()];
    for key in &format_list {
        rv.string_vecs.insert(key.clone(), Vec::new());
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
                    if let Some(values) = rv.string_vecs.get_mut(key) {
                        values.push(line_key[k].to_string());
                    }
                    n_snp_tagged[k] += 1;
                }
                cnt += 1;
                indices.push(i as i64);
            }
            indptr.push(cnt);
        }
        rv.i64_vecs.insert("indices".to_string(), indices);
        rv.i64_vecs.insert("indptr".to_string(), indptr);
        rv.i64_vecs.insert(
            "shape".to_string(),
            vec![(sample_dat[0].len() - 1) as i64, sample_dat.len() as i64],
        );
    } else {
        rv.string_vecs.clear();
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
                if let Some(rows) = rv.string_matrices.get_mut(key) {
                    rows.push(values);
                } else {
                    rv.string_matrices.insert(key.clone(), vec![values]);
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
) -> Option<VcfData> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(_) => return None,
    };
    let mut reader: Box<dyn Read> = if path.ends_with(".gz") || path.ends_with(".bgz") {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut fixed_info: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut contig_lines = Vec::new();
    let mut comment_lines = Vec::new();
    let mut var_ids = Vec::new();
    let mut obs_ids = Vec::new();
    let mut obs_dat = Vec::new();
    let mut key_ids = Vec::<String>::new();
    if load_sample && sparse && !(path.ends_with(".gz") || path.ends_with(".bgz")) {
        let mmap_file = match File::open(path) {
            Ok(file) => file,
            Err(_) => return None,
        };
        let mmap = match unsafe { memmap2::MmapOptions::new().map(&mmap_file) } {
            Ok(mmap) => mmap,
            Err(_) => return None,
        };
        let bytes = mmap.as_ref();
        let reserve_rows = bytes.len() / 250;
        let mut data_start = 0usize;
        let mut line_start = 0usize;
        while line_start < bytes.len() {
            let mut line_end = match memchr::memchr(b'\n', &bytes[line_start..]) {
                Some(offset) => line_start + offset,
                None => bytes.len(),
            };
            let next_line = if line_end < bytes.len() {
                line_end + 1
            } else {
                line_end
            };
            if line_end > line_start && bytes[line_end - 1] == b'\r' {
                line_end -= 1;
            }
            if line_start == line_end {
                line_start = next_line;
                continue;
            }
            let line = &bytes[line_start..line_end];
            if line[0] != b'#' {
                data_start = line_start;
                break;
            }
            if line.starts_with(b"##contig=") {
                contig_lines.push(unsafe { std::str::from_utf8_unchecked(line) }.to_string());
            }
            if line.starts_with(b"#CHROM") {
                key_ids.clear();
                let mut p = 0usize;
                let mut field_i = 0usize;
                while p <= line.len() {
                    let start = p;
                    while p < line.len() && line[p] != b'\t' {
                        p += 1;
                    }
                    let mut field = &line[start..p];
                    if field_i < 8 {
                        if field.first() == Some(&b'#') {
                            field = &field[1..];
                        }
                        key_ids.push(unsafe { std::str::from_utf8_unchecked(field) }.to_string());
                    } else if field_i >= 9 {
                        obs_ids.push(unsafe { std::str::from_utf8_unchecked(field) }.to_string());
                    }
                    field_i += 1;
                    if p == line.len() {
                        break;
                    }
                    p += 1;
                }
            } else {
                comment_lines.push(unsafe { std::str::from_utf8_unchecked(line) }.to_string());
            }
            line_start = next_line;
        }
        if data_start == 0 || key_ids.len() < 8 {
            return None;
        }
        let mut format_keys = format_list.map(|x| x.to_vec()).unwrap_or_default();
        if format_keys.is_empty() {
            let mut first_end = match memchr::memchr(b'\n', &bytes[data_start..]) {
                Some(offset) => data_start + offset,
                None => bytes.len(),
            };
            if first_end > data_start && bytes[first_end - 1] == b'\r' {
                first_end -= 1;
            }
            let line = &bytes[data_start..first_end];
            let mut p = 0usize;
            for _ in 0..8 {
                while p < line.len() && line[p] != b'\t' {
                    p += 1;
                }
                if p < line.len() {
                    p += 1;
                }
            }
            let fmt_start = p;
            while p < line.len() && line[p] != b'\t' {
                p += 1;
            }
            let fmt_b = &line[fmt_start..p];
            let mut q = 0usize;
            while q <= fmt_b.len() {
                let start = q;
                while q < fmt_b.len() && fmt_b[q] != b':' {
                    q += 1;
                }
                format_keys
                    .push(unsafe { std::str::from_utf8_unchecked(&fmt_b[start..q]) }.to_string());
                if q == fmt_b.len() {
                    break;
                }
                q += 1;
            }
        }
        let expected_format = format_keys.join(":").into_bytes();
        let n_threads = std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1)
            .clamp(1, 16);
        let body_len = bytes.len().saturating_sub(data_start);
        let mut starts = Vec::with_capacity(n_threads + 1);
        starts.push(data_start);
        for t in 1..n_threads {
            let mut start = data_start + body_len * t / n_threads;
            if start < bytes.len() && bytes[start - 1] != b'\n' {
                match memchr::memchr(b'\n', &bytes[start..]) {
                    Some(offset) => start += offset + 1,
                    None => start = bytes.len(),
                }
            }
            starts.push(start);
        }
        starts.push(bytes.len());
        let bytes_ref = bytes;
        let parsed = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(n_threads);
            for t in 0..n_threads {
                let start = starts[t];
                let end = starts[t + 1];
                let format_keys = &format_keys;
                let expected_format = expected_format.as_slice();
                handles.push(scope.spawn(move || {
                    let reserve_rows = (end - start) / 350;
                    let reserve_nnz = reserve_rows.saturating_mul(8);
                    let mut local_var_ids = Vec::<String>::with_capacity(reserve_rows);
                    let mut local_fixed_cols: Vec<Vec<String>> =
                        (0..8).map(|_| Vec::with_capacity(reserve_rows)).collect();
                    let mut local_geno = (0..format_keys.len())
                        .map(|_| Vec::<String>::with_capacity(reserve_nnz))
                        .collect::<Vec<_>>();
                    let mut local_n_tagged = vec![0i64; format_keys.len()];
                    let mut local_indices = Vec::<i64>::with_capacity(reserve_nnz);
                    let mut local_indptr = Vec::<i64>::with_capacity(reserve_rows + 1);
                    local_indptr.push(0);
                    let mut local_cnt = 0i64;
                    let mut line_start = start;
                    while line_start < end {
                        let mut line_end = match memchr::memchr(b'\n', &bytes_ref[line_start..end])
                        {
                            Some(offset) => line_start + offset,
                            None => end,
                        };
                        let next_line = if line_end < end {
                            line_end + 1
                        } else {
                            line_end
                        };
                        if line_end > line_start && bytes_ref[line_end - 1] == b'\r' {
                            line_end -= 1;
                        }
                        if line_start == line_end {
                            line_start = next_line;
                            continue;
                        }
                        let line = &bytes_ref[line_start..line_end];
                        let mut fields = [(0usize, 0usize); 9];
                        let mut p = 0usize;
                        for field in &mut fields {
                            let start = p;
                            while p < line.len() && line[p] != b'\t' {
                                p += 1;
                            }
                            *field = (start, p);
                            if p < line.len() {
                                p += 1;
                            }
                        }
                        let chrom_b = &line[fields[0].0..fields[0].1];
                        let pos_b = &line[fields[1].0..fields[1].1];
                        let id_b = &line[fields[2].0..fields[2].1];
                        let ref_b = &line[fields[3].0..fields[3].1];
                        let alt_b = &line[fields[4].0..fields[4].1];
                        let qual_b = &line[fields[5].0..fields[5].1];
                        let filter_b = &line[fields[6].0..fields[6].1];
                        let info_b = &line[fields[7].0..fields[7].1];
                        let fmt_b = &line[fields[8].0..fields[8].1];
                        if biallelic_only && (ref_b.len() > 1 || alt_b.len() > 1) {
                            line_start = next_line;
                            continue;
                        }
                        if fmt_b != expected_format {
                            return None;
                        }
                        let chrom = unsafe { std::str::from_utf8_unchecked(chrom_b) };
                        let pos_s = unsafe { std::str::from_utf8_unchecked(pos_b) };
                        let ref_s = unsafe { std::str::from_utf8_unchecked(ref_b) };
                        let alt_s = unsafe { std::str::from_utf8_unchecked(alt_b) };
                        local_fixed_cols[0].push(chrom.to_string());
                        local_fixed_cols[1].push(pos_s.to_string());
                        local_fixed_cols[2]
                            .push(unsafe { std::str::from_utf8_unchecked(id_b) }.to_string());
                        local_fixed_cols[3].push(ref_s.to_string());
                        local_fixed_cols[4].push(alt_s.to_string());
                        local_fixed_cols[5]
                            .push(unsafe { std::str::from_utf8_unchecked(qual_b) }.to_string());
                        local_fixed_cols[6]
                            .push(unsafe { std::str::from_utf8_unchecked(filter_b) }.to_string());
                        local_fixed_cols[7]
                            .push(unsafe { std::str::from_utf8_unchecked(info_b) }.to_string());
                        let mut variant = String::with_capacity(
                            chrom.len() + pos_s.len() + ref_s.len() + alt_s.len() + 3,
                        );
                        variant.push_str(chrom);
                        variant.push('_');
                        variant.push_str(pos_s);
                        variant.push('_');
                        variant.push_str(ref_s);
                        variant.push('_');
                        variant.push_str(alt_s);
                        local_var_ids.push(variant);
                        let mut sample_i = 0usize;
                        while p <= line.len() {
                            let start = p;
                            while p < line.len() && line[p] != b'\t' {
                                p += 1;
                            }
                            let cell = &line[start..p];
                            if cell != b"." && cell.iter().any(|&b| b != b'.' && b != b':') {
                                let mut q = 0usize;
                                for (k, values) in local_geno.iter_mut().enumerate() {
                                    let val_start = q;
                                    while q < cell.len() && cell[q] != b':' {
                                        q += 1;
                                    }
                                    values.push(
                                        unsafe {
                                            std::str::from_utf8_unchecked(&cell[val_start..q])
                                        }
                                        .to_string(),
                                    );
                                    local_n_tagged[k] += 1;
                                    if q < cell.len() {
                                        q += 1;
                                    }
                                }
                                local_indices.push(sample_i as i64);
                                local_cnt += 1;
                            }
                            sample_i += 1;
                            if p == line.len() {
                                break;
                            }
                            p += 1;
                        }
                        local_indptr.push(local_cnt);
                        line_start = next_line;
                    }
                    Some((
                        local_var_ids,
                        local_fixed_cols,
                        local_geno,
                        local_n_tagged,
                        local_indices,
                        local_indptr,
                        local_cnt,
                    ))
                }));
            }
            let mut parsed = Vec::with_capacity(n_threads);
            for handle in handles {
                parsed.push(handle.join().ok()??);
            }
            Some(parsed)
        })?;
        let mut fixed_cols: Vec<Vec<String>> =
            (0..8).map(|_| Vec::with_capacity(reserve_rows)).collect();
        let mut geno_info = VcfGenoInfo::default();
        let mut n_snp_tagged = vec![0i64; format_keys.len()];
        let mut indices = Vec::<i64>::new();
        let mut indptr = vec![0i64];
        let mut cnt = 0i64;
        for key in &format_keys {
            geno_info.string_vecs.insert(key.clone(), Vec::new());
        }
        for (
            local_var_ids,
            local_fixed_cols,
            local_geno,
            local_n_tagged,
            local_indices,
            local_indptr,
            local_cnt,
        ) in parsed
        {
            var_ids.extend(local_var_ids);
            for (col, values) in local_fixed_cols.into_iter().enumerate() {
                fixed_cols[col].extend(values);
            }
            for (k, (key, values_in)) in format_keys.iter().zip(local_geno).enumerate() {
                if let Some(values) = geno_info.string_vecs.get_mut(key) {
                    values.extend(values_in);
                }
                n_snp_tagged[k] += local_n_tagged[k];
            }
            indices.extend(local_indices);
            for value in local_indptr.into_iter().skip(1) {
                indptr.push(cnt + value);
            }
            cnt += local_cnt;
        }
        for (key, values) in key_ids.into_iter().zip(fixed_cols.into_iter()) {
            fixed_info.insert(key, values);
        }
        geno_info.i64_vecs.insert("indices".to_string(), indices);
        geno_info.i64_vecs.insert("indptr".to_string(), indptr);
        geno_info.i64_vecs.insert(
            "shape".to_string(),
            vec![obs_ids.len() as i64, var_ids.len() as i64],
        );
        return Some(VcfData {
            variants: var_ids,
            fixed_info,
            contigs: contig_lines,
            comments: comment_lines,
            samples: obs_ids,
            geno_info: Some(geno_info),
            n_snp_tagged,
        });
    }
    if !load_sample {
        if !(path.ends_with(".gz") || path.ends_with(".bgz")) {
            let mmap_file = match File::open(path) {
                Ok(file) => file,
                Err(_) => return None,
            };
            let mmap = match unsafe { memmap2::MmapOptions::new().map(&mmap_file) } {
                Ok(mmap) => mmap,
                Err(_) => return None,
            };
            let bytes = mmap.as_ref();
            let reserve_rows = bytes.len() / 50;
            var_ids.reserve(reserve_rows);
            let mut fixed_cols = Vec::<Vec<String>>::new();
            let mut line_start = 0usize;
            while line_start < bytes.len() {
                let mut line_end = match memchr::memchr(b'\n', &bytes[line_start..]) {
                    Some(offset) => line_start + offset,
                    None => bytes.len(),
                };
                let next_line = if line_end < bytes.len() {
                    line_end + 1
                } else {
                    line_end
                };
                if line_end > line_start && bytes[line_end - 1] == b'\r' {
                    line_end -= 1;
                }
                if line_start == line_end {
                    line_start = next_line;
                    continue;
                }
                let line = &bytes[line_start..line_end];
                if line[0] == b'#' {
                    if line.starts_with(b"##contig=") {
                        contig_lines
                            .push(unsafe { std::str::from_utf8_unchecked(line) }.to_string());
                    }
                    if line.starts_with(b"#CHROM") {
                        key_ids.clear();
                        let mut p = 0usize;
                        for _ in 0..8 {
                            let start = p;
                            while p < line.len() && line[p] != b'\t' {
                                p += 1;
                            }
                            let mut field = &line[start..p];
                            if field.first() == Some(&b'#') {
                                field = &field[1..];
                            }
                            key_ids
                                .push(unsafe { std::str::from_utf8_unchecked(field) }.to_string());
                            if p < line.len() {
                                p += 1;
                            }
                        }
                        fixed_cols = (0..key_ids.len())
                            .map(|_| Vec::with_capacity(reserve_rows))
                            .collect();
                    } else {
                        comment_lines
                            .push(unsafe { std::str::from_utf8_unchecked(line) }.to_string());
                    }
                } else {
                    if key_ids.len() < 8 || fixed_cols.len() < 8 {
                        return None;
                    }
                    let mut fields = [(0usize, 0usize); 8];
                    let mut p = 0usize;
                    for field in &mut fields {
                        let start = p;
                        while p < line.len() && line[p] != b'\t' {
                            p += 1;
                        }
                        *field = (start, p);
                        if p < line.len() {
                            p += 1;
                        }
                    }
                    let chrom_b = &line[fields[0].0..fields[0].1];
                    let pos_b = &line[fields[1].0..fields[1].1];
                    let id_b = &line[fields[2].0..fields[2].1];
                    let ref_b = &line[fields[3].0..fields[3].1];
                    let alt_b = &line[fields[4].0..fields[4].1];
                    let qual_b = &line[fields[5].0..fields[5].1];
                    let filter_b = &line[fields[6].0..fields[6].1];
                    let info_b = &line[fields[7].0..fields[7].1];
                    if biallelic_only && (ref_b.len() > 1 || alt_b.len() > 1) {
                        line_start = next_line;
                        continue;
                    }
                    let chrom = unsafe { std::str::from_utf8_unchecked(chrom_b) };
                    let pos_s = unsafe { std::str::from_utf8_unchecked(pos_b) };
                    let ref_s = unsafe { std::str::from_utf8_unchecked(ref_b) };
                    let alt_s = unsafe { std::str::from_utf8_unchecked(alt_b) };
                    fixed_cols[0].push(chrom.to_string());
                    fixed_cols[1].push(pos_s.to_string());
                    fixed_cols[2].push(unsafe { std::str::from_utf8_unchecked(id_b) }.to_string());
                    fixed_cols[3].push(ref_s.to_string());
                    fixed_cols[4].push(alt_s.to_string());
                    fixed_cols[5]
                        .push(unsafe { std::str::from_utf8_unchecked(qual_b) }.to_string());
                    fixed_cols[6]
                        .push(unsafe { std::str::from_utf8_unchecked(filter_b) }.to_string());
                    fixed_cols[7]
                        .push(unsafe { std::str::from_utf8_unchecked(info_b) }.to_string());
                    let mut variant = String::with_capacity(
                        chrom.len() + pos_s.len() + ref_s.len() + alt_s.len() + 3,
                    );
                    variant.push_str(chrom);
                    variant.push('_');
                    variant.push_str(pos_s);
                    variant.push('_');
                    variant.push_str(ref_s);
                    variant.push('_');
                    variant.push_str(alt_s);
                    var_ids.push(variant);
                }
                line_start = next_line;
            }
            for (key, values) in key_ids.into_iter().zip(fixed_cols.into_iter()) {
                fixed_info.insert(key, values);
            }
            return Some(VcfData {
                variants: var_ids,
                fixed_info,
                contigs: contig_lines,
                comments: comment_lines,
                ..Default::default()
            });
        }
        let mut text = String::new();
        if reader.read_to_string(&mut text).is_err() {
            return None;
        }
        let reserve_rows = text.len() / 50;
        var_ids.reserve(reserve_rows);
        let mut fixed_cols = Vec::<Vec<String>>::new();
        for line_raw in text.lines() {
            let line = line_raw.trim_end_matches('\r');
            if line.starts_with('#') {
                if line.starts_with("##contig=") {
                    contig_lines.push(line.to_string());
                }
                if line.starts_with("#CHROM") {
                    let parts: Vec<&str> = line.split('\t').collect();
                    key_ids = parts
                        .iter()
                        .take(8)
                        .map(|s| s.trim_start_matches('#').to_string())
                        .collect();
                    fixed_cols = (0..key_ids.len())
                        .map(|_| Vec::with_capacity(reserve_rows))
                        .collect();
                } else {
                    comment_lines.push(line.to_string());
                }
            } else {
                if key_ids.len() < 8 || fixed_cols.len() < 8 {
                    return None;
                }
                let mut parts = line.split('\t');
                let chrom = parts.next()?;
                let pos = parts.next()?;
                let id = parts.next()?;
                let ref_allele = parts.next()?;
                let alt_allele = parts.next()?;
                let qual = parts.next()?;
                let filter = parts.next()?;
                let info = parts.next()?;
                if biallelic_only && (ref_allele.len() > 1 || alt_allele.len() > 1) {
                    continue;
                }
                fixed_cols[0].push(chrom.to_string());
                fixed_cols[1].push(pos.to_string());
                fixed_cols[2].push(id.to_string());
                fixed_cols[3].push(ref_allele.to_string());
                fixed_cols[4].push(alt_allele.to_string());
                fixed_cols[5].push(qual.to_string());
                fixed_cols[6].push(filter.to_string());
                fixed_cols[7].push(info.to_string());
                var_ids.push(format!("{chrom}_{pos}_{ref_allele}_{alt_allele}"));
            }
        }
        for (key, values) in key_ids.into_iter().zip(fixed_cols.into_iter()) {
            fixed_info.insert(key, values);
        }
        return Some(VcfData {
            variants: var_ids,
            fixed_info,
            contigs: contig_lines,
            comments: comment_lines,
            ..Default::default()
        });
    }
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
                    fixed_info.insert(key.clone(), Vec::new());
                }
            } else {
                comment_lines.push(line);
            }
        } else if load_sample {
            let list_val: Vec<String> =
                line.trim_end().split('\t').map(|s| s.to_string()).collect();
            if biallelic_only && (list_val[3].len() > 1 || list_val[4].len() > 1) {
                continue;
            }
            obs_dat.push(list_val.iter().skip(8).cloned().collect());
            for (i, key) in key_ids.iter().enumerate() {
                if let Some(values) = fixed_info.get_mut(key) {
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
        } else {
            let line_trimmed = line.trim_end();
            let list_val: Vec<&str> = line_trimmed.split('\t').take(key_ids.len()).collect();
            if list_val.len() < key_ids.len() || list_val.len() < 5 {
                return None;
            }
            if biallelic_only && (list_val[3].len() > 1 || list_val[4].len() > 1) {
                continue;
            }
            for (i, key) in key_ids.iter().enumerate() {
                if let Some(values) = fixed_info.get_mut(key) {
                    values.push(list_val[i].to_string());
                }
            }
            var_ids.push(format!(
                "{}_{}_{}_{}",
                list_val[0], list_val[1], list_val[3], list_val[4]
            ));
        }
    }
    let mut rv = VcfData {
        variants: var_ids,
        fixed_info,
        contigs: contig_lines,
        comments: comment_lines,
        ..Default::default()
    };
    if load_sample {
        rv.samples = obs_ids;
        let (geno_info, n_snp_tagged) = parse_sample_info(&obs_dat, sparse, format_list)?;
        rv.geno_info = Some(geno_info);
        rv.n_snp_tagged = n_snp_tagged;
    }
    Some(rv)
}

pub fn write_VCF_to_hdf5(vcf_dat: &VcfData, out_file: &str) -> Option<()> {
    let file = match hdf5::File::create(out_file) {
        Ok(f) => f,
        Err(_) => return None,
    };
    for (key, values) in [
        ("contigs", &vcf_dat.contigs),
        ("samples", &vcf_dat.samples),
        ("variants", &vcf_dat.variants),
        ("comments", &vcf_dat.comments),
    ] {
        if !values.is_empty() {
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
    if !vcf_dat.fixed_info.is_empty() {
        let group = match file.create_group("FixedINFO") {
            Ok(g) => g,
            Err(_) => return None,
        };
        for (key, values) in &vcf_dat.fixed_info {
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
    if let Some(geno_info) = &vcf_dat.geno_info {
        let group = match file.create_group("GenoINFO") {
            Ok(g) => g,
            Err(_) => return None,
        };
        for (key, values) in &geno_info.string_vecs {
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
        for (key, rows) in &geno_info.string_matrices {
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
        for (key, values) in &geno_info.i64_vecs {
            if group
                .new_dataset_builder()
                .with_data(values)
                .create(key.as_str())
                .is_err()
            {
                return None;
            }
        }
    }
    Some(())
}

pub fn read_sparse_GeneINFO(
    geno_info: &VcfGenoInfo,
    keys: Option<&[String]>,
    axes: Option<&[i64]>,
) -> Option<BTreeMap<String, Array2<f64>>> {
    let keys = keys
        .map(|x| x.to_vec())
        .unwrap_or_else(|| vec!["AD".to_string(), "DP".to_string()]);
    let axes = axes
        .map(|x| x.to_vec())
        .unwrap_or_else(|| vec![-1; keys.len()]);
    let shape = match geno_info.i64_vecs.get("shape") {
        Some(v) if v.len() == 2 => (v[0] as usize, v[1] as usize),
        _ => return None,
    };
    let indptr = match geno_info.i64_vecs.get("indptr") {
        Some(v) => v,
        _ => return None,
    };
    let indices = match geno_info.i64_vecs.get("indices") {
        Some(v) => v,
        _ => return None,
    };
    let mut rv = BTreeMap::new();
    for (ki, key) in keys.iter().enumerate() {
        let values = match geno_info.string_vecs.get(key) {
            Some(v) => v,
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

pub fn write_VCF(out_file: &str, vcf_dat: &VcfData, geno_tags: Option<&[String]>) -> Option<()> {
    let geno_tags = geno_tags
        .map(|x| x.to_vec())
        .unwrap_or_else(|| vec!["GT".into(), "AD".into(), "DP".into(), "PL".into()]);
    let out_file_use = out_file.strip_suffix(".gz").unwrap_or(out_file).to_string();
    let mut f = match File::create(&out_file_use) {
        Ok(f) => f,
        Err(_) => return None,
    };
    for line in &vcf_dat.comments {
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
    let samples = vcf_dat.samples.clone();
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
    let variants = &vcf_dat.variants;
    let fixed_info = &vcf_dat.fixed_info;
    let geno_info = vcf_dat.geno_info.as_ref();
    let fixed_cols = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"];
    for i in 0..variants.len() {
        let mut line = Vec::new();
        for col in fixed_cols {
            let values = match fixed_info.get(col) {
                Some(v) => v,
                _ => return None,
            };
            let value = values.get(i)?;
            line.push(value.clone());
        }
        if !geno_tags.is_empty() {
            let geno_info = geno_info?;
            line.push(geno_tags.join(":"));
            for s in 0..samples.len() {
                let mut values = Vec::new();
                for tag in &geno_tags {
                    let rows = match geno_info.string_matrices.get(tag) {
                        Some(v) => v,
                        _ => return None,
                    };
                    let row = rows.get(i)?;
                    let value = row.get(s)?;
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
    let mut map2 = HashMap::with_capacity(snps_ids2.len());
    for (i, id2) in snps_ids2.iter().enumerate() {
        map2.entry(id2.as_str()).or_insert(i);
    }
    let mut out: Vec<Option<usize>> = snp_ids1
        .iter()
        .map(|id1| map2.get(id1.as_str()).copied())
        .collect();
    if out.iter().all(Option::is_none) {
        out = snp_ids1
            .iter()
            .map(|id1| {
                let id1_chr = format!("chr{id1}");
                map2.get(id1_chr.as_str()).copied()
            })
            .collect();
    }
    if out.iter().all(Option::is_none) {
        let mut chr_map2 = HashMap::with_capacity(snps_ids2.len());
        for (i, id2) in snps_ids2.iter().enumerate() {
            chr_map2.entry(format!("chr{id2}")).or_insert(i);
        }
        out = snp_ids1
            .iter()
            .map(|id1| chr_map2.get(id1).copied())
            .collect();
    }
    out
}

pub fn match_VCF_samples(
    vcf_file1: &str,
    vcf_file2: &str,
    gt_tag1: &str,
    gt_tag2: &str,
) -> Option<VcfSampleMatchResult> {
    let gt_tags1 = [gt_tag1.to_string()];
    let gt_tags2 = [gt_tag2.to_string()];
    let vcf0 = load_VCF(vcf_file1, true, true, false, Some(&gt_tags1))?;
    let vcf1 = load_VCF(vcf_file2, true, true, false, Some(&gt_tags2))?;
    let var0 = vcf0.variants.clone();
    let var1 = vcf1.variants.clone();
    let donor0 = vcf0.samples.clone();
    let donor1 = vcf1.samples.clone();
    let geno0 = match &vcf0.geno_info {
        Some(m) => match m.string_matrices.get(gt_tag1) {
            Some(v) => v.clone(),
            _ => return None,
        },
        _ => return None,
    };
    let geno1 = match &vcf1.geno_info {
        Some(m) => match m.string_matrices.get(gt_tag2) {
            Some(v) => v.clone(),
            _ => return None,
        },
        _ => return None,
    };
    let gpb0 = parse_donor_GPb(&geno0, gt_tag1, 0.0)?;
    let gpb1 = parse_donor_GPb(&geno1, gt_tag2, 0.0)?;
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
    let n_gt = gpb0.shape()[2].min(gpb1.shape()[2]);
    let mut flat0 = ndarray::Array2::<f64>::zeros((pairs.len() * n_gt, donor0.len()));
    let mut flat1 = ndarray::Array2::<f64>::zeros((pairs.len() * n_gt, donor1.len()));
    for (pair_idx, &(i1, i0)) in pairs.iter().enumerate() {
        for g in 0..n_gt {
            let row = pair_idx * n_gt + g;
            for d0 in 0..donor0.len() {
                flat0[[row, d0]] = gpb0[[i0, d0, g]];
            }
            for d1 in 0..donor1.len() {
                flat1[[row, d1]] = gpb1[[i1, d1, g]];
            }
        }
    }
    let (idx0, idx1, _) = vireo_base::optimal_match(&flat0, &flat1, Some(1), false)?;
    let mut matched = ndarray::Array2::<f64>::zeros((idx0.len(), idx1.len()));
    for (i, &d0) in idx0.iter().enumerate() {
        for (j, &d1) in idx1.iter().enumerate() {
            matched[[i, j]] = diff[[d0, d1]];
        }
    }
    Some(VcfSampleMatchResult {
        matched_gpb_diff: matched,
        matched_donors1: idx0.iter().map(|&i| donor0[i].clone()).collect(),
        matched_donors2: idx1.iter().map(|&i| donor1[i].clone()).collect(),
        full_gpb_diff: diff,
        full_donors1: donor0,
        full_donors2: donor1,
        matched_n_var: pairs.len(),
    })
}

pub fn snp_gene_match(
    var_fixed_info: &BTreeMap<String, Vec<String>>,
    gene_df: &GeneData,
    gene_key: Option<&str>,
    multi_gene: bool,
    gaps: Option<&[i64]>,
    _verbose: bool,
) -> Option<(Vec<Vec<String>>, Vec<i64>)> {
    let gene_key = gene_key.unwrap_or("gene");
    let gaps: Vec<i64> = gaps
        .map(|x| x.to_vec())
        .unwrap_or_else(|| vec![0, 1000, 10000, 100000]);
    let chroms = var_fixed_info.get("CHROM")?;
    let pos: Vec<i64> = match var_fixed_info.get("POS") {
        Some(v) => v.iter().filter_map(|x| x.parse().ok()).collect(),
        _ => return None,
    };
    if pos.len() != chroms.len() {
        return None;
    }
    if gene_key != "gene" {
        return None;
    }
    let gene_chrom = &gene_df.chrom;
    let gene_start = &gene_df.start;
    let gene_stop = &gene_df.stop;
    let gene_names = &gene_df.gene;
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
