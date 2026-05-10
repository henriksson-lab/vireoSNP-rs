use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::BTreeSet;
#[cfg(feature = "cli")]
use std::env;
#[cfg(feature = "cli")]
use std::fs;
use std::fs::File;
use std::io::Write;
#[cfg(feature = "cli")]
use std::io::{BufRead, BufReader};
#[cfg(feature = "cli")]
use std::process::Command;

pub fn show_progress<T>(rv: T) -> T {
    rv
}

pub fn sample_barcodes(
    mut barcodes: Vec<Vec<String>>,
    n_cell_each: usize,
    minor_sample: f64,
    seed: Option<u64>,
) -> Option<Vec<Vec<String>>> {
    let seed = seed.unwrap_or(0);
    let mut rng = StdRng::seed_from_u64(seed);
    for sample in &mut barcodes {
        if sample.len() < n_cell_each {
            return None;
        }
        sample.shuffle(&mut rng);
        sample.truncate(n_cell_each);
    }
    if let Some(first) = barcodes.first_mut() {
        first.truncate((minor_sample * n_cell_each as f64).round() as usize);
    }
    Some(barcodes)
}

pub fn pool_barcodes(
    barcodes: &[Vec<String>],
    out_dir: &str,
    doublet_rate: Option<f64>,
    sample_suffix: bool,
    seed: Option<u64>,
) -> Option<Vec<Vec<String>>> {
    let seed = seed.unwrap_or(0);
    let mut barcodes_out = if sample_suffix {
        barcodes
            .iter()
            .enumerate()
            .map(|(ss, sample)| {
                sample
                    .iter()
                    .map(|x| {
                        if x.is_empty() {
                            (ss + 1).to_string()
                        } else {
                            format!("{}{}", &x[..x.len() - 1], ss + 1)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    } else {
        barcodes.to_vec()
    };
    let mut flat: Vec<String> = barcodes_out.iter().flatten().cloned().collect();
    let n_cells = flat.len();
    let doublet_rate = doublet_rate.unwrap_or(n_cells as f64 / 100000.0);
    if !(0.0..=1.0).contains(&doublet_rate) {
        return None;
    }
    let n_doublets = if doublet_rate == 0.0 {
        0
    } else {
        (n_cells as f64 / (1.0 + 1.0 / doublet_rate)).round() as usize
    };
    let mut perm_idx: Vec<usize> = (0..n_cells).collect();
    perm_idx.shuffle(&mut StdRng::seed_from_u64(seed));
    for ii in 0..n_doublets {
        let a = perm_idx[ii];
        let b = perm_idx[ii + n_doublets];
        let same_sample = flat[a].split('-').nth(1) == flat[b].split('-').nth(1);
        let barcode = format!("{}{}", flat[a], if same_sample { "S" } else { "D" });
        flat[a] = barcode.clone();
        flat[b] = barcode;
    }
    let mut start = 0;
    for sample in &mut barcodes_out {
        let n = sample.len();
        *sample = flat[start..start + n].to_vec();
        start += n;
    }
    let unique: BTreeSet<String> = flat.into_iter().collect();
    let mut f = match File::create(format!("{out_dir}/barcodes_pool.tsv")) {
        Ok(f) => f,
        Err(_) => return None,
    };
    for barcode in unique {
        if writeln!(f, "{barcode}").is_err() {
            return None;
        }
    }
    let mut f = match File::create(format!("{out_dir}/cell_info.tsv")) {
        Ok(f) => f,
        Err(_) => return None,
    };
    if writeln!(f, "CB_pool\tCB_origin\tSample_id").is_err() {
        return None;
    }
    for (ss, sample) in barcodes_out.iter().enumerate() {
        for (ii, cb_pool) in sample.iter().enumerate() {
            if writeln!(f, "{}\t{}\t{}", cb_pool, barcodes[ss][ii], ss + 1).is_err() {
                return None;
            }
        }
    }
    Some(barcodes_out)
}

pub fn fetch_reads(
    _samfile_list: &[String],
    _chroms: &[String],
    _positions: &[i64],
    _outbam: &str,
    _barcodes_in: &[Vec<String>],
    _barcodes_out: Option<&[Vec<String>]>,
    _cell_tag: &str,
    _test_val: i64,
) -> Option<()> {
    None
}

pub fn merge_bams(
    _samfile_list: &[String],
    _outbam: &str,
    _barcodes_in: &[Vec<String>],
    _barcodes_out: Option<&[Vec<String>]>,
    _cell_tag: &str,
) -> Option<()> {
    None
}

#[cfg(feature = "cli")]
pub fn main() -> Option<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() <= 1 {
        return None;
    }
    let mut sam_files = None;
    let mut barcodes_files = None;
    let mut region_file = None;
    let mut noregion_file = false;
    let mut doublet_rate = None;
    let mut out_dir = None;
    let mut shuffle = false;
    let mut test_val = -1;
    let mut n_cell = None;
    let mut minor_sample = 1.0;
    let mut random_seed = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--samFiles" | "-s" => {
                i += 1;
                sam_files = args.get(i).cloned();
            }
            "--barcodeFiles" | "-b" => {
                i += 1;
                barcodes_files = args.get(i).cloned();
            }
            "--regionFile" | "-r" => {
                i += 1;
                region_file = args.get(i).cloned();
            }
            "--noregionFile" => noregion_file = true,
            "--doubletRate" | "-d" => {
                i += 1;
                doublet_rate = args.get(i).and_then(|v| v.parse::<f64>().ok()).or(None);
            }
            "--outDir" | "-o" => {
                i += 1;
                out_dir = args.get(i).cloned();
            }
            "--nproc" | "-p" => i += 1,
            "--shuffle" => shuffle = true,
            "--test" => {
                i += 1;
                test_val = args
                    .get(i)
                    .and_then(|v| v.parse::<i64>().ok())
                    .unwrap_or(-1);
            }
            "--nCELL" => {
                i += 1;
                n_cell = args.get(i).and_then(|v| v.parse::<i64>().ok());
            }
            "--minorSAMPLE" => {
                i += 1;
                minor_sample = args
                    .get(i)
                    .and_then(|v| v.parse::<f64>().ok())
                    .unwrap_or(1.0);
            }
            "--randomSEED" => {
                i += 1;
                random_seed = args.get(i).and_then(|v| v.parse::<u64>().ok());
            }
            _ => {}
        }
        i += 1;
    }
    if noregion_file && region_file.is_some() {
        return None;
    }
    let Some(out_dir) = out_dir else {
        return None;
    };
    if fs::create_dir_all(&out_dir).is_err() {
        return None;
    }
    let sam_list: Vec<String> = match sam_files {
        Some(v) => v.split(',').map(|s| s.to_string()).collect(),
        None => return None,
    };
    let barcode_files: Vec<String> = match barcodes_files {
        Some(v) => v.split(',').map(|s| s.to_string()).collect(),
        None => return None,
    };
    if barcode_files.len() != sam_list.len() {
        return None;
    }
    let mut barcodes_in = Vec::new();
    for path in barcode_files {
        let file = match File::open(path) {
            Ok(file) => file,
            Err(_) => return None,
        };
        barcodes_in.push(
            BufReader::new(file)
                .lines()
                .map_while(Result::ok)
                .collect::<Vec<_>>(),
        );
    }
    let barcodes_in = if let Some(n_cell) = n_cell {
        match sample_barcodes(barcodes_in, n_cell as usize, minor_sample, random_seed) {
            Some(v) => v,
            None => return None,
        }
    } else {
        barcodes_in
    };
    let barcodes_out = match pool_barcodes(&barcodes_in, &out_dir, doublet_rate, true, random_seed)
    {
        Some(v) => v,
        None => return None,
    };
    if noregion_file {
        merge_bams(
            &sam_list,
            &format!("{out_dir}/pooled.bam"),
            &barcodes_in,
            Some(&barcodes_out),
            "CB",
        );
    } else {
        let Some(region_file) = region_file else {
            return None;
        };
        let vcf = match crate::vireo_snp::utils::vcf_utils::load_VCF(
            &region_file,
            false,
            false,
            true,
            None,
        ) {
            Some(m) => m,
            None => return None,
        };
        let mut chroms = match vcf.fixed_info.get("CHROM") {
            Some(v) => v.clone(),
            _ => return None,
        };
        let mut positions: Vec<i64> = match vcf.fixed_info.get("POS") {
            Some(v) => v.iter().filter_map(|x| x.parse().ok()).collect(),
            _ => return None,
        };
        if shuffle {
            let mut pairs: Vec<(String, i64)> = chroms.into_iter().zip(positions).collect();
            pairs.shuffle(&mut StdRng::seed_from_u64(match random_seed {
                Some(v) => v,
                None => 0,
            }));
            let (c, p): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
            chroms = c;
            positions = p;
        }
        fetch_reads(
            &sam_list,
            &chroms,
            &positions,
            &format!("{out_dir}/pooled.bam"),
            &barcodes_in,
            Some(&barcodes_out),
            "CB",
            test_val,
        );
    }
    let _ = Command::new("samtools")
        .args([
            "sort",
            &format!("{out_dir}/pooled.bam"),
            "-o",
            &format!("{out_dir}/pooled.sorted.bam"),
        ])
        .status();
    let _ = Command::new("samtools")
        .args(["index", &format!("{out_dir}/pooled.sorted.bam")])
        .status();
    let _ = fs::remove_file(format!("{out_dir}/pooled.bam"));
    Some(())
}
