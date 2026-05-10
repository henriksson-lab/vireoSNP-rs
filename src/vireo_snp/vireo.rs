use crate::vireo_snp::utils::io_utils;
use crate::vireo_snp::utils::vcf_utils;
use crate::vireo_snp::utils::vireo_wrap::{self, VireoWrapResult};
use ndarray::Axis;
use std::env;
use std::fs;
use std::path::Path;

pub fn show_progress<T>(rv: T) -> T {
    rv
}

pub fn main() -> Option<VireoWrapResult> {
    let args: Vec<String> = env::args().collect();
    if args.len() <= 1 {
        return None;
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
        return None;
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
        return None;
    }
    let mut cell_dat = if Path::new(&cell_data_path).is_dir() {
        match io_utils::read_cellSNP(&cell_data_path, None) {
            Some(m) => m,
            None => return None,
        }
    } else {
        let cell_vcf = match vcf_utils::load_VCF(&cell_data_path, true, true, true, None) {
            Some(m) => m,
            None => return None,
        };
        let layers = match &cell_vcf.geno_info {
            Some(value) => {
                let keys = ["AD".to_string(), "DP".to_string()];
                match vcf_utils::read_sparse_GeneINFO(value, Some(&keys), None) {
                    Some(m) => m
                        .into_iter()
                        .map(|(k, v)| (k, io_utils::CountMatrix::Dense(v)))
                        .collect(),
                    None => return None,
                }
            }
            _ => return None,
        };
        io_utils::CellData {
            variants: cell_vcf.variants,
            fixed_info: cell_vcf.fixed_info,
            contigs: cell_vcf.contigs,
            comments: cell_vcf.comments,
            samples: cell_vcf.samples,
            layers,
        }
    };
    let mut donor_gpb = None;
    let donor_names;
    let mut learn_gt = true;
    if let Some(donor_file) = donor_file {
        let geno_tags = [geno_tag.clone()];
        let donor_vcf = match vcf_utils::load_VCF(&donor_file, true, true, false, Some(&geno_tags))
        {
            Some(m) => m,
            None => return None,
        };
        let (cell_matched, donor_vcf) = match io_utils::match_donor_VCF(cell_dat, donor_vcf) {
            Some(v) => v,
            None => return None,
        };
        cell_dat = cell_matched;
        let geno = match &donor_vcf.geno_info {
            Some(m) => match m.string_matrices.get(&geno_tag) {
                Some(v) => v.clone(),
                None => return None,
            },
            _ => return None,
        };
        let donor_gpb_arr = match vcf_utils::parse_donor_GPb(&geno, &geno_tag, 0.0) {
            Some(x) => x,
            None => return None,
        };
        let donor_count = donor_gpb_arr.shape()[1] as i64;
        donor_gpb = Some(donor_gpb_arr);
        match n_donor {
            None => {
                n_donor = Some(donor_count);
                donor_names = donor_vcf.samples.clone();
                learn_gt = false;
            }
            Some(n) if n == donor_count => {
                donor_names = donor_vcf.samples.clone();
                learn_gt = false;
            }
            Some(n) if n < donor_count => {
                donor_names = (0..n).map(|x| format!("donor{x}")).collect();
                learn_gt = false;
            }
            Some(n) => {
                let mut names = donor_vcf.samples.clone();
                names.extend((donor_count..n).map(|x| format!("donor{x}")));
                donor_names = names;
                learn_gt = true;
            }
        }
    } else if let Some(n) = n_donor {
        donor_names = (0..n).map(|x| format!("donor{x}")).collect();
    } else {
        return None;
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
    let ad_arr = match cell_dat.layers.get("AD") {
        Some(io_utils::CountMatrix::Dense(v)) => v.clone(),
        Some(io_utils::CountMatrix::DenseU32(v)) => v.mapv(|x| x as f64),
        Some(io_utils::CountMatrix::SparseCsc {
            nrows,
            ncols,
            indptr,
            indices,
            data,
        }) => {
            let mut out = ndarray::Array2::<f64>::zeros((*nrows, *ncols));
            for col in 0..*ncols {
                for p in indptr[col]..indptr[col + 1] {
                    out[[indices[p], col]] = data[p];
                }
            }
            out
        }
        _ => return None,
    };
    let dp_arr = match cell_dat.layers.get("DP") {
        Some(io_utils::CountMatrix::Dense(v)) => v.clone(),
        Some(io_utils::CountMatrix::DenseU32(v)) => v.mapv(|x| x as f64),
        Some(io_utils::CountMatrix::SparseCsc {
            nrows,
            ncols,
            indptr,
            indices,
            data,
        }) => {
            let mut out = ndarray::Array2::<f64>::zeros((*nrows, *ncols));
            for col in 0..*ncols {
                for p in indptr[col]..indptr[col + 1] {
                    out[[indices[p], col]] = data[p];
                }
            }
            out
        }
        _ => return None,
    };
    let n_vars_vec: Vec<f64> = dp_arr
        .mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
        .sum_axis(Axis(0))
        .iter()
        .copied()
        .collect();
    let res = match vireo_wrap::vireo_wrap(
        &ad_arr,
        &dp_arr,
        donor_gpb.as_ref(),
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
        None => return None,
    };
    let cell_names = if cell_dat.samples.is_empty() {
        (0..dp_arr.ncols()).map(|i| format!("cell{i}")).collect()
    } else {
        cell_dat.samples.clone()
    };
    io_utils::write_donor_id(&out_dir, &donor_names, &cell_names, &n_vars_vec, &res);
    if learn_gt && !cell_dat.variants.is_empty() {
        let geno_info = match vcf_utils::GenoINFO_maker(
            &res.gt_prob,
            &ad_arr.dot(&res.id_prob),
            &dp_arr.dot(&res.id_prob),
        ) {
            Some(m) => m,
            None => return Some(res),
        };
        let out_dat = vcf_utils::VcfData {
            variants: cell_dat.variants,
            fixed_info: cell_dat.fixed_info,
            contigs: cell_dat.contigs,
            comments: cell_dat.comments,
            samples: donor_names,
            geno_info: Some(vcf_utils::VcfGenoInfo {
                string_matrices: geno_info,
                ..Default::default()
            }),
            n_snp_tagged: Vec::new(),
        };
        let out_vcf = format!("{out_dir}/GT_donors.vireo.vcf.gz");
        vcf_utils::write_VCF(&out_vcf, &out_dat, None);
    }
    Some(res)
}
