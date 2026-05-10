use crate::vireo_snp::utils::io_utils;
use crate::vireo_snp::utils::vcf_utils;
use crate::vireo_snp::utils::vireo_wrap::{self, VireoWrapResult};
use ndarray::Axis;
#[cfg(feature = "cli")]
use std::env;
use std::fmt;
use std::fs;
use std::path::Path;

pub fn show_progress<T>(rv: T) -> T {
    rv
}

pub type Result<T> = std::result::Result<T, VireoError>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VireoError {
    MissingCellData,
    MissingDonorConfig,
    CreateOutputDirFailed(String),
    ReadCellSnpFailed(String),
    ReadCellVcfFailed(String),
    ReadCellVcfGenoInfoFailed(String),
    ReadDonorVcfFailed(String),
    VariantMatchFailed,
    MissingGenotypeTag(String),
    ParseDonorGenotypeFailed(String),
    MissingLayer(&'static str),
    ModelFitFailed,
    DonorOutputFailed(String),
    DonorVcfOutputFailed(String),
    FitFailed,
    OutputFailed(String),
}

impl fmt::Display for VireoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VireoError::MissingCellData => write!(f, "cellSNP data path is required"),
            VireoError::MissingDonorConfig => {
                write!(f, "either a donor VCF or a donor count is required")
            }
            VireoError::CreateOutputDirFailed(path) => {
                write!(f, "failed to create output directory {path}")
            }
            VireoError::ReadCellSnpFailed(path) => {
                write!(f, "failed to read cellSNP directory {path}")
            }
            VireoError::ReadCellVcfFailed(path) => write!(f, "failed to read cell VCF {path}"),
            VireoError::ReadCellVcfGenoInfoFailed(path) => {
                write!(f, "failed to materialize AD/DP layers from cell VCF {path}")
            }
            VireoError::ReadDonorVcfFailed(path) => write!(f, "failed to read donor VCF {path}"),
            VireoError::VariantMatchFailed => write!(f, "failed to match cell and donor variants"),
            VireoError::MissingGenotypeTag(tag) => {
                write!(f, "donor VCF is missing genotype tag {tag}")
            }
            VireoError::ParseDonorGenotypeFailed(tag) => {
                write!(f, "failed to parse donor genotype tag {tag}")
            }
            VireoError::MissingLayer(layer) => write!(f, "cell data is missing {layer} layer"),
            VireoError::ModelFitFailed => write!(f, "vireo model fit failed"),
            VireoError::DonorOutputFailed(path) => {
                write!(f, "failed to write donor assignment outputs to {path}")
            }
            VireoError::DonorVcfOutputFailed(path) => {
                write!(f, "failed to write donor genotype VCF to {path}")
            }
            VireoError::FitFailed => write!(f, "vireo fit failed"),
            VireoError::OutputFailed(path) => write!(f, "failed to write vireo outputs to {path}"),
        }
    }
}

impl std::error::Error for VireoError {}

pub type FitBuilder = VireoSnpBuilder;
pub type FitResult = VireoSnpResult;

pub fn fit(cell_data: impl Into<String>) -> FitBuilder {
    VireoSnpBuilder::new().cell_data(cell_data)
}

#[derive(Clone, Debug)]
pub struct VireoSnpBuilder {
    pub cell_data: Option<String>,
    pub donor_file: Option<String>,
    pub n_donor: Option<i64>,
    pub out_dir: Option<String>,
    pub geno_tag: String,
    pub no_doublet: bool,
    pub n_init: i64,
    pub n_extra_donor: i64,
    pub extra_donor_mode: String,
    pub force_learn_gt: bool,
    pub ase_mode: bool,
    pub check_ambient: bool,
    pub rand_seed: Option<u64>,
    pub write_outputs: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VireoSnpResult {
    pub result: VireoWrapResult,
    pub donor_names: Vec<String>,
    pub cell_names: Vec<String>,
    pub n_vars: Vec<f64>,
    pub out_dir: Option<String>,
}

impl Default for VireoSnpBuilder {
    fn default() -> Self {
        Self {
            cell_data: None,
            donor_file: None,
            n_donor: None,
            out_dir: None,
            geno_tag: "PL".to_string(),
            no_doublet: false,
            n_init: 50,
            n_extra_donor: 0,
            extra_donor_mode: "distance".to_string(),
            force_learn_gt: false,
            ase_mode: false,
            check_ambient: false,
            rand_seed: None,
            write_outputs: false,
        }
    }
}

impl VireoSnpBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cell_data(mut self, path: impl Into<String>) -> Self {
        self.cell_data = Some(path.into());
        self
    }

    pub fn donor_file(mut self, path: impl Into<String>) -> Self {
        self.donor_file = Some(path.into());
        self
    }

    pub fn with_donors(self, path: impl Into<String>) -> Self {
        self.donor_file(path)
    }

    pub fn n_donor(mut self, value: i64) -> Self {
        self.n_donor = Some(value);
        self
    }

    pub fn infer_donors(self, value: i64) -> Self {
        self.n_donor(value)
    }

    pub fn out_dir(mut self, path: impl Into<String>) -> Self {
        self.out_dir = Some(path.into());
        self.write_outputs = true;
        self
    }

    pub fn geno_tag(mut self, value: impl Into<String>) -> Self {
        self.geno_tag = value.into();
        self
    }

    pub fn genotype_tag(self, value: impl Into<String>) -> Self {
        self.geno_tag(value)
    }

    pub fn no_doublet(mut self, value: bool) -> Self {
        self.no_doublet = value;
        self
    }

    pub fn doublets(mut self, value: bool) -> Self {
        self.no_doublet = !value;
        self
    }

    pub fn learn_genotypes(mut self, value: bool) -> Self {
        self.force_learn_gt = value;
        self
    }

    pub fn n_init(mut self, value: i64) -> Self {
        self.n_init = value;
        self
    }

    pub fn extra_donor(mut self, value: i64) -> Self {
        self.n_extra_donor = value;
        self
    }

    pub fn extra_donor_mode(mut self, value: impl Into<String>) -> Self {
        self.extra_donor_mode = value.into();
        self
    }

    pub fn force_learn_gt(mut self, value: bool) -> Self {
        self.force_learn_gt = value;
        self
    }

    pub fn ase_mode(mut self, value: bool) -> Self {
        self.ase_mode = value;
        self
    }

    pub fn check_ambient(mut self, value: bool) -> Self {
        self.check_ambient = value;
        self
    }

    pub fn rand_seed(mut self, value: u64) -> Self {
        self.rand_seed = Some(value);
        self
    }

    pub fn seed(self, value: u64) -> Self {
        self.rand_seed(value)
    }

    pub fn write_outputs(mut self, value: bool) -> Self {
        self.write_outputs = value;
        self
    }

    pub fn fit(self) -> Option<VireoSnpResult> {
        self.run().ok()
    }

    pub fn run(mut self) -> Result<FitResult> {
        let cell_data_path = self.cell_data.clone().ok_or(VireoError::MissingCellData)?;
        let out_dir = self.out_dir.clone().or_else(|| {
            if self.write_outputs {
                let input = fs::canonicalize(&cell_data_path)
                    .unwrap_or_else(|_| Path::new(&cell_data_path).to_path_buf());
                Some(
                    input
                        .parent()
                        .unwrap_or_else(|| Path::new("."))
                        .join("vireo")
                        .to_string_lossy()
                        .into_owned(),
                )
            } else {
                None
            }
        });
        if let Some(out_dir) = &out_dir {
            if fs::create_dir_all(out_dir).is_err() {
                return Err(VireoError::CreateOutputDirFailed(out_dir.clone()));
            }
        }
        let mut cell_dat = if Path::new(&cell_data_path).is_dir() {
            io_utils::read_cellSNP(&cell_data_path, None)
                .ok_or_else(|| VireoError::ReadCellSnpFailed(cell_data_path.clone()))?
        } else {
            let cell_vcf = vcf_utils::load_VCF(&cell_data_path, true, true, true, None)
                .ok_or_else(|| VireoError::ReadCellVcfFailed(cell_data_path.clone()))?;
            let layers = match &cell_vcf.geno_info {
                Some(value) => {
                    let keys = ["AD".to_string(), "DP".to_string()];
                    vcf_utils::read_sparse_GeneINFO(value, Some(&keys), None)
                        .ok_or_else(|| {
                            VireoError::ReadCellVcfGenoInfoFailed(cell_data_path.clone())
                        })?
                        .into_iter()
                        .map(|(k, v)| (k, io_utils::CountMatrix::Dense(v)))
                        .collect()
                }
                _ => {
                    return Err(VireoError::ReadCellVcfGenoInfoFailed(
                        cell_data_path.clone(),
                    ))
                }
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
        if let Some(donor_file) = self.donor_file {
            let geno_tags = [self.geno_tag.clone()];
            let donor_vcf = vcf_utils::load_VCF(&donor_file, true, true, false, Some(&geno_tags))
                .ok_or_else(|| VireoError::ReadDonorVcfFailed(donor_file.clone()))?;
            let (cell_matched, donor_vcf) = io_utils::match_donor_VCF(cell_dat, donor_vcf)
                .ok_or(VireoError::VariantMatchFailed)?;
            cell_dat = cell_matched;
            let geno = match &donor_vcf.geno_info {
                Some(m) => m
                    .string_matrices
                    .get(&self.geno_tag)
                    .ok_or_else(|| VireoError::MissingGenotypeTag(self.geno_tag.clone()))?
                    .clone(),
                _ => return Err(VireoError::MissingGenotypeTag(self.geno_tag.clone())),
            };
            let donor_gpb_arr = vcf_utils::parse_donor_GPb(&geno, &self.geno_tag, 0.0)
                .ok_or_else(|| VireoError::ParseDonorGenotypeFailed(self.geno_tag.clone()))?;
            let donor_count = donor_gpb_arr.shape()[1] as i64;
            donor_gpb = Some(donor_gpb_arr);
            match self.n_donor {
                None => {
                    self.n_donor = Some(donor_count);
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
        } else if let Some(n) = self.n_donor {
            donor_names = (0..n).map(|x| format!("donor{x}")).collect();
        } else {
            return Err(VireoError::MissingDonorConfig);
        }
        if self.force_learn_gt {
            learn_gt = true;
        }
        let n_donor = self.n_donor.unwrap_or(donor_names.len() as i64);
        let n_extra_donor = if learn_gt && self.n_extra_donor == 0 {
            (n_donor as f64).sqrt().round() as i64
        } else if learn_gt {
            self.n_extra_donor
        } else {
            0
        };
        let n_init = if learn_gt { self.n_init } else { 1 };
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
            _ => return Err(VireoError::MissingLayer("AD")),
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
            _ => return Err(VireoError::MissingLayer("DP")),
        };
        let n_vars_vec: Vec<f64> = dp_arr
            .mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
            .sum_axis(Axis(0))
            .iter()
            .copied()
            .collect();
        let res = vireo_wrap::vireo_wrap(
            &ad_arr,
            &dp_arr,
            donor_gpb.as_ref(),
            Some(n_donor as usize),
            learn_gt,
            n_init as usize,
            self.rand_seed,
            !self.no_doublet,
            20,
            3,
            n_extra_donor as usize,
            Some(&self.extra_donor_mode),
            self.check_ambient,
            1,
            self.ase_mode,
            false,
            3,
        )
        .ok_or(VireoError::ModelFitFailed)?;
        let cell_names = if cell_dat.samples.is_empty() {
            (0..dp_arr.ncols()).map(|i| format!("cell{i}")).collect()
        } else {
            cell_dat.samples.clone()
        };
        if let Some(out_dir) = &out_dir {
            if io_utils::write_donor_id(out_dir, &donor_names, &cell_names, &n_vars_vec, &res)
                .is_none()
            {
                return Err(VireoError::DonorOutputFailed(out_dir.clone()));
            }
        }
        if let Some(out_dir) = out_dir
            .as_ref()
            .filter(|_| learn_gt && !cell_dat.variants.is_empty())
        {
            let geno_info = vcf_utils::GenoINFO_maker(
                &res.gt_prob,
                &ad_arr.dot(&res.id_prob),
                &dp_arr.dot(&res.id_prob),
            )
            .ok_or_else(|| VireoError::DonorVcfOutputFailed(out_dir.clone()))?;
            let out_dat = vcf_utils::VcfData {
                variants: cell_dat.variants,
                fixed_info: cell_dat.fixed_info,
                contigs: cell_dat.contigs,
                comments: cell_dat.comments,
                samples: donor_names.clone(),
                geno_info: Some(vcf_utils::VcfGenoInfo {
                    string_matrices: geno_info,
                    ..Default::default()
                }),
                n_snp_tagged: Vec::new(),
            };
            let out_vcf = format!("{out_dir}/GT_donors.vireo.vcf.gz");
            if vcf_utils::write_VCF(&out_vcf, &out_dat, None).is_none() {
                return Err(VireoError::DonorVcfOutputFailed(out_vcf));
            }
        }
        Ok(VireoSnpResult {
            result: res,
            donor_names,
            cell_names,
            n_vars: n_vars_vec,
            out_dir,
        })
    }
}

impl VireoSnpResult {
    pub fn write_outputs(&self, out_dir: impl AsRef<str>) -> Result<()> {
        let out_dir = out_dir.as_ref();
        if fs::create_dir_all(out_dir).is_err() {
            return Err(VireoError::OutputFailed(out_dir.to_string()));
        }
        if io_utils::write_donor_id(
            out_dir,
            &self.donor_names,
            &self.cell_names,
            &self.n_vars,
            &self.result,
        )
        .is_none()
        {
            return Err(VireoError::OutputFailed(out_dir.to_string()));
        }
        Ok(())
    }
}

#[cfg(feature = "cli")]
pub fn main() -> Option<VireoWrapResult> {
    let args: Vec<String> = env::args().collect();
    if args.len() <= 1 {
        return None;
    }
    let mut builder = VireoSnpBuilder::new().write_outputs(true);
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--cellData" | "-c" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    builder = builder.cell_data(v.clone());
                }
            }
            "--donorFile" | "-d" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    builder = builder.donor_file(v.clone());
                }
            }
            "--nDonor" | "-N" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|v| v.parse::<i64>().ok()) {
                    builder = builder.n_donor(v);
                }
            }
            "--outDir" | "-o" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    builder = builder.out_dir(v.clone());
                }
            }
            "--genoTag" | "-t" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    builder = builder.geno_tag(v.clone());
                }
            }
            "--noDoublet" => builder = builder.no_doublet(true),
            "--nInit" | "-M" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|v| v.parse::<i64>().ok()) {
                    builder = builder.n_init(v);
                }
            }
            "--extraDonor" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|v| v.parse::<i64>().ok()) {
                    builder = builder.extra_donor(v);
                }
            }
            "--extraDonorMode" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    builder = builder.extra_donor_mode(v.clone());
                }
            }
            "--forceLearnGT" => builder = builder.force_learn_gt(true),
            "--ASEmode" => builder = builder.ase_mode(true),
            "--callAmbientRNAs" => builder = builder.check_ambient(true),
            "--randSeed" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|v| v.parse::<u64>().ok()) {
                    builder = builder.rand_seed(v);
                }
            }
            _ => {}
        }
        i += 1;
    }
    builder.fit().map(|x| x.result)
}
