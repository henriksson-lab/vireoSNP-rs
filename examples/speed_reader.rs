use std::env;
use std::path::Path;
use std::time::Instant;
use vireosnp_rs::vireo_snp::utils::{io_utils, vcf_utils};

fn main() {
    let cell_snp_dir =
        env::var("CELL_SNP_DIR").unwrap_or_else(|_| "vireo/data/cellSNP_mat".to_string());
    let base_vcf = if Path::new(&format!("{cell_snp_dir}/cellSNP.base.vcf.gz")).exists() {
        format!("{cell_snp_dir}/cellSNP.base.vcf.gz")
    } else {
        format!("{cell_snp_dir}/cellSNP.base.vcf")
    };
    let t0 = Instant::now();
    let _ = vcf_utils::load_VCF(&base_vcf, false, false, true, None).unwrap();
    println!("reader\tbase_vcf\t{:.9}", t0.elapsed().as_secs_f64());
    let t0 = Instant::now();
    let _ = io_utils::read_cellSNP(&cell_snp_dir, Some(&["AD".to_string()])).unwrap();
    println!("reader\tread_ad\t{:.9}", t0.elapsed().as_secs_f64());
    let t0 = Instant::now();
    let _ = io_utils::read_cellSNP(&cell_snp_dir, Some(&["DP".to_string()])).unwrap();
    println!("reader\tread_dp\t{:.9}", t0.elapsed().as_secs_f64());
    let t0 = Instant::now();
    let _ =
        io_utils::read_cellSNP(&cell_snp_dir, Some(&["AD".to_string(), "DP".to_string()])).unwrap();
    println!("reader\tread_ad_dp\t{:.9}", t0.elapsed().as_secs_f64());
}
