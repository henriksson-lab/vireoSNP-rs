use ndarray::s;
use std::env;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use vireosnp_rs::vireo_snp::utils::vireo_model::Vireo;
use vireosnp_rs::vireo_snp::utils::{io_utils, vcf_utils, vireo_wrap};

fn dense(mat: &io_utils::CountMatrix) -> ndarray::Array2<f64> {
    match mat {
        io_utils::CountMatrix::Dense(x) => x.clone(),
        io_utils::CountMatrix::DenseU32(x) => x.mapv(|v| v as f64),
        io_utils::CountMatrix::SparseCsc {
            nrows,
            ncols,
            indptr,
            indices,
            data,
        } => {
            let mut out = ndarray::Array2::<f64>::zeros((*nrows, *ncols));
            for col in 0..*ncols {
                for p in indptr[col]..indptr[col + 1] {
                    out[[indices[p], col]] = data[p];
                }
            }
            out
        }
    }
}

fn main() {
    let cell_snp_dir =
        env::var("CELL_SNP_DIR").unwrap_or_else(|_| "vireo/data/cellSNP_mat".to_string());
    let cells_vcf = if Path::new(&format!("{cell_snp_dir}/cellSNP.cells.vcf.gz")).exists() {
        format!("{cell_snp_dir}/cellSNP.cells.vcf.gz")
    } else if Path::new(&format!("{cell_snp_dir}/cellSNP.cells.vcf")).exists() {
        format!("{cell_snp_dir}/cellSNP.cells.vcf")
    } else {
        "vireo/data/cells.cellSNP.vcf.gz".to_string()
    };
    let repeats: usize = env::var("BENCH_REPEATS")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(10);
    let fit_repeats = repeats.min(5).max(1);
    let mut stdout = std::io::stdout();

    let mut samples = Vec::new();
    for _ in 0..repeats {
        let t0 = Instant::now();
        let _ = vcf_utils::load_VCF(&cells_vcf, false, true, true, None).unwrap();
        samples.push(t0.elapsed().as_secs_f64());
    }
    samples.sort_by(f64::total_cmp);
    println!("rust\tload_cells_vcf\t{:.9}", samples[samples.len() / 2]);
    stdout.flush().unwrap();

    let mut samples = Vec::new();
    for _ in 0..repeats {
        let t0 = Instant::now();
        let _ = io_utils::read_cellSNP(&cell_snp_dir, Some(&["AD".to_string(), "DP".to_string()]))
            .unwrap();
        samples.push(t0.elapsed().as_secs_f64());
    }
    samples.sort_by(f64::total_cmp);
    println!("rust\tread_cellsnp\t{:.9}", samples[samples.len() / 2]);
    stdout.flush().unwrap();

    let mut samples = Vec::new();
    for _ in 0..repeats {
        let cell_dat =
            io_utils::read_cellSNP(&cell_snp_dir, Some(&["AD".to_string(), "DP".to_string()]))
                .unwrap();
        let donor_vcf = vcf_utils::load_VCF(
            "vireo/data/donors.two.cellSNP.vcf.gz",
            false,
            true,
            false,
            Some(&["GT".to_string(), "PL".to_string()]),
        )
        .unwrap();
        let t0 = Instant::now();
        let _ = io_utils::match_donor_VCF(cell_dat, donor_vcf).unwrap();
        samples.push(t0.elapsed().as_secs_f64());
    }
    samples.sort_by(f64::total_cmp);
    println!("rust\tmatch_donor_vcf\t{:.9}", samples[samples.len() / 2]);
    stdout.flush().unwrap();

    let dat =
        io_utils::read_cellSNP(&cell_snp_dir, Some(&["AD".to_string(), "DP".to_string()])).unwrap();
    let n_var80 = 80.min(dat.variants.len());
    let n_cell60 = 60.min(dat.samples.len());
    let ad80 = dense(dat.layers.get("AD").unwrap())
        .slice(s![0..n_var80, 0..n_cell60])
        .to_owned();
    let dp80 = dense(dat.layers.get("DP").unwrap())
        .slice(s![0..n_var80, 0..n_cell60])
        .to_owned();
    let mut samples = Vec::new();
    for _ in 0..fit_repeats {
        let mut model = Vireo::default();
        model
            .__init__(
                n_cell60, n_var80, 2, 3, true, true, false, false, None, None, None, None,
            )
            .unwrap();
        let t0 = Instant::now();
        model
            .fit(&ad80, &dp80, 4, 1, Some(1e-2), 1, false, None, 1)
            .unwrap();
        samples.push(t0.elapsed().as_secs_f64());
    }
    samples.sort_by(f64::total_cmp);
    println!("rust\tvireo_fit_slice\t{:.9}", samples[samples.len() / 2]);
    stdout.flush().unwrap();

    let n_var50 = 50.min(dat.variants.len());
    let n_cell24 = 24.min(dat.samples.len());
    let ad50 = dense(dat.layers.get("AD").unwrap())
        .slice(s![0..n_var50, 0..n_cell24])
        .to_owned();
    let dp50 = dense(dat.layers.get("DP").unwrap())
        .slice(s![0..n_var50, 0..n_cell24])
        .to_owned();
    let mut samples = Vec::new();
    for _ in 0..fit_repeats {
        let t0 = Instant::now();
        let _ = vireo_wrap::vireo_wrap(
            &ad50,
            &dp50,
            None,
            Some(2),
            true,
            1,
            Some(0),
            false,
            3,
            1,
            0,
            Some("distance"),
            false,
            1,
            false,
            false,
            3,
        )
        .unwrap();
        samples.push(t0.elapsed().as_secs_f64());
    }
    samples.sort_by(f64::total_cmp);
    println!("rust\tvireo_wrap_slice\t{:.9}", samples[samples.len() / 2]);
    stdout.flush().unwrap();
}
