use vireosnp_rs::{fit, Result};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(cell_snp_dir) = args.next() else {
        eprintln!(
            "usage: cargo run --example high_level -- <cellSNP-dir> [donor.vcf.gz] [out-dir]"
        );
        return Ok(());
    };
    let mut builder = fit(cell_snp_dir).genotype_tag("GT").seed(1);
    if let Some(donor_vcf) = args.next() {
        builder = builder.with_donors(donor_vcf);
    } else {
        builder = builder.infer_donors(2);
    }
    let result = builder.run()?;
    if let Some(out_dir) = args.next() {
        result.write_outputs(out_dir)?;
    }
    println!(
        "{} cells, {} donors",
        result.cell_names.len(),
        result.donor_names.len()
    );
    Ok(())
}
