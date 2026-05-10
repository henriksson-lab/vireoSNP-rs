# vireosnp-rs

Faithful Rust translation of `vireoSNP`, commit  `e3654633f7663732572c03c5dcf9fb00ec43b653`

* 2026-05-10: Possibly working, near 4x faster primarily due to differences in VCF parsing. May add support for more compact file formats in the future. Same output for 10M reads but need to check corner cases better. Be vigilant to early translation errors


## This is an LLM-mediated faithful (hopefully) translation, not the original code! 

Most users should probably first see if the existing original code works for them, unless they have reason otherwise. The original source
may have newer features and it has had more love in terms of fixing bugs. In fact, we aim to replicate bugs if they are present, for the
sake of reproducibility! (but then we might have added a few more in the process)

There are however cases when you might prefer this Rust version. We generally agree with [this manifesto](https://rewrites.bio/) but more specifically:
* We have had many issues with ensuring that our software works using existing containers (Docker, PodMan, Singularity). One size does not fit all and it eats our resources trying to keep up with every way of delivering software
* Common package managers do not work well. It was great when we had a few Linux distributions with stable procedures, but now there are just too many ecosystems (Homebrew, Conda). Conda has an NP-complete resolver which does not scale. Homebrew is only so-stable. And our dependencies in Python still break. These can no longer be considered professional serious options. Meanwhile, Cargo enables multiple versions of packages to be available, even within the same program(!)
* The future is the web. We deploy software in the web browser, and until now that has meant Javascript. This is a language where even the == operator is broken. Typescript is one step up, but a game changer is the ability to compile Rust code into webassembly, enabling performance and sharing of code with the backend. Translating code to Rust enables new ways of deployment and running code in the browser has especial benefits for science - researchers do not have deep pockets to run servers, so pushing compute to the user enables deployment that otherwise would be impossible
* Old CLI-based utilities are bad for the environment(!). A large amount of compute resources are spent creating and communicating via small files, which we can bypass by using code as libraries. Even better, we can avoid frequent reloading of databases by hoisting this stage, with up to 100x speedups in some cases. Less compute means faster compute and less electricity wasted
* LLM-mediated translations may actually be safer to use than the original code. This article shows that [running the same code on different operating systems can give somewhat different answers](https://doi.org/10.1038/nbt.3820). This is a gap that Rust+Cargo can reduce. Typesafe interfaces also reduce coding mistakes and error handling, as opposed to typical command-line scripting

But:

* **This approach should still be considered experimental**. The LLM technology is immature and has sharp corners. But there are opportunities to reap, and the genie is not going back into the bottle. This translation is as much aimed to learn how to improve the technology and get feedback on the results.
* Translations are not endorsed by the original authors unless otherwise noted. **Do not send bug reports to the original developers**. Use our Github issues page instead.
* **Do not trust the benchmarks on this page**. They are used to help evaluate the translation. If you want improved performance, you generally have to use this code as a library, and use the additional tricks it offers. We generally accept performance losses in order to reduce our dependency issues
* **Check the original Github pages for information about the package**. This README is kept sparse on purpose. It is not meant to be the primary source of information
* **If you are the author of the original code and wish to move to Rust, you can obtain ownership of this repository and crate**. Until then, our commitment is to offer an as-faithful-as-possible translation of a snapshot of your code. If we find serious bugs, we will report them to you. Otherwise we will just replicate them, to ensure comparability across studies that claim to use package XYZ v.666. Think of this like a fancy Ubuntu .deb-package of your software - that is how we treat it

This blurb might be out of date. Go to [this page](https://github.com/henriksson-lab/rustification) for the latest information and further information about how we approach translation

## Usage

### Library

Add the crate with default features disabled:

```toml
[dependencies]
vireosnp-rs = { version = "0.1", default-features = false }
```

The high-level API covers the main Vireo workflow:

```rust
use vireosnp_rs::{fit, Result};

fn main() -> Result<()> {
    let result = fit("cellSnp2bOutput")
        .with_donors("donors.cellSNP.vcf.gz")
        .genotype_tag("GT")
        .seed(1)
        .run()?;

    println!("{} cells assigned", result.cell_names.len());
    Ok(())
}
```

To infer donors without a donor VCF:

```rust
use vireosnp_rs::{fit, Result};

fn main() -> Result<()> {
    let result = fit("cellSnp2bOutput")
        .infer_donors(4)
        .doublets(true)
        .run()?;

    result.write_outputs("vireo_out")?;
    Ok(())
}
```

The Rust library name is `vireosnp_rs`, matching the package name `vireosnp-rs`.

Lower-level translated functions remain available under `vireosnp_rs::vireo_snp::utils` for audit-oriented or custom workflows.

The same high-level API is available as a runnable example:

```sh
cargo run --example high_level -- cellSnp2bOutput donors.cellSNP.vcf.gz vireo_out
```

### CLI

The command-line entry points are optional and are disabled by default. Install them with the `cli` feature:

```sh
cargo install vireosnp-rs --features cli
```

Then run the translated entry points:

```sh
vireoSNP \
  --cellData cellSnp2bOutput \
  --donorFile donors.cellSNP.vcf.gz \
  --outDir vireo_out

GTbarcode \
  --vcfFile donors.cellSNP.vcf.gz \
  --outFile GTbarcode.tsv

synth_pool \
  --samFiles sample1.bam,sample2.bam \
  --barcodeFiles sample1.barcodes.tsv,sample2.barcodes.tsv \
  --noregionFile \
  --outDir pooled
```

## Benchmarks

The comparison below uses a 10M-scale real cellSNP output directory with 1,848,727 variants, 8 cells, 14,061,090 AD matrix entries, and 14,789,371 DP matrix entries. It was run on an Intel Xeon Gold 6138 CPU @ 2.00 GHz with:

```sh
cargo build --release --examples

/usr/bin/time -v env \
  BENCH_REPEATS=1 \
  CELL_SNP_DIR=/path/to/cellSNP/output \
  target/release/examples/speed_rust

/usr/bin/time -v env \
  BENCH_REPEATS=1 \
  CELL_SNP_DIR=/path/to/cellSNP/output \
  python3 benches/speed_python.py
```

Wall time by benchmark task:

```text
task               Rust       original Python   speedup
load_cells_vcf      8.788 s    49.177 s          5.6x
read_cellsnp        3.143 s     3.603 s          1.1x
match_donor_vcf     2.457 s     6.181 s          2.5x
vireo_fit_slice     0.000342 s  0.011 s         32.5x
vireo_wrap_slice    0.000441 s  0.261 s        591.4x
```

Whole benchmark process:

```text
implementation     wall time    speed ratio    max RSS
Rust               18.77 s      3.6x faster    6.24 GiB
original Python    67.82 s      1.0x          8.45 GiB
```

These numbers are workload- and machine-specific. The benchmark scripts are included so the comparison can be rerun on local data.


## Citation

Yuanhua Huang, Davis J. McCarthy, and Oliver Stegle. Vireo: Bayesian demultiplexing of pooled single-cell RNA-seq data without genotype reference. Genome Biology 20, 273 (2019). <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1865-2>


## License

Apache-2.0
