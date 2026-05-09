# vireo-rs

Rust translation scaffold for `vireoSNP`, aiming for faithful, identical-output behavior against the original Python implementation.

## Upstream Source

The original code is checked into this repository under `vireo/`.

- Upstream repository: <https://github.com/huangyh09/vireo>
- Translated commit: `e3654633f7663732572c03c5dcf9fb00ec43b653`
- Original package name: `vireoSNP`
- Original license: Apache-2.0, copied to [`LICENSE`](LICENSE)

## Citation

If this translation or the original method is used in scientific work, cite:

Yuanhua Huang, Davis J. McCarthy, and Oliver Stegle. Vireo: Bayesian demultiplexing of pooled single-cell RNA-seq data without genotype reference. Genome Biology 20, 273 (2019). <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1865-2>

The upstream README also links a Zenodo DOI badge for the project: <https://zenodo.org/badge/latestdoi/187803798>

## Translation Plan

This crate starts as a scaffold: each original Python function has a corresponding Rust stub, and each original Python class has a Rust struct. The stubs use `PyValue` placeholders until each function is translated bottom-up.

Translation should proceed from leaf functions toward callers. Avoid introducing helper functions; if a helper seems necessary, treat that as a sign that the original function boundary or logic has not been translated faithfully enough.

## Audit Tooling

`ccc_mapping.toml` maps Rust functions to original Python functions, including path and class disambiguation. Use `code-complexity-comparator` from `/data/henriksson/github/claude/code-complexity-comparator`:

```sh
/data/henriksson/github/claude/code-complexity-comparator/target/release/ccc-rs analyze src -l rust --recurse -o rust.json
/data/henriksson/github/claude/code-complexity-comparator/target/release/ccc-rs analyze vireo -l python --recurse -o python.json
/data/henriksson/github/claude/code-complexity-comparator/target/release/ccc-rs order python.json -o translation_order.csv
/data/henriksson/github/claude/code-complexity-comparator/target/release/ccc-rs missing rust.json python.json --mapping ccc_mapping.toml
```

`gdb-translation-verifier-rs` at `/home/mahogny/github/claude/gdb-translation-verifier-rs` was reviewed. It is primarily useful for compiled C/C++ reference binaries, so it is not directly applicable to this Python-to-Rust port until a comparable executable-level harness exists.

`tracehash-rs` is noted for future deterministic trace hashing once concrete Rust logic and Python comparison harnesses are added.

## Test Data

Prioritize the included real example data under `vireo/data/` and workflows under `vireo/examples/` when building regression tests. Synthetic or tiny unit data should only supplement real-world fixtures.
