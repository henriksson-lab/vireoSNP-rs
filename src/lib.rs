#![allow(dead_code, non_snake_case, unused_variables)]
#![allow(clippy::too_many_arguments)]

pub mod setup;
pub mod simulate;
pub mod vireo_snp;

pub use crate::vireo_snp::vireo::{FitBuilder, FitResult, Result, VireoError};

pub fn fit(cell_data: impl Into<String>) -> FitBuilder {
    vireo_snp::vireo::fit(cell_data)
}
