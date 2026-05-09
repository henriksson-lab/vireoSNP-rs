#![allow(dead_code, non_snake_case, unused_variables)]
#![allow(clippy::too_many_arguments)]

use ndarray::ArrayD;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, PartialEq)]
pub enum PyValue {
    #[default]
    None,
    Bool(bool),
    I64(i64),
    F64(f64),
    String(String),
    StringVec(Vec<String>),
    StringMatrix(Vec<Vec<String>>),
    I64Vec(Vec<i64>),
    UsizeVec(Vec<usize>),
    F64Vec(Vec<f64>),
    ArrayF64(ArrayD<f64>),
    Map(BTreeMap<String, PyValue>),
    Tuple(Vec<PyValue>),
}

pub mod setup;
pub mod simulate;
pub mod vireo_snp;
