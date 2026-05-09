use ndarray::Array2;
use std::collections::BTreeSet;

pub fn get_confusion(
    ids1: &[String],
    ids2: &[String],
) -> Option<(Array2<f64>, Vec<String>, Vec<String>)> {
    if ids1.len() != ids2.len() {
        return None;
    }
    let ids1_uniq: Vec<String> = ids1
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let ids2_uniq: Vec<String> = ids2
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut confuse_mat = Array2::<f64>::zeros((ids1_uniq.len(), ids2_uniq.len()));
    for (i, id1) in ids1_uniq.iter().enumerate() {
        for (j, id2) in ids2_uniq.iter().enumerate() {
            confuse_mat[[i, j]] = ids1
                .iter()
                .zip(ids2.iter())
                .filter(|(a, b)| *a == id1 && *b == id2)
                .count() as f64;
        }
    }
    Some((confuse_mat, ids1_uniq, ids2_uniq))
}
