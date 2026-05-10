use ndarray::{Array2, Array3};

pub fn heat_matrix(
    x: &Array2<f64>,
    yticks: Option<&[String]>,
    xticks: Option<&[String]>,
    rotation: f64,
    cmap: &str,
    alpha: f64,
    display_value: bool,
    row_sort: bool,
    aspect: &str,
    interpolation: &str,
) -> Option<()> {
    if x.is_empty() {
        return None;
    }
    if let Some(yticks) = yticks {
        if yticks.len() != x.nrows() {
            return None;
        }
    }
    if let Some(xticks) = xticks {
        if xticks.len() != x.ncols() {
            return None;
        }
    }
    let mut mat = x.clone();
    if row_sort {
        let mut row_idx: Vec<usize> = (0..mat.nrows()).collect();
        row_idx.sort_by(|&a, &b| {
            let va: f64 = mat
                .row(a)
                .iter()
                .enumerate()
                .map(|(j, v)| *v * 2f64.powi(j as i32))
                .sum();
            let vb: f64 = mat
                .row(b)
                .iter()
                .enumerate()
                .map(|(j, v)| *v * 2f64.powi(j as i32))
                .sum();
            va.total_cmp(&vb)
        });
        let old = mat.clone();
        for (new_i, old_i) in row_idx.into_iter().enumerate() {
            mat.row_mut(new_i).assign(&old.row(old_i));
        }
    }
    let _ = (
        mat,
        rotation,
        cmap,
        alpha,
        display_value,
        aspect,
        interpolation,
    );
    Some(())
}

pub fn plot_GT(
    out_dir: &str,
    cell_gpb: &Array3<f64>,
    donor_names: &[String],
    donor_gpb: Option<&Array3<f64>>,
    donor_names_in: Option<&[String]>,
) -> Option<()> {
    Some(())
}

pub fn minicode_plot(
    barcode_set: &[String],
    var_ids: Option<&[String]>,
    sample_ids: Option<&[String]>,
    cmap: &str,
    interpolation: &str,
) -> Option<Array2<f64>> {
    if barcode_set.is_empty() {
        return None;
    }
    let n_row = barcode_set[0].len().saturating_sub(1);
    let mut mat = Array2::<f64>::zeros((n_row, barcode_set.len()));
    for i in 0..n_row {
        for j in 0..barcode_set.len() {
            mat[[i, j]] = barcode_set[j]
                .chars()
                .nth(i + 1)
                .and_then(|c| c.to_digit(10))
                .unwrap_or(0) as f64;
        }
    }
    Some(mat)
}

pub fn anno_heat(
    x: &Array2<f64>,
    row_anno: Option<&[String]>,
    col_anno: Option<&[String]>,
    row_order_ids: Option<&[String]>,
    col_order_ids: Option<&[String]>,
    xticklabels: bool,
    yticklabels: bool,
    row_cluster: bool,
    col_cluster: bool,
) -> Option<()> {
    Some(())
}
