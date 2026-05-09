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
    kwargs: Option<&std::collections::BTreeMap<String, String>>,
) -> Option<()> {
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
    kwargs: Option<&std::collections::BTreeMap<String, String>>,
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
    kwargs: Option<&std::collections::BTreeMap<String, String>>,
) -> Option<()> {
    Some(())
}
