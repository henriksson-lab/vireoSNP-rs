use ndarray::{Array2, Array3};
#[cfg(feature = "plotting")]
use plotters::coord::Shift;
#[cfg(feature = "plotting-pdf")]
use plotters::drawing::IntoDrawingArea;
#[cfg(feature = "plotting")]
use plotters::prelude::*;
#[cfg(feature = "plotting-pdf")]
use plotters_cairo::CairoBackend;
use std::collections::BTreeSet;
#[cfg(feature = "plotting")]
use std::fs;
use std::path::Path;

#[cfg(feature = "plotting")]
fn matrix_range(x: &Array2<f64>) -> Option<(f64, f64)> {
    let mut iter = x.iter().copied().filter(|v| v.is_finite());
    let first = iter.next()?;
    let mut min_v = first;
    let mut max_v = first;
    for value in iter {
        min_v = min_v.min(value);
        max_v = max_v.max(value);
    }
    Some((min_v, max_v))
}

#[cfg(feature = "plotting")]
fn heat_color(value: f64, min_v: f64, max_v: f64, cmap: &str, alpha: f64) -> RGBAColor {
    let denom = (max_v - min_v).abs();
    let t = if denom <= f64::EPSILON {
        0.5
    } else {
        ((value - min_v) / denom).clamp(0.0, 1.0)
    };
    let (r, g, b) = match cmap {
        "Set3" => {
            const COLORS: [(u8, u8, u8); 12] = [
                (141, 211, 199),
                (255, 255, 179),
                (190, 186, 218),
                (251, 128, 114),
                (128, 177, 211),
                (253, 180, 98),
                (179, 222, 105),
                (252, 205, 229),
                (217, 217, 217),
                (188, 128, 189),
                (204, 235, 197),
                (255, 237, 111),
            ];
            COLORS[value.round().max(0.0) as usize % COLORS.len()]
        }
        "Reds" => (255, (245.0 - 175.0 * t) as u8, (240.0 - 210.0 * t) as u8),
        _ => (
            (247.0 - 205.0 * t) as u8,
            (252.0 - 112.0 * t) as u8,
            (245.0 - 150.0 * t) as u8,
        ),
    };
    RGBAColor(r, g, b, alpha.clamp(0.0, 1.0))
}

#[cfg(feature = "plotting")]
fn draw_heat_matrix<B: DrawingBackend>(
    root: DrawingArea<B, Shift>,
    x: &Array2<f64>,
    yticks: Option<&[String]>,
    xticks: Option<&[String]>,
    title: Option<&str>,
    cmap: &str,
    alpha: f64,
    display_value: bool,
) -> std::result::Result<(), DrawingAreaErrorKind<B::ErrorType>> {
    root.fill(&WHITE)?;
    let (width, height) = root.dim_in_pixel();
    let left = 96i32;
    let top = if title.is_some() { 42i32 } else { 18i32 };
    let right = 24i32;
    let bottom = if xticks.is_some() { 86i32 } else { 24i32 };
    let plot_w = (width as i32 - left - right).max(x.ncols() as i32);
    let plot_h = (height as i32 - top - bottom).max(x.nrows() as i32);
    let cell_w = plot_w as f64 / x.ncols().max(1) as f64;
    let cell_h = plot_h as f64 / x.nrows().max(1) as f64;
    let (min_v, max_v) = matrix_range(x).unwrap_or((0.0, 1.0));

    if let Some(title) = title {
        root.draw(&Text::new(
            title.to_string(),
            (left, 22),
            ("sans-serif", 20).into_font(),
        ))?;
    }

    for row in 0..x.nrows() {
        for col in 0..x.ncols() {
            let x0 = left + (col as f64 * cell_w).round() as i32;
            let y0 = top + (row as f64 * cell_h).round() as i32;
            let x1 = left + ((col + 1) as f64 * cell_w).round() as i32;
            let y1 = top + ((row + 1) as f64 * cell_h).round() as i32;
            root.draw(&Rectangle::new(
                [(x0, y0), (x1, y1)],
                heat_color(x[[row, col]], min_v, max_v, cmap, alpha).filled(),
            ))?;
            root.draw(&Rectangle::new(
                [(x0, y0), (x1, y1)],
                ShapeStyle::from(&RGBColor(235, 235, 235)).stroke_width(1),
            ))?;
            if display_value && cell_w >= 24.0 && cell_h >= 18.0 {
                root.draw(&Text::new(
                    format!("{:.2}", x[[row, col]]),
                    ((x0 + x1) / 2 - 11, (y0 + y1) / 2 + 5),
                    ("sans-serif", 12).into_font().color(&BLACK),
                ))?;
            }
        }
    }

    if let Some(yticks) = yticks {
        for (row, label) in yticks.iter().enumerate() {
            let y = top + ((row as f64 + 0.5) * cell_h).round() as i32 + 5;
            root.draw(&Text::new(
                label.clone(),
                (8, y),
                ("sans-serif", 13).into_font(),
            ))?;
        }
    }
    if let Some(xticks) = xticks {
        for (col, label) in xticks.iter().enumerate() {
            let x_pos = left + ((col as f64 + 0.5) * cell_w).round() as i32 - 18;
            root.draw(&Text::new(
                label.clone(),
                (x_pos, top + plot_h + 24),
                ("sans-serif", 13).into_font(),
            ))?;
        }
    }
    root.present()
}

#[cfg(feature = "plotting")]
fn save_heat_matrix(
    path: &str,
    x: &Array2<f64>,
    yticks: Option<&[String]>,
    xticks: Option<&[String]>,
    title: Option<&str>,
    cmap: &str,
    alpha: f64,
    display_value: bool,
    size: (u32, u32),
) -> Option<()> {
    if x.is_empty() {
        return None;
    }
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() && fs::create_dir_all(parent).is_err() {
            return None;
        }
    }
    let extension = Path::new(path)
        .extension()
        .and_then(|x| x.to_str())
        .unwrap_or("svg")
        .to_ascii_lowercase();
    match extension.as_str() {
        "png" => {
            let root = BitMapBackend::new(path, size).into_drawing_area();
            draw_heat_matrix(root, x, yticks, xticks, title, cmap, alpha, display_value).ok()
        }
        "svg" => {
            let root = SVGBackend::new(path, size).into_drawing_area();
            draw_heat_matrix(root, x, yticks, xticks, title, cmap, alpha, display_value).ok()
        }
        "pdf" => save_heat_matrix_pdf(
            path,
            x,
            yticks,
            xticks,
            title,
            cmap,
            alpha,
            display_value,
            size,
        ),
        _ => None,
    }
}

#[cfg(feature = "plotting-pdf")]
fn save_heat_matrix_pdf(
    path: &str,
    x: &Array2<f64>,
    yticks: Option<&[String]>,
    xticks: Option<&[String]>,
    title: Option<&str>,
    cmap: &str,
    alpha: f64,
    display_value: bool,
    size: (u32, u32),
) -> Option<()> {
    let surface = cairo::PdfSurface::new(size.0 as f64, size.1 as f64, path).ok()?;
    let context = cairo::Context::new(&surface).ok()?;
    let root = CairoBackend::new(&context, size).ok()?.into_drawing_area();
    draw_heat_matrix(root, x, yticks, xticks, title, cmap, alpha, display_value).ok()?;
    surface.finish();
    Some(())
}

#[cfg(all(feature = "plotting", not(feature = "plotting-pdf")))]
fn save_heat_matrix_pdf(
    path: &str,
    x: &Array2<f64>,
    yticks: Option<&[String]>,
    xticks: Option<&[String]>,
    title: Option<&str>,
    cmap: &str,
    alpha: f64,
    display_value: bool,
    size: (u32, u32),
) -> Option<()> {
    let _ = (
        path,
        x,
        yticks,
        xticks,
        title,
        cmap,
        alpha,
        display_value,
        size,
    );
    None
}

#[cfg(not(feature = "plotting"))]
fn save_heat_matrix(
    path: &str,
    x: &Array2<f64>,
    yticks: Option<&[String]>,
    xticks: Option<&[String]>,
    title: Option<&str>,
    cmap: &str,
    alpha: f64,
    display_value: bool,
    size: (u32, u32),
) -> Option<()> {
    let _ = (
        path,
        x,
        yticks,
        xticks,
        title,
        cmap,
        alpha,
        display_value,
        size,
    );
    Some(())
}

#[cfg(feature = "plotting-pdf")]
fn gt_distance_format() -> &'static str {
    "pdf"
}

#[cfg(not(feature = "plotting-pdf"))]
fn gt_distance_format() -> &'static str {
    "svg"
}

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
    if cell_gpb.shape()[1] != donor_names.len() {
        return None;
    }
    let n_donor = cell_gpb.shape()[1];
    let mut diff_mat = Array2::<f64>::zeros((n_donor, n_donor));
    for i in 0..n_donor {
        for j in 0..n_donor {
            let mut total = 0.0;
            let mut count = 0.0;
            for v in 0..cell_gpb.shape()[0] {
                for g in 0..cell_gpb.shape()[2] {
                    total += (cell_gpb[[v, i, g]] - cell_gpb[[v, j, g]]).abs();
                    count += 1.0;
                }
            }
            diff_mat[[i, j]] = if count > 0.0 { total / count } else { 0.0 };
        }
    }
    save_heat_matrix(
        &format!(
            "{out_dir}/fig_GT_distance_estimated.{}",
            gt_distance_format()
        ),
        &diff_mat,
        Some(donor_names),
        Some(donor_names),
        Some(&format!("Geno Prob Delta: {} SNPs", cell_gpb.shape()[0])),
        "BuGn",
        0.8,
        true,
        (720, 560),
    )?;

    if let Some(donor_gpb) = donor_gpb {
        let donor_names_in = donor_names_in?;
        if donor_gpb.shape()[1] != donor_names_in.len()
            || donor_gpb.shape()[0] != cell_gpb.shape()[0]
            || donor_gpb.shape()[2] != cell_gpb.shape()[2]
        {
            return None;
        }
        let mut diff_mat = Array2::<f64>::zeros((n_donor, donor_gpb.shape()[1]));
        for i in 0..n_donor {
            for j in 0..donor_gpb.shape()[1] {
                let mut total = 0.0;
                let mut count = 0.0;
                for v in 0..cell_gpb.shape()[0] {
                    for g in 0..cell_gpb.shape()[2] {
                        total += (cell_gpb[[v, i, g]] - donor_gpb[[v, j, g]]).abs();
                        count += 1.0;
                    }
                }
                diff_mat[[i, j]] = if count > 0.0 { total / count } else { 0.0 };
            }
        }
        save_heat_matrix(
            &format!("{out_dir}/fig_GT_distance_input.{}", gt_distance_format()),
            &diff_mat,
            Some(donor_names),
            Some(donor_names_in),
            Some(&format!("Geno Prob Delta: {} SNPs", cell_gpb.shape()[0])),
            "BuGn",
            0.8,
            true,
            (720, 560),
        )?;
    }
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

pub fn save_minicode_plot(
    out_file: &str,
    barcode_set: &[String],
    var_ids: Option<&[String]>,
    sample_ids: Option<&[String]>,
    fig_size: (f64, f64),
    fig_format: &str,
) -> Option<()> {
    let mat = minicode_plot(barcode_set, var_ids, sample_ids, "Set3", "none")?;
    let path = {
        let base = Path::new(out_file);
        let stem = base.file_stem()?.to_string_lossy();
        let parent = base.parent().unwrap_or_else(|| Path::new(""));
        parent.join(format!("{stem}.{fig_format}"))
    };
    let ylabels = match var_ids {
        Some(v) => v.to_vec(),
        None => (0..mat.nrows()).map(|x| x.to_string()).collect(),
    };
    let xlabels = match sample_ids {
        Some(samples) => barcode_set
            .iter()
            .zip(samples.iter())
            .map(|(barcode, sample)| format!("{barcode}\n{sample}"))
            .collect::<Vec<_>>(),
        None => barcode_set
            .iter()
            .enumerate()
            .map(|(i, barcode)| format!("{barcode}\nS{i}"))
            .collect::<Vec<_>>(),
    };
    let width = (fig_size.0.max(1.0) * 120.0).round() as u32;
    let height = (fig_size.1.max(1.0) * 120.0).round() as u32;
    save_heat_matrix(
        &path.to_string_lossy(),
        &mat,
        Some(&ylabels),
        Some(&xlabels),
        None,
        "Set3",
        0.9,
        true,
        (width, height),
    )
}

fn ordered_unique(values: &[String]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for value in values {
        if seen.insert(value.clone()) {
            out.push(value.clone());
        }
    }
    out
}

fn annotation_order(
    annotation: &[String],
    order_ids: Option<&[String]>,
    span: usize,
) -> Option<Vec<usize>> {
    if annotation.len() != span {
        return None;
    }
    let order = match order_ids {
        Some(ids) => ids.to_vec(),
        None => ordered_unique(annotation),
    };
    let mut ranks = Vec::with_capacity(annotation.len());
    for value in annotation {
        ranks.push(order.iter().position(|x| x == value)?);
    }
    let mut idx: Vec<usize> = (0..annotation.len()).collect();
    idx.sort_by_key(|&i| ranks[i]);
    Some(idx)
}

fn anno_heat_ordered(
    x: &Array2<f64>,
    row_anno: Option<&[String]>,
    col_anno: Option<&[String]>,
    row_order_ids: Option<&[String]>,
    col_order_ids: Option<&[String]>,
) -> Option<(Array2<f64>, Vec<usize>, Vec<usize>)> {
    if x.is_empty() {
        return None;
    }
    let row_idx = match row_anno {
        Some(anno) => annotation_order(anno, row_order_ids, x.nrows())?,
        None => (0..x.nrows()).collect(),
    };
    let col_idx = match col_anno {
        Some(anno) => annotation_order(anno, col_order_ids, x.ncols())?,
        None => (0..x.ncols()).collect(),
    };
    let mut out = Array2::<f64>::zeros((row_idx.len(), col_idx.len()));
    for (new_r, &old_r) in row_idx.iter().enumerate() {
        for (new_c, &old_c) in col_idx.iter().enumerate() {
            out[[new_r, new_c]] = x[[old_r, old_c]];
        }
    }
    Some((out, row_idx, col_idx))
}

pub fn save_anno_heat(
    out_file: &str,
    x: &Array2<f64>,
    row_anno: Option<&[String]>,
    col_anno: Option<&[String]>,
    row_order_ids: Option<&[String]>,
    col_order_ids: Option<&[String]>,
    xticklabels: bool,
    yticklabels: bool,
    fig_size: (f64, f64),
    fig_format: &str,
) -> Option<()> {
    let (mat, row_idx, col_idx) =
        anno_heat_ordered(x, row_anno, col_anno, row_order_ids, col_order_ids)?;
    let path = {
        let base = Path::new(out_file);
        let stem = base.file_stem()?.to_string_lossy();
        let parent = base.parent().unwrap_or_else(|| Path::new(""));
        parent.join(format!("{stem}.{fig_format}"))
    };
    let row_labels = yticklabels.then(|| {
        row_idx
            .iter()
            .map(|&i| match row_anno {
                Some(anno) => anno[i].clone(),
                None => i.to_string(),
            })
            .collect::<Vec<_>>()
    });
    let col_labels = xticklabels.then(|| {
        col_idx
            .iter()
            .map(|&i| match col_anno {
                Some(anno) => anno[i].clone(),
                None => i.to_string(),
            })
            .collect::<Vec<_>>()
    });
    let width = (fig_size.0.max(1.0) * 120.0).round() as u32;
    let height = (fig_size.1.max(1.0) * 120.0).round() as u32;
    save_heat_matrix(
        &path.to_string_lossy(),
        &mat,
        row_labels.as_deref(),
        col_labels.as_deref(),
        None,
        "BuGn",
        0.85,
        false,
        (width, height),
    )
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
    let _ = (xticklabels, yticklabels, row_cluster, col_cluster);
    anno_heat_ordered(x, row_anno, col_anno, row_order_ids, col_order_ids)?;
    Some(())
}
