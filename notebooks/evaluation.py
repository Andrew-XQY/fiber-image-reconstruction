"""Resuable components, Run once for all evaluations"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from xflow.extensions.style.aps import APS_COLORS, set_aps_double_column

# Save every figure as transparent vector PDF by default.
# `.pdf` extension on savefig() triggers the vector backend automatically;
# this rcParam handles the transparent background once for all callers.
plt.rcParams["savefig.transparent"] = True

PREVIEW_DPI = 500  # high-res raster companion for publication-style previews


def save_fig(fig, path_no_ext, preview=False, **kwargs):
    """Save `fig` as a transparent PDF; if `preview`, also a high-res PNG.

    Pass paths *without* an extension. Extra kwargs (e.g. `bbox_inches="tight"`)
    are forwarded to both savefigs.
    """
    fig.savefig(f"{path_no_ext}.pdf", **kwargs)
    if preview:
        fig.savefig(f"{path_no_ext}.png", dpi=PREVIEW_DPI, **kwargs)

CROP_METRIC_COLS = (
    "crop_inside_sum",
    "crop_outside_sum",
    "crop_outside_inside_ratio",
)


def attach_crop_metrics_from_db(
    beam_df: pd.DataFrame,
    metrics_db_path: str | Path,
    table_name: str = "mmf_dataset_metadata",
    sample_col: str = "sample_name",
    image_path_col: str = "image_path",
    metric_cols: tuple[str, ...] = CROP_METRIC_COLS,
) -> pd.DataFrame:
    """Left-join crop metric columns onto a beam-parameter DataFrame.

    The metrics DB's `image_path` (e.g. 'dataset/3/1763715592801104000.png') is
    reduced to its filename ('1763715592801104000.png') to match
    `beam_df[sample_col]`, which stores filenames with the extension included.
    All beam_df rows are preserved.
    """
    if sample_col not in beam_df.columns:
        raise KeyError(f"Column {sample_col!r} not found in beam_df.")

    metrics_db_path = Path(metrics_db_path).expanduser()
    if not metrics_db_path.is_file():
        raise FileNotFoundError(f"Metrics DB not found: {metrics_db_path}")

    cols = ", ".join(f'"{c}"' for c in (image_path_col, *metric_cols))
    with sqlite3.connect(metrics_db_path) as conn:
        metrics_df = pd.read_sql_query(f'SELECT {cols} FROM "{table_name}"', conn)

    metrics_df[sample_col] = metrics_df[image_path_col].map(lambda p: Path(str(p)).name)
    metrics_df = (
        metrics_df.drop(columns=[image_path_col])
                  .drop_duplicates(subset=sample_col, keep="first")
    )
    return beam_df.merge(metrics_df, on=sample_col, how="left")


def filter_df_by_threshold(
    df: pd.DataFrame,
    target_column: str,
    threshold: float,
    keep: str = "below",
    reset_index: bool = True,
) -> pd.DataFrame:
    """Filter rows by one numeric threshold while preserving the table contract.

    `keep="below"` keeps rows where `target_column <= threshold`; `"above"` keeps
    rows where `target_column >= threshold`.
    """
    if target_column not in df.columns:
        raise KeyError(f"Column {target_column!r} not found. Available columns: {list(df.columns)}")

    keep = str(keep).strip().lower()
    if keep not in {"below", "above"}:
        raise ValueError("keep must be either 'below' or 'above'")

    values = pd.to_numeric(df[target_column], errors="coerce")
    mask = values <= float(threshold) if keep == "below" else values >= float(threshold)

    out = df.loc[mask.fillna(False)].copy()
    return out.reset_index(drop=True) if reset_index else out


def apply_optional_threshold_filter(
    df: pd.DataFrame,
    target_column: str | None = None,
    threshold: float | None = None,
    keep: str = "below",
) -> pd.DataFrame:
    """Apply `filter_df_by_threshold` only when both column and threshold are set.

    Leave `target_column=None` or `threshold=None` to keep the DataFrame unchanged.
    """
    if target_column is None or threshold is None:
        return df
    return filter_df_by_threshold(df, target_column=target_column, threshold=threshold, keep=keep)


# Global validation filter controls used by downstream cells below.
# Leave disabled by default so existing evaluation behavior is unchanged.


import numpy as np
import pandas as pd
from dirs import *

# =========================================================
# Generic NaN/inf row cleaner
# Drops any row containing NaN or ±inf in any column.
# Returns a new df; original is untouched.
# =========================================================


def _select_columns(df, check_contains=None):
    if check_contains is None:
        return df.columns

    if isinstance(check_contains, str):
        keywords = [check_contains]
    else:
        keywords = list(check_contains)

    return [
        c for c in df.columns
        if any(keyword in c for keyword in keywords)
    ]


def drop_invalid_rows(df, check_contains=None):
    columns = _select_columns(df, check_contains=check_contains)

    checked = df[columns].replace([np.inf, -np.inf], np.nan)
    mask = checked.notna().all(axis=1)

    return df.loc[mask].reset_index(drop=True)


def drop_zero_rows(df, check_contains=None):
    columns = _select_columns(df, check_contains=check_contains)

    checked = df[columns].apply(pd.to_numeric, errors="coerce")
    mask = ~checked.eq(0).any(axis=1)

    return df.loc[mask].reset_index(drop=True)


def load_beam_param_df(
    csv_path,
    invalid_check_contains=None,
    zero_check_contains=None,
):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = (
        df.columns.astype(str)
        .str.replace("﻿", "", regex=False)
        .str.strip()
    )

    df = drop_invalid_rows(df, check_contains=invalid_check_contains)
    df = drop_zero_rows(df, check_contains=zero_check_contains)

    return df


_DIMS_PARAMS = [
    ("h", "centroid"),
    ("v", "centroid"),
    ("h", "width"),
    ("v", "width"),
]



# =========================================================
# Shared imports + function definitions used by the three
# downstream evaluation cells:
#   1) Classic three statistical plots on testset
#   2) Error distribution hexbin plots
#   3) Statistical overview metrics
# Each downstream cell keeps its own bottom "run + save"
# section so it can still be executed individually.
# =========================================================


# ---------------------------------------------------------
# Shared plot styling constants
# ---------------------------------------------------------
TICK_LABEL_SIZE   = 11
AXIS_LABEL_SIZE   = 12
TITLE_FONT_SIZE   = 13
LEGEND_FONT_SIZE  = 10
CBAR_TICK_SIZE    = 10
CBAR_LABEL_SIZE   = 11
CBAR_NUM_TICKS    = 5
TICK_MAJOR_LEN    = 5
TICK_MINOR_LEN    = 3
TICK_WIDTH        = 1.0

SAVE_PREFIX = "_"


# #########################################################
#                                                         #
#   1) CLASSIC THREE STATISTICAL PLOTS ON TESTSET         #
#      scatter / residual scatter / residual histogram    #
#                                                         #
# #########################################################

PLOT_SPECS = (
    ("centroid", "gaussian"),
    ("width", "gaussian"),
    ("core_sigma", None),
)
DIMS_TO_USE = ("h", "v")

FIGSIZE = (3.8, 3.8)
SCATTER_POINT_SIZE = 5
REF_LINE_WIDTH = 2.4

PRED_GT_XLIM = (0.0, 1.0)
PRED_GT_YLIM = (0.0, 1.0)
RESIDUAL_SCATTER_XLIM = (0.0, 1.0)
RESIDUAL_SCATTER_YLIM = (-1.0, 1.0)
HIST_X_RANGE_PCT = (-50.0, 50.0)
HIST_BINS = 50
HIST_Y_MAX = None
HIST_X_COVERAGE_PCT = 90.0
HIST_Y_HEADROOM = 0.10

DIM_STYLE = {
    "h": {"name": "Horizontal", "color": APS_COLORS[4]},
    "v": {"name": "Vertical",   "color": APS_COLORS[5]},
}
REF_COLOR = APS_COLORS[7]


def _col_names(fit_method, param_type, dim):
    param_type = str(param_type).strip().lower()
    dim = str(dim).strip().lower()
    fit_method = "" if fit_method is None else str(fit_method).strip().lower()
    if fit_method:
        gt_col = f"label_{fit_method}_{dim}_{param_type}"
        pred_col = f"reconstructed_{fit_method}_{dim}_{param_type}"
    else:
        gt_col = f"label_{dim}_{param_type}"
        pred_col = f"reconstructed_{dim}_{param_type}"
    return gt_col, pred_col


def prepare_plot_df(df, param_type, fit_method, dims=("h", "v")):
    out = df.copy().replace([np.inf, -np.inf], np.nan)
    active_dims = []
    needed_cols = []
    for dim in dims:
        gt_col, pred_col = _col_names(fit_method, param_type, dim)
        if gt_col in out.columns and pred_col in out.columns:
            active_dims.append(dim)
            needed_cols.extend([gt_col, pred_col])
    if not active_dims:
        raise ValueError(
            f"No usable columns for fit_method={fit_method}, param_type={param_type}."
        )
    for c in needed_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    valid_mask = np.zeros(len(out), dtype=bool)
    for dim in active_dims:
        gt_col, pred_col = _col_names(fit_method, param_type, dim)
        valid_mask |= out[gt_col].notna() & out[pred_col].notna()
    out = out.loc[valid_mask].reset_index(drop=True)
    return out, active_dims


def _pair(df, fit_method, param_type, dim):
    gt_col, pred_col = _col_names(fit_method, param_type, dim)
    if gt_col not in df.columns or pred_col not in df.columns:
        return np.array([], dtype=float), np.array([], dtype=float)
    gt   = pd.to_numeric(df[gt_col],   errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(gt) & np.isfinite(pred)
    return gt[m], pred[m]


def _new_ax(scale=1.0):
    set_aps_double_column(figsize=FIGSIZE, scale=scale, legend_background=True)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_box_aspect(1)
    ax.tick_params(axis="both", which="major",
                   labelsize=TICK_LABEL_SIZE,
                   length=TICK_MAJOR_LEN, width=TICK_WIDTH)
    ax.tick_params(axis="both", which="minor",
                   length=TICK_MINOR_LEN, width=TICK_WIDTH)
    return fig, ax


def _legend(ax, loc="best", fontsize=LEGEND_FONT_SIZE, frameon=True,
            marker_scale=1.0, handletextpad=None):
    kwargs = {"loc": loc, "fontsize": fontsize, "markerscale": marker_scale}
    if handletextpad is not None:
        kwargs["handletextpad"] = handletextpad
    if frameon:
        ax.legend(**kwargs, frameon=True,
                  facecolor="white", edgecolor="black", framealpha=0.95)
    else:
        ax.legend(**kwargs, frameon=False)


def plot_pred_vs_label(df, param_type="centroid", fit_method="gaussian",
                       dims=("h", "v"), xlim=(0.0, 1.0), ylim=(0.0, 1.0),
                       scale=1.0,
                       legend_fontsize=LEGEND_FONT_SIZE,
                       legend_marker_scale=1.0,
                       legend_handletextpad=None):
    fig, ax = _new_ax(scale=scale)
    for dim in dims:
        gt, pred = _pair(df, fit_method, param_type, dim)
        if gt.size == 0:
            continue
        ax.scatter(gt, pred, s=SCATTER_POINT_SIZE, alpha=0.72, marker="o",
                   color=DIM_STYLE[dim]["color"], edgecolors="none",
                   label=f"{DIM_STYLE[dim]['name']} {param_type}")
    ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], "--",
            lw=REF_LINE_WIDTH, color=REF_COLOR)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    xlabel_map = {"centroid": "Actual beam centroids", "width": "Actual beam widths"}
    ax.set_xlabel(xlabel_map.get(param_type, f"Actual {param_type}"),
                  fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Predicted values", fontsize=AXIS_LABEL_SIZE)
    _legend(ax, fontsize=legend_fontsize,
            marker_scale=legend_marker_scale,
            handletextpad=legend_handletextpad)
    return fig, ax


def plot_residual_vs_label(df, param_type="centroid", fit_method="gaussian",
                           dims=("h", "v"), xlim=(0.0, 1.0), ylim=(-1.0, 1.0),
                           scale=1.0):
    fig, ax = _new_ax(scale=scale)
    for dim in dims:
        gt, pred = _pair(df, fit_method, param_type, dim)
        if gt.size == 0:
            continue
        residual = np.clip(pred - gt, ylim[0], ylim[1])
        ax.scatter(gt, residual, s=SCATTER_POINT_SIZE, alpha=0.72, marker="o",
                   color=DIM_STYLE[dim]["color"], edgecolors="none",
                   label=f"{DIM_STYLE[dim]['name']} {param_type}")
    ax.axhline(0.0, ls="--", lw=REF_LINE_WIDTH, color=REF_COLOR)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("Ground truth",          fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Residual (pred - gt)",  fontsize=AXIS_LABEL_SIZE)
    _legend(ax)
    return fig, ax


def plot_residual_hist_pct(df, param_type="centroid", fit_method="gaussian",
                           dims=("h", "v"), x_range_pct=HIST_X_RANGE_PCT,
                           bins=HIST_BINS, y_max=HIST_Y_MAX,
                           x_coverage_pct=HIST_X_COVERAGE_PCT,
                           y_headroom=HIST_Y_HEADROOM,
                           annotate_stats=False,
                           annotate_loc="upper left",
                           legend_fontsize=LEGEND_FONT_SIZE,
                           legend_frameon=True,
                           scale=1.0,
                           y_num_ticks=None):
    # Pre-pass: gather residuals per dim so we can size axes adaptively.
    per_dim = []
    pooled = []
    for dim in dims:
        gt, pred = _pair(df, fit_method, param_type, dim)
        if gt.size == 0:
            continue
        r = (pred - gt) * 100.0
        r = r[np.isfinite(r)]
        if r.size == 0:
            continue
        per_dim.append((dim, r))
        pooled.append(r)

    # Display window (set_xlim only): adaptive symmetric covering x_coverage_pct of |residual|, else fixed x_range_pct.
    if x_coverage_pct is not None and pooled:
        half = float(np.percentile(np.abs(np.concatenate(pooled)), float(x_coverage_pct)))
        if half <= 0:
            half = 1.0
        x_min, x_max = -half, half
    else:
        x_min, x_max = float(x_range_pct[0]), float(x_range_pct[1])

    # Bin range: full data extent (symmetric around 0). Independent of the display window
    # so changing x_range_pct only zooms — bin width and bar heights stay invariant.
    if pooled:
        data_extent = float(np.max(np.abs(np.concatenate(pooled))))
        if data_extent <= 0:
            data_extent = max(abs(x_min), abs(x_max), 1.0)
        bin_min, bin_max = -data_extent, data_extent
    else:
        bin_min, bin_max = x_min, x_max

    fig, ax = _new_ax(scale=scale)
    hist_bins = np.linspace(bin_min, bin_max, int(bins) + 1)
    bin_centers = 0.5 * (hist_bins[:-1] + hist_bins[1:])
    visible_mask = (bin_centers >= x_min) & (bin_centers <= x_max)
    max_count = 0.0
    for dim, r in per_dim:
        counts, edges, _ = ax.hist(r, bins=hist_bins, alpha=0.38,
                                   color=DIM_STYLE[dim]["color"],
                                   label=f"{DIM_STYLE[dim]['name']} {param_type}")
        if counts.size and visible_mask.any():
            max_count = max(max_count, float(np.max(counts[visible_mask])))
        # Truncated-sample Gaussian fit: estimate μ and σ from residuals inside
        # the visible window [x_min, x_max] only, so extreme outliers don't
        # inflate σ relative to the bars the viewer sees. Scaled to the in-window
        # count so curve height matches the visible histogram. Note: truncation
        # biases σ slightly low; the effect is small when the window contains
        # the bulk of the distribution.
        fit_mask = (r >= x_min) & (r <= x_max)
        r_fit = r[fit_mask]
        if r_fit.size > 1:
            mu = float(np.mean(r_fit))
            sigma = float(np.std(r_fit, ddof=1))
            if sigma > 0:
                xfit = np.linspace(edges[0], edges[-1], 300)
                bw = edges[1] - edges[0]
                yfit = (r_fit.size * bw
                        * (1.0 / (sigma * np.sqrt(2.0 * np.pi)))
                        * np.exp(-0.5 * ((xfit - mu) / sigma) ** 2))
                ax.plot(xfit, yfit, lw=2.0, color=DIM_STYLE[dim]["color"], alpha=0.95)
    ax.axvline(0.0, ls="--", lw=REF_LINE_WIDTH, color=REF_COLOR)
    ax.set_xlim(x_min, x_max)

    # Y limit: explicit y_max wins; else adaptive from tallest bin + headroom; else autoscale.
    if y_max is not None:
        ax.set_ylim(0.0, float(y_max))
    elif y_headroom is not None and max_count > 0:
        ax.set_ylim(0.0, max_count * (1.0 + float(y_headroom)))

    ax.set_xlabel(f"{str(param_type).capitalize()} error (% of frame)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Count",        fontsize=AXIS_LABEL_SIZE)
    if y_num_ticks is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=int(y_num_ticks)))

    if annotate_stats and pooled:
        legend_loc = "upper right" if annotate_loc == "upper left" else "upper left"
    else:
        legend_loc = "best"
    _legend(ax, loc=legend_loc, fontsize=legend_fontsize, frameon=legend_frameon)

    if annotate_stats and pooled:
        abs_r = np.abs(np.concatenate(pooled))
        n = int(len(df))
        median_pct = float(np.median(abs_r))
        q1, q3 = np.percentile(abs_r, [25, 75])
        iqr_pct = float(q3 - q1)
        txt = f"n = {n}\nmedian = {median_pct:.2f}%\nIQR = {iqr_pct:.2f}%"
        if annotate_loc == "upper left":
            tx, ty, ha = 0.04, 0.96, "left"
        else:
            tx, ty, ha = 0.96, 0.96, "right"
        ax.text(tx, ty, txt, transform=ax.transAxes, ha=ha, va="top",
                multialignment="left",
                fontsize=LEGEND_FONT_SIZE)
    return fig, ax


def make_three_plots(df, param_type="centroid", fit_method="gaussian",
                     dims=("h", "v"),
                     pred_gt_xlim=(0.0, 1.0), pred_gt_ylim=(0.0, 1.0),
                     residual_xlim=(0.0, 1.0), residual_ylim=(-1.0, 1.0),
                     hist_x_range_pct=HIST_X_RANGE_PCT,
                     hist_bins=HIST_BINS, hist_y_max=HIST_Y_MAX,
                     hist_x_coverage_pct=HIST_X_COVERAGE_PCT,
                     hist_y_headroom=HIST_Y_HEADROOM,
                     hist_annotate_stats=False,
                     hist_annotate_loc="upper left",
                     scale=1.0,
                     scatter_legend_fontsize=LEGEND_FONT_SIZE,
                     scatter_legend_marker_scale=1.0,
                     scatter_legend_handletextpad=None,
                     hist_y_num_ticks=None):
    clean_df, active_dims = prepare_plot_df(
        df=df, param_type=param_type, fit_method=fit_method, dims=dims,
    )
    f1, _ = plot_pred_vs_label(clean_df, param_type=param_type, fit_method=fit_method,
                               dims=active_dims, xlim=pred_gt_xlim, ylim=pred_gt_ylim,
                               scale=scale,
                               legend_fontsize=scatter_legend_fontsize,
                               legend_marker_scale=scatter_legend_marker_scale,
                               legend_handletextpad=scatter_legend_handletextpad)
    f2, _ = plot_residual_vs_label(clean_df, param_type=param_type, fit_method=fit_method,
                                   dims=active_dims, xlim=residual_xlim, ylim=residual_ylim,
                                   scale=scale)
    f3, _ = plot_residual_hist_pct(clean_df, param_type=param_type, fit_method=fit_method,
                                   dims=active_dims, x_range_pct=hist_x_range_pct,
                                   bins=hist_bins, y_max=hist_y_max,
                                   x_coverage_pct=hist_x_coverage_pct,
                                   y_headroom=hist_y_headroom,
                                   annotate_stats=hist_annotate_stats,
                                   annotate_loc=hist_annotate_loc,
                                   scale=scale,
                                   y_num_ticks=hist_y_num_ticks)
    return f1, f2, f3


# #########################################################
#                                                         #
#   2) ERROR DISTRIBUTION HEXBIN PLOTS                    #
#      mean |error| over (label_h, label_v)               #
#                                                         #
# #########################################################

def add_gaussian_error_columns(df):
    out = df.copy()
    for dim, param in _DIMS_PARAMS:
        label_col = f"label_gaussian_{dim}_{param}"
        pred_col  = f"reconstructed_gaussian_{dim}_{param}"
        err_col   = f"error_gaussian_{dim}_{param}"
        out[err_col] = (
            pd.to_numeric(out[label_col], errors="coerce")
            - pd.to_numeric(out[pred_col], errors="coerce")
        )
    return out


def plot_hex_error_2d(df, param_type="centroid", gridsize=25,
                      xlim=(0.0, 1.0), ylim=(0.0, 1.0),
                      range_mode="full", cmap="viridis",
                      vmax=None, figsize=(3.8, 3.8), scale=1.0):
    x   = df[f"label_gaussian_h_{param_type}"].to_numpy()
    y   = df[f"label_gaussian_v_{param_type}"].to_numpy()
    e_h = df[f"error_gaussian_h_{param_type}"].to_numpy()
    e_v = df[f"error_gaussian_v_{param_type}"].to_numpy()
    mae = 0.5 * (np.abs(e_h) + np.abs(e_v))

    if range_mode == "tight":
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.02 * (hi - lo) if hi > lo else 0.01
        xlim = (lo - pad, hi + pad)
        ylim = xlim

    set_aps_double_column(figsize=figsize, scale=scale, legend_background=True)
    fig, ax = plt.subplots()
    ax.set_box_aspect(1)
    hb = ax.hexbin(x, y, C=mae, reduce_C_function=np.mean, gridsize=gridsize,
                   cmap=cmap, extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                   vmin=0.0, vmax=vmax, mincnt=1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim[1], ylim[0])  # invert y so origin is top-left
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Horizontal {param_type} (label)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(f"Vertical {param_type} (label)",   fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", which="major",
                   labelsize=TICK_LABEL_SIZE,
                   length=TICK_MAJOR_LEN, width=TICK_WIDTH)
    ax.tick_params(axis="both", which="minor",
                   length=TICK_MINOR_LEN, width=TICK_WIDTH)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label("Mean |error|", fontsize=CBAR_LABEL_SIZE)
    cb.locator = MaxNLocator(nbins=CBAR_NUM_TICKS)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=CBAR_TICK_SIZE,
                      length=TICK_MAJOR_LEN, width=TICK_WIDTH)
    return fig, ax


# #########################################################
#                                                         #
#   3) STATISTICAL OVERVIEW METRICS                       #
#      per-param metrics + image metrics → JSON           #
#                                                         #
# #########################################################

def _r2_safe(y, yh):
    denom = np.sum((y - np.mean(y)) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y - yh) ** 2) / denom)


_METRICS = {
    "mse":       lambda y, yh: float(np.mean((y - yh) ** 2)),
    "rmse":      lambda y, yh: float(np.sqrt(np.mean((y - yh) ** 2))),
    "mae":       lambda y, yh: float(np.mean(np.abs(y - yh))),
    "medae":     lambda y, yh: float(np.median(np.abs(y - yh))),
    "max_error": lambda y, yh: float(np.max(np.abs(y - yh))),
    "r2":        _r2_safe,
}


def compute_fit_metrics(df, fit_methods=("gaussian", "moments"),
                        metrics=("rmse", "mse"), save_path=None):
    unknown = [m for m in metrics if m not in _METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Available: {sorted(_METRICS)}")
    results = {}
    for fit_method in fit_methods:
        for dim, param in _DIMS_PARAMS:
            y_col  = f"label_{fit_method}_{dim}_{param}"
            yh_col = f"reconstructed_{fit_method}_{dim}_{param}"
            if y_col not in df.columns or yh_col not in df.columns:
                raise KeyError(f"Missing required columns: {y_col} or {yh_col}")
            y  = pd.to_numeric(df[y_col],  errors="coerce").to_numpy(dtype=float)
            yh = pd.to_numeric(df[yh_col], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(y) & np.isfinite(yh)
            y, yh = y[m], yh[m]
            for metric_name in metrics:
                key = f"{fit_method}_{dim}_{param}_{metric_name}"
                results[key] = float("nan") if y.size == 0 else _METRICS[metric_name](y, yh)
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
    return results


def append_image_metrics(results, df, image_cols=None, agg="mean"):
    if image_cols is None:
        image_cols = {
            "reconstructed_label_psnr": "image_psnr",
            "reconstructed_label_ssim": "image_ssim",
            "reconstructed_label_mae":  "image_pixelmae",
        }
    if agg not in ("mean", "median"):
        raise ValueError("agg must be 'mean' or 'median'")
    reducer = np.mean if agg == "mean" else np.median
    for col, out_key in image_cols.items():
        if col not in df.columns:
            results[out_key] = float("nan")
            continue
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        results[out_key] = float(reducer(x)) if x.size > 0 else float("nan")
    return results


def compute_fit_and_image_metrics(df, fit_methods=("gaussian", "moments"),
                                  metrics=("rmse", "mae", "r2", "mse"),
                                  image_cols=None, image_agg="mean",
                                  save_path=None):
    results = compute_fit_metrics(df=df, fit_methods=fit_methods,
                                  metrics=metrics, save_path=None)
    results = append_image_metrics(results=results, df=df,
                                   image_cols=image_cols, agg=image_agg)
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
    return results
