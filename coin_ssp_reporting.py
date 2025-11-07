"""
Visualization and Reporting Module for COIN_SSP

This module contains all visualization and reporting functions including:
- Forward model visualization functions
- Target GDP visualization
- Scaling factors visualization
- Objective function visualization
- TFP baseline visualization
- GDP weighted scaling summaries

Extracted from coin_ssp_utils.py and main.py for better organization.
"""

import copy
import json
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm, LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import xarray as xr
from datetime import datetime
from typing import Dict, Any, List
from coin_ssp_models import ScalingParams

# Import functions from other coin_ssp modules
from coin_ssp_netcdf import (
    create_serializable_config, extract_year_coordinate
)
from coin_ssp_math_utils import (
    apply_loess_subtract, calculate_zero_biased_range, calculate_zero_biased_axis_range,
    calculate_area_weights, calculate_time_means, calculate_global_mean, calculate_gdp_weighted_mean
)
from coin_ssp_utils import get_ssp_data, get_grid_metadata


def format_log_colorbar_ticks(colorbar, base=10):
    """
    Format colorbar tick labels to show linear values instead of log values.

    For a log-scale colorbar where ticks show log values (e.g., -0.05),
    this converts them to show the corresponding linear values (e.g., 10^-0.05 = 0.891).

    Parameters
    ----------
    colorbar : matplotlib.colorbar.Colorbar
        The colorbar to format
    base : float, default=10
        The logarithm base (10 for log10, np.e for natural log)

    Examples
    --------
    >>> cbar = plt.colorbar(im, ax=ax)
    >>> format_log_colorbar_ticks(cbar, base=10)  # For log10 scale
    >>> format_log_colorbar_ticks(cbar, base=np.e)  # For natural log scale
    """
    def log_tick_formatter(x, pos):
        """Format tick label as linear value from log value."""
        linear_value = base ** x
        # Use appropriate formatting based on magnitude
        if abs(linear_value) >= 1000 or abs(linear_value) < 0.01:
            return f'{linear_value:.2e}'
        elif abs(linear_value) >= 10:
            return f'{linear_value:.1f}'
        else:
            return f'{linear_value:.3f}'

    colorbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))


def get_adaptive_subplot_layout(n_targets):
    """
    Calculate optimal subplot layout based on number of targets.

    For 3 or fewer targets: single column layout
    For 4+ targets: two-column layout to maintain reasonable aspect ratios

    Parameters
    ----------
    n_targets : int
        Number of targets/plots per page

    Returns
    -------
    tuple
        (rows, cols, figsize) for matplotlib subplot layout
    """
    if n_targets <= 3:
        # Single column layout for 3 or fewer
        return (n_targets, 1, (12, 16))
    else:
        # Two column layout for 4+
        rows = (n_targets + 1) // 2  # Ceiling division
        cols = 2
        height = 4 * rows + 4  # Scale height with number of rows
        return (rows, cols, (16, height))


def add_extremes_info_box(ax, data_min, data_max, gdp_weighted_mean=None, lat=None, lon=None, data_array=None, valid_mask=None):
    """
    Add a text box showing min/max values to a map visualization.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the info box to
    data_min : float
        Minimum value to display
    data_max : float
        Maximum value to display
    gdp_weighted_mean : float, optional
        GDP-weighted mean value to display
    lat : array, optional
        Latitude coordinates for location display
    lon : array, optional
        Longitude coordinates for location display
    data_array : array, optional
        2D data array [lat, lon] to find max/min locations
    valid_mask : array, optional
        Boolean mask [lat, lon] for valid cells
    """
    # Build text with coordinates if available
    if data_array is not None and lat is not None and lon is not None:
        # Apply valid mask if provided
        if valid_mask is not None:
            masked_data = np.where(valid_mask, data_array, np.nan)
        else:
            masked_data = data_array

        # Find max location
        max_idx = np.unravel_index(np.nanargmax(masked_data), masked_data.shape)
        max_lat, max_lon = lat[max_idx[0]], lon[max_idx[1]]

        # Find min location
        min_idx = np.unravel_index(np.nanargmin(masked_data), masked_data.shape)
        min_lat, min_lon = lat[min_idx[0]], lon[min_idx[1]]

        max_text = f'Max: {data_max:.6f} ({max_lat:.1f}, {max_lon:.1f})'
        min_text = f'Min: {data_min:.6f} ({min_lat:.1f}, {min_lon:.1f})'
    else:
        max_text = f'Max: {data_max:.6f}'
        min_text = f'Min: {data_min:.6f}'

    if gdp_weighted_mean is not None:
        max_min_text = f'{max_text}\n{min_text}\nGDP-wtd mean: {gdp_weighted_mean:.6f}'
    else:
        max_min_text = f'{max_text}\n{min_text}'

    ax.text(0.02, 0.08, max_min_text, transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
           fontsize=10, verticalalignment='bottom')


def create_forward_model_visualization(forward_results, config, output_dir, all_data):
    """
    Create comprehensive PDF visualization for Step 4 forward model results.

    Generates a multi-page PDF with one page per (target, response_function, SSP) combination.
    Each page shows global mean time series with three lines:
    - y_climate: GDP with full climate change effects
    - y_weather: GDP with weather variability only
    - baseline: Original SSP GDP projections

    Parameters
    ----------
    forward_results : dict
        Results from Step 4 forward integration containing SSP-specific data
    config : dict
        Configuration dictionary with scenarios and response functions
    output_dir : str
        Directory for output files
    model_name : str
        Climate model name for labeling
    all_data : dict
        All loaded NetCDF data for baseline GDP access

    Returns
    -------
    str
        Path to generated PDF file
    """

    # Extract configuration values for standardized naming
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']

    # Generate output filename using standardized pattern
    pdf_filename = f"step4_{json_id}_{model_name}_forward_model_lineplots.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Extract metadata
    valid_mask = all_data['_metadata']['valid_mask']
    lat = forward_results['_coordinates']['lat']
    response_function_names = forward_results['response_function_names']
    target_names = forward_results['target_names']
    forward_ssps = config['ssp_scenarios']['forward_simulation_ssps']

    # Calculate adaptive layout based on number of targets
    n_targets = len(target_names)
    subplot_rows, subplot_cols, fig_size = get_adaptive_subplot_layout(n_targets)

    # Calculate total pages (one page per response function × SSP combination)
    total_pages = len(response_function_names) * len(forward_ssps)

    print(f"Creating Step 4 line charts: {n_targets} targets per page across {total_pages} pages")
    print(f"  {len(response_function_names)} response functions × {len(forward_ssps)} SSPs")
    print(f"  Layout: {subplot_rows} rows × {subplot_cols} cols per page")

    # Create PDF with multi-page layout
    with PdfPages(pdf_path) as pdf:
        page_num = 0

        # Loop through response functions and SSPs (each gets its own page)
        for response_idx, response_name in enumerate(response_function_names):
            response_config = config['response_function_scalings'][response_idx]

            for ssp in forward_ssps:

                # Create new page for this response function × SSP combination
                page_num += 1
                fig = plt.figure(figsize=fig_size)
                fig.suptitle(f'Step 4: Forward Model Time Series - {model_name}\n'
                            f'SSP: {ssp.upper()} | Response Function: {response_config["scaling_name"]} - Page {page_num}/{total_pages}',
                            fontsize=16, fontweight='bold', y=0.98)

                # Plot all targets on this page
                for target_idx, target_name in enumerate(target_names):
                    target_config = config['gdp_targets'][target_idx]

                    # Calculate subplot position (1-indexed)
                    if subplot_cols == 1:
                        # Single column layout
                        subplot_idx = target_idx + 1
                        ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)
                    else:
                        # Two column layout
                        row = target_idx // subplot_cols
                        col = target_idx % subplot_cols
                        subplot_idx = row * subplot_cols + col + 1
                        ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)

                    # Get SSP-specific data
                    ssp_results = forward_results['forward_results'][ssp]
                    gdp_climate = ssp_results['gdp_climate']  # [response_func, target, time, lat, lon]
                    gdp_weather = ssp_results['gdp_weather']  # [response_func, target, time, lat, lon]

                    # Get baseline GDP data from all_data
                    baseline_gdp = get_ssp_data(all_data, ssp, 'gdp')  # [time, lat, lon]

                    # Get years array from pre-computed metadata
                    years = all_data['years']


                    # Extract time series for this combination
                    ntime = gdp_climate.shape[2]
                    y_climate_series = np.zeros(ntime)
                    y_weather_series = np.zeros(ntime)
                    baseline_series = np.zeros(ntime)

                    for t in range(ntime):
                        # Extract spatial slice for this time
                        gdp_climate_t = gdp_climate[response_idx, target_idx, t, :, :]
                        gdp_weather_t = gdp_weather[response_idx, target_idx, t, :, :]
                        baseline_t = baseline_gdp[t, :, :]  # Note: baseline is [time, lat, lon]

                        # Calculate global means
                        y_climate_series[t] = calculate_global_mean(gdp_climate_t, valid_mask)
                        y_weather_series[t] = calculate_global_mean(gdp_weather_t, valid_mask)
                        baseline_series[t] = calculate_global_mean(baseline_t, valid_mask)

                    # Plot the time series
                    ax.plot(years, y_climate_series, 'r-', linewidth=2, label='GDP with Climate Effects')
                    ax.plot(years, y_weather_series, 'b--', linewidth=2, label='GDP with Weather Only')
                    ax.plot(years, baseline_series, 'k:', linewidth=2, label=f'Baseline {ssp.upper()} GDP')

                    # Formatting
                    ax.set_xlabel('Year', fontsize=12)
                    ax.set_ylabel('Global Mean GDP', fontsize=12)
                    target_type_label = target_config.get('target_type', 'unknown').upper()
                    ax.set_title(f'{target_config["target_name"]} × {response_config["scaling_name"]} × {ssp.upper()}\n'
                                f'[{target_type_label}] ({target_config.get("description", "")[:50]}...)',
                                fontsize=14, fontweight='bold')
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)

                    # Set reasonable y-axis limits using zero-biased range
                    all_values = np.concatenate([y_climate_series, y_weather_series, baseline_series])
                    valid_values = all_values[np.isfinite(all_values)]
                    if len(valid_values) > 0:
                        # Use 1-99 percentiles to exclude extreme outliers
                        percentile_values = valid_values[(valid_values >= np.percentile(valid_values, 1)) &
                                                        (valid_values <= np.percentile(valid_values, 99))]
                        y_min, y_max = calculate_zero_biased_axis_range(percentile_values, padding_factor=0.15)
                        ax.set_ylim(y_min, y_max)

                    # Add target reduction info as text
                    if 'target_amount' in target_config:
                        target_text = f"Target Reduction: {target_config['target_amount']*100:.1f}%"
                        ax.text(0.02, 0.98, target_text, transform=ax.transAxes,
                               fontsize=10, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                # Save this page after plotting all targets
                plt.tight_layout()
                plt.subplots_adjust(top=0.90, bottom=0.05)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        print(f"Generated {total_pages} pages in Step 4 visualization")

    print(f"Forward model visualization saved to {pdf_path}")
    return pdf_path


def create_forward_model_ratio_visualization(forward_results, config, output_dir, all_data):
    """
    Create PDF visualization for Step 4 forward model results showing ratios relative to baseline.
    Generates a multi-page PDF with one page per (target, response_function, SSP) combination.
    Each page shows global mean time series with two lines:
    - (GDP weather / baseline GDP) - 1: Weather effects only (dashed blue)
    - (GDP climate / baseline GDP) - 1: Full climate effects (solid red)

    Parameters
    ----------
    forward_results : dict
        Results from Step 4 forward integration containing SSP-specific data
    config : dict
        Configuration dictionary containing scenarios, response functions, climate_model.model_name, and run_metadata.json_id
    output_dir : str
        Directory for output files
    all_data : dict
        All loaded NetCDF data for baseline GDP access

    Returns
    -------
    str
        Path to generated PDF file
    """

    # Extract configuration values for standardized naming
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']

    # Generate output filename using standardized pattern
    pdf_filename = f"step4_{json_id}_{model_name}_forward_model_ratios.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Extract metadata
    valid_mask = all_data['_metadata']['valid_mask']
    lat = forward_results['_coordinates']['lat']
    lon = forward_results['_coordinates']['lon']
    years = forward_results['_coordinates']['years']
    response_function_names = forward_results['response_function_names']
    target_names = forward_results['target_names']
    forward_ssps = config['ssp_scenarios']['forward_simulation_ssps']

    # Calculate total pages and subplot layout
    n_targets = len(target_names)
    total_pages = len(forward_ssps) * len(response_function_names)
    print(f"Creating Step 4 ratio visualization: {total_pages} pages ({n_targets} targets per page)")

    # Create PDF with multi-page layout
    with PdfPages(pdf_path) as pdf:
        page_num = 0

        # Loop through combinations (target innermost for 3-per-page grouping)
        for ssp in forward_ssps:
            for response_idx, response_name in enumerate(response_function_names):
                page_num += 1

                # Create new page with 3 subplots (one per target)
                fig = plt.figure(figsize=(12, 16))  # Taller figure for vertical arrangement
                fig.suptitle(f'Step 4: GDP Ratios to Baseline - {model_name}\n'
                           f'SSP: {ssp.upper()} | Response Function: {response_name}',
                           fontsize=16, fontweight='bold', y=0.98)

                # Get SSP-specific data
                ssp_results = forward_results['forward_results'][ssp]
                gdp_climate = ssp_results['gdp_climate']  # [response_func, target, time, lat, lon]
                gdp_weather = ssp_results['gdp_weather']  # [response_func, target, time, lat, lon]

                # Get baseline GDP for this SSP
                baseline_gdp = get_ssp_data(all_data, ssp, 'gdp')  # [time, lat, lon]

                # Calculate global means for baseline (area-weighted using valid cells only)
                baseline_global = []
                for t_idx in range(len(years)):
                    baseline_slice = baseline_gdp[t_idx, :, :]  # [lat, lon]
                    baseline_global.append(calculate_global_mean(baseline_slice, valid_mask))
                baseline_global = np.array(baseline_global)

                # Plot each target on this page
                for target_idx, target_name in enumerate(target_names):
                    target_config = config['gdp_targets'][target_idx]
                    ax = plt.subplot(n_targets, 1, target_idx + 1)  # Dynamic rows, 1 column

                    # Extract data for this combination [time, lat, lon]
                    gdp_climate_combo = gdp_climate[response_idx, target_idx, :, :, :]
                    gdp_weather_combo = gdp_weather[response_idx, target_idx, :, :, :]

                    # Calculate global means for this combination
                    climate_global = []
                    weather_global = []
                    for t_idx in range(len(years)):
                        climate_slice = gdp_climate_combo[t_idx, :, :]  # [lat, lon]
                        weather_slice = gdp_weather_combo[t_idx, :, :]  # [lat, lon]

                        climate_global.append(calculate_global_mean(climate_slice, valid_mask))
                        weather_global.append(calculate_global_mean(weather_slice, valid_mask))

                    climate_global = np.array(climate_global)
                    weather_global = np.array(weather_global)

                    # Calculate ratios minus 1 (fractional change from baseline)
                    weather_ratio = weather_global / baseline_global - 1.0
                    climate_ratio = climate_global / baseline_global - 1.0

                    # Plot the ratio lines
                    ax.plot(years, weather_ratio, 'b--', linewidth=2, label='Weather Effects Only', alpha=0.8)
                    ax.plot(years, climate_ratio, 'r-', linewidth=2, label='Full Climate Effects', alpha=0.8)

                    # Add horizontal line at zero for reference
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

                    # Formatting
                    ax.set_xlabel('Year', fontsize=12)
                    ax.set_ylabel('Fractional Change from Baseline', fontsize=12)
                    target_type_label = target_config.get('target_type', 'unknown').upper()
                    ax.set_title(f'Target: {target_name} [{target_type_label}]', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=10, loc='best')

                    # Set reasonable y-axis limits using zero-biased range with 20% padding
                    all_values = np.concatenate([weather_ratio, climate_ratio])
                    valid_values = all_values[np.isfinite(all_values)]
                    if len(valid_values) > 0:
                        vmin, vmax = calculate_zero_biased_axis_range(valid_values, padding_factor=0.20)
                        ax.set_ylim(vmin, vmax)

                    # Add info box with final values
                    final_weather = weather_ratio[-1]
                    final_climate = climate_ratio[-1]
                    info_text = f'2100 Values:\nWeather: {final_weather:+.3f}\nClimate: {final_climate:+.3f}'
                    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                # Save this page
                plt.tight_layout()
                plt.subplots_adjust(top=0.90, bottom=0.05)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    print(f"Forward model ratio visualization saved to {pdf_path} ({total_pages} pages)")
    return pdf_path


def create_forward_model_maps_visualization(forward_results, config, output_dir, all_data):
    """
    Create spatial maps visualization for Step 4 forward model results.

    Generates two multi-page PDFs:
    1. Linear scale: (y_climate/y_weather) - 1 (original maps)
    2. Log10 scale: log10(y_climate/y_weather) showing extreme values and off-scale points

    Each PDF has one map per (target, response_function, SSP) combination,
    with data averaged over the configured target period.

    Parameters
    ----------
    forward_results : dict
        Results from Step 4 forward integration containing SSP-specific data
    config : dict
        Configuration dictionary with scenarios and response functions
    output_dir : str
        Directory for output files
    model_name : str
        Climate model name for labeling
    all_data : dict
        All loaded NetCDF data (not used but kept for consistency)

    Returns
    -------
    tuple
        (linear_pdf_path, log10_pdf_path) - Paths to both generated PDF files
    """

    # Extract configuration values for standardized naming
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']

    # Generate output filenames using standardized pattern
    linear_pdf_filename = f"step4_{json_id}_{model_name}_forward_model_maps.pdf"
    log10_pdf_filename = f"step4_{json_id}_{model_name}_forward_model_maps_log10.pdf"
    linear_pdf_path = os.path.join(output_dir, linear_pdf_filename)
    log10_pdf_path = os.path.join(output_dir, log10_pdf_filename)

    # Extract metadata
    valid_mask = all_data['_metadata']['valid_mask']
    lat = forward_results['_coordinates']['lat']
    lon = forward_results['_coordinates']['lon']
    response_function_names = forward_results['response_function_names']
    target_names = forward_results['target_names']
    forward_ssps = config['ssp_scenarios']['forward_simulation_ssps']

    # Create coordinate grids for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Get target period from config
    target_start = config['time_periods']['target_period']['start_year']
    target_end = config['time_periods']['target_period']['end_year']

    # Calculate adaptive layout based on number of targets
    n_targets = len(target_names)
    subplot_rows, subplot_cols, fig_size = get_adaptive_subplot_layout(n_targets)

    # Calculate total pages (one page per response function × SSP combination)
    total_pages = len(response_function_names) * len(forward_ssps)

    print(f"Creating Step 4 maps: {n_targets} targets per page across {total_pages} pages")
    print(f"  {len(response_function_names)} response functions × {len(forward_ssps)} SSPs")
    print(f"  Layout: {subplot_rows} rows × {subplot_cols} cols per page")
    print(f"  Generating both linear and log10 scale PDFs in parallel")

    # Create both PDFs with multi-page layout
    with PdfPages(linear_pdf_path) as linear_pdf, PdfPages(log10_pdf_path) as log10_pdf:
        page_num = 0

        # Loop through SSPs and response functions (each gets its own page)
        for ssp in forward_ssps:

            for response_idx, response_name in enumerate(response_function_names):
                response_config = config['response_function_scalings'][response_idx]

                # Create new pages for both PDFs for this SSP × damage combination
                page_num += 1
                linear_fig = plt.figure(figsize=fig_size)
                linear_fig.suptitle(f'Step 4: Forward Model Results (Linear Scale) - {model_name}\n'
                                   f'SSP: {ssp.upper()} | Response Function: {response_config["scaling_name"]} - Page {page_num}/{total_pages}',
                                   fontsize=16, fontweight='bold', y=0.98)

                log10_fig = plt.figure(figsize=fig_size)
                log10_fig.suptitle(f'Step 4: Forward Model Results (Log10 Scale) - {model_name}\n'
                                  f'SSP: {ssp.upper()} | Response Function: {response_config["scaling_name"]} - Page {page_num}/{total_pages}',
                                  fontsize=16, fontweight='bold', y=0.98)

                # Plot all targets on this page
                for target_idx, target_name in enumerate(target_names):
                    target_config = config['gdp_targets'][target_idx]

                    # Calculate subplot position (1-indexed)
                    if subplot_cols == 1:
                        # Single column layout
                        subplot_idx = target_idx + 1
                    else:
                        # Two column layout
                        row = target_idx // subplot_cols
                        col = target_idx % subplot_cols
                        subplot_idx = row * subplot_cols + col + 1

                    # Get SSP-specific data
                    ssp_results = forward_results['forward_results'][ssp]
                    gdp_climate = ssp_results['gdp_climate']  # [response_func, target, time, lat, lon]
                    gdp_weather = ssp_results['gdp_weather']  # [response_func, target, time, lat, lon]

                    # Extract data for this combination: [time, lat, lon]
                    gdp_climate_combo = gdp_climate[response_idx, target_idx, :, :, :]
                    gdp_weather_combo = gdp_weather[response_idx, target_idx, :, :, :]

                    # Calculate time indices for target period using actual time coordinates
                    time_coords = forward_results['_coordinates']['years']
                    target_start_idx = np.where(time_coords == target_start)[0][0]
                    target_end_idx = np.where(time_coords == target_end)[0][0] + 1

                    # Calculate mean ratios over target period for each grid cell
                    # Convert valid_mask to numpy for indexing
                    valid_mask_values = valid_mask.values if hasattr(valid_mask, 'values') else valid_mask
                    nlat, nlon = valid_mask_values.shape
                    impact_ratio_linear = np.full((nlat, nlon), np.nan)  # (climate/weather) - 1
                    impact_ratio_log10 = np.full((nlat, nlon), np.nan)   # log10(climate/weather)

                    for lat_idx in range(nlat):
                        for lon_idx in range(nlon):
                            if valid_mask_values[lat_idx, lon_idx]:
                                # Extract target period time series for this grid cell
                                climate_target = gdp_climate_combo[target_start_idx:target_end_idx, lat_idx, lon_idx]
                                weather_target = gdp_weather_combo[target_start_idx:target_end_idx, lat_idx, lon_idx]

                                if len(climate_target) > 0 and len(weather_target) > 0:
                                    # Add epsilon to prevent division by zero
                                    epsilon = 1e-20
                                    ratios = climate_target / (weather_target + epsilon)
                                    mean_ratio = np.nanmean(ratios)

                                    # Linear scale: (climate/weather) - 1
                                    impact_ratio_linear[lat_idx, lon_idx] = mean_ratio - 1.0

                                    # Log10 scale: log10(climate/weather)
                                    if mean_ratio > 0:
                                        impact_ratio_log10[lat_idx, lon_idx] = np.log10(mean_ratio)
                                    # Zeros or negative values remain NaN (will be white)

                    # Create linear scale map
                    plt.figure(linear_fig.number)
                    linear_ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)

                    # Determine color scale for linear using zero-biased range
                    valid_linear = impact_ratio_linear[valid_mask_values & np.isfinite(impact_ratio_linear)]
                    if len(valid_linear) > 0:
                        lin_vmin, lin_vmax = calculate_zero_biased_range(valid_linear)
                        lin_actual_min = np.min(valid_linear)
                        lin_actual_max = np.max(valid_linear)
                    else:
                        lin_vmin, lin_vmax = -0.01, 0.01
                        lin_actual_min = lin_actual_max = 0.0

                    # Linear map: blue-red colormap (blue=positive, red=negative, white=zero)
                    lin_cmap = plt.cm.RdBu_r
                    lin_norm = mcolors.TwoSlopeNorm(vmin=lin_vmin, vcenter=0.0, vmax=lin_vmax)

                    masked_linear = np.where(valid_mask, impact_ratio_linear, np.nan)
                    lin_im = linear_ax.pcolormesh(lon_grid, lat_grid, masked_linear, cmap=lin_cmap, norm=lin_norm, shading='auto')

                    # Add coastlines
                    linear_ax.contour(lon_grid, lat_grid, valid_mask, levels=[0.5], colors='black', linewidths=0.5, alpha=0.7)

                    # Linear map formatting
                    linear_ax.set_xlabel('Longitude', fontsize=12)
                    linear_ax.set_ylabel('Latitude', fontsize=12)
                    target_type_label = target_config.get('target_type', 'unknown').upper()
                    linear_ax.set_title(f'{ssp.upper()} × {target_name} × {response_name} [{target_type_label}]\n'
                                f'Climate Impact: (GDP_climate/GDP_weather - 1)\nTarget Period Mean: {target_start}-{target_end}',
                                fontsize=14, fontweight='bold')

                    # Linear max/min box with coordinates
                    add_extremes_info_box(linear_ax, lin_actual_min, lin_actual_max,
                                        lat=lat, lon=lon, data_array=impact_ratio_linear, valid_mask=valid_mask)

                    # Linear colorbar
                    lin_cbar = plt.colorbar(lin_im, ax=linear_ax, shrink=0.6, aspect=12)
                    lin_cbar.set_label('Climate Impact Ratio - 1', rotation=270, labelpad=15, fontsize=12)
                    lin_cbar.ax.tick_params(labelsize=10)
                    if hasattr(lin_cbar, 'ax'):
                        lin_cbar.ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1, alpha=0.8)

                    # Set linear map aspect and limits
                    linear_ax.set_xlim(lon.min(), lon.max())
                    linear_ax.set_ylim(lat.min(), lat.max())
                    linear_ax.set_aspect('equal')

                    # Create log10 scale map
                    plt.figure(log10_fig.number)
                    log10_ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)

                    # Determine color scale for log10 - use FULL data range to show outliers
                    valid_log10 = impact_ratio_log10[valid_mask_values & np.isfinite(impact_ratio_log10)]
                    if len(valid_log10) > 0:
                        log_actual_min = np.min(valid_log10)
                        log_actual_max = np.max(valid_log10)
                        # Use full data range (not symmetric) to highlight outliers
                        log_vmin, log_vmax = log_actual_min, log_actual_max

                        # Report extreme ratios for diagnostic purposes
                        if log_actual_max > 5:  # log10(ratio) > 5 means ratio > 100,000
                            # Find indices of maximum ratio
                            max_indices = np.where((valid_mask & np.isfinite(impact_ratio_log10)) &
                                                 (impact_ratio_log10 == log_actual_max))
                            max_lat_idx, max_lon_idx = max_indices[0][0], max_indices[1][0]
                            print(f"    WARNING: Extreme high ratios detected for {ssp.upper()} × {target_name} × {response_name}")
                            print(f"             log10(max_ratio) = {log_actual_max:.2f} (ratio = {10**log_actual_max:.2e})")
                            print(f"             at grid cell indices: lat_idx={max_lat_idx}, lon_idx={max_lon_idx}")
                        if log_actual_min < -5:  # log10(ratio) < -5 means ratio < 0.00001
                            # Find indices of minimum ratio
                            min_indices = np.where((valid_mask & np.isfinite(impact_ratio_log10)) &
                                                 (impact_ratio_log10 == log_actual_min))
                            min_lat_idx, min_lon_idx = min_indices[0][0], min_indices[1][0]
                            print(f"    WARNING: Extreme low ratios detected for {ssp.upper()} × {target_name} × {response_name}")
                            print(f"             log10(min_ratio) = {log_actual_min:.2f} (ratio = {10**log_actual_min:.2e})")
                            print(f"             at grid cell indices: lat_idx={min_lat_idx}, lon_idx={min_lon_idx}")
                    else:
                        log_vmin, log_vmax = -0.1, 0.1
                        log_actual_min = log_actual_max = 0.0

                    # Log10 map: viridis colormap for non-zero-centered scales (standard for outlier detection)
                    log_cmap = plt.cm.viridis
                    log_norm = mcolors.Normalize(vmin=log_vmin, vmax=log_vmax)

                    masked_log10 = np.where(valid_mask, impact_ratio_log10, np.nan)
                    log_im = log10_ax.pcolormesh(lon_grid, lat_grid, masked_log10, cmap=log_cmap, norm=log_norm, shading='auto')

                    # Add coastlines
                    log10_ax.contour(lon_grid, lat_grid, valid_mask, levels=[0.5], colors='black', linewidths=0.5, alpha=0.7)

                    # Log10 map formatting
                    log10_ax.set_xlabel('Longitude', fontsize=12)
                    log10_ax.set_ylabel('Latitude', fontsize=12)
                    log10_ax.set_title(f'{ssp.upper()} × {target_name} × {response_name} [{target_type_label}]\n'
                                f'Climate Impact: log10(GDP_climate/GDP_weather)\nTarget Period Mean: {target_start}-{target_end}',
                                fontsize=14, fontweight='bold')

                    # Log10 max/min box (show both log and original ratio values with coordinates)
                    if len(valid_log10) > 0:
                        max_ratio = 10**log_actual_max
                        min_ratio = 10**log_actual_min

                        # Find max/min locations
                        masked_log = np.where(valid_mask, impact_ratio_log10, np.nan)
                        max_idx = np.unravel_index(np.nanargmax(masked_log), masked_log.shape)
                        min_idx = np.unravel_index(np.nanargmin(masked_log), masked_log.shape)
                        max_lat, max_lon = lat[max_idx[0]], lon[max_idx[1]]
                        min_lat, min_lon = lat[min_idx[0]], lon[min_idx[1]]

                        log_max_min_text = f'Max: {log_actual_max:.2f} (×{max_ratio:.2e}) ({max_lat:.1f}, {max_lon:.1f})\nMin: {log_actual_min:.2f} (×{min_ratio:.2e}) ({min_lat:.1f}, {min_lon:.1f})'
                    else:
                        log_max_min_text = 'No valid data'
                    log10_ax.text(0.02, 0.08, log_max_min_text, transform=log10_ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
                           fontsize=10, verticalalignment='bottom')

                    # Log10 colorbar (no zero reference line for viridis)
                    log_cbar = plt.colorbar(log_im, ax=log10_ax, shrink=0.6, aspect=12)
                    log_cbar.set_label('GDP_climate/GDP_weather', rotation=270, labelpad=15, fontsize=12)
                    log_cbar.ax.tick_params(labelsize=10)
                    # Format ticks to show linear values (10^x) instead of log values (x)
                    format_log_colorbar_ticks(log_cbar, base=10)

                    # Set log10 map aspect and limits
                    log10_ax.set_xlim(lon.min(), lon.max())
                    log10_ax.set_ylim(lat.min(), lat.max())
                    log10_ax.set_aspect('equal')

                # Save both pages after plotting all targets for this SSP × damage combination
                for fig, pdf in [(linear_fig, linear_pdf), (log10_fig, log10_pdf)]:
                    plt.figure(fig.number)
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.90, bottom=0.05)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

    print(f"Forward model maps saved to:")
    print(f"  Linear scale: {linear_pdf_path}")
    print(f"  Log10 scale: {log10_pdf_path}")
    print(f"  ({total_pages} pages each, {n_targets} targets per page)")
    return (linear_pdf_path, log10_pdf_path)


def print_gdp_weighted_scaling_summary(scaling_results: Dict[str, Any], config: Dict[str, Any], all_data: Dict[str, Any], output_dir: str) -> None:
    """
    Generate GDP-weighted summary of scaling factors and write to CSV file.

    Parameters
    ----------
    scaling_results : Dict[str, Any]
        Results from Step 3 containing scaling factors
    config : Dict[str, Any]
        Configuration dictionary
    all_data : Dict[str, Any]
        Pre-loaded NetCDF data containing GDP information
    output_dir : str, optional
        Directory to write CSV file. If None, prints to terminal.
    """
    print("\n" + "="*80)
    print("STEP 3 SCALING FACTOR SUMMARY")
    print("="*80)

    # Extract data
    response_function_names = scaling_results['response_function_names']
    target_names = scaling_results['target_names']
    valid_mask = all_data['_metadata']['valid_mask']
    scaling_factors = scaling_results['scaling_factors']  # [response_func, target, lat, lon]
    optimization_errors = scaling_results['optimization_errors']  # [lat, lon, response_func, target]

    # Get reference SSP GDP data for weighting
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    gdp_data = get_ssp_data(all_data, reference_ssp, 'gdp')  # [time, lat, lon]

    # Calculate target period GDP for weighting (use same period as target calculation)
    target_start = config['time_periods']['target_period']['start_year']
    target_end = config['time_periods']['target_period']['end_year']
    years = all_data['years']

    start_idx = target_start - years[0]
    end_idx = target_end - years[0] + 1

    # Average GDP over target period for weighting (keep as xarray)
    gdp_target_period = gdp_data.isel(time=slice(start_idx, end_idx)).mean(dim='time')  # [lat, lon]

    # Calculate historical period GDP for slope weighting (keep as xarray)
    historical_start = config['time_periods']['historical_period']['start_year']
    historical_end = config['time_periods']['historical_period']['end_year']
    hist_start_idx = historical_start - years[0]
    hist_end_idx = historical_end - years[0] + 1
    gdp_hist_period = gdp_data.isel(time=slice(hist_start_idx, hist_end_idx)).mean(dim='time')  # [lat, lon]

    print(f"GDP-weighted global statistics for scaling factors (using {reference_ssp} GDP, {target_start}-{target_end}):")
    print(f"Valid grid cells: {np.sum(valid_mask)} of {valid_mask.size}")

    # Prepare data for CSV
    csv_data = []

    # Get coordinates from metadata
    metadata = get_grid_metadata(all_data)
    lat_values = metadata['lat']

    # Calculate GDP-weighted global means for each combination
    for target_idx, target_name in enumerate(target_names):
        for resp_idx, resp_name in enumerate(response_function_names):
            # Extract scaling factor for this combination
            scale_data = scaling_factors[resp_idx, target_idx, :, :]  # [lat, lon]

            # Use calculate_global_mean with GDP*scaling data to get proper area+GDP weighted mean
            # Since GDP is in units per km², calculate_global_mean will properly handle the spatial weighting
            gdp_weighted_scaling_data = gdp_target_period * scale_data
            total_weighted_scaling = calculate_global_mean(gdp_weighted_scaling_data, valid_mask)
            total_gdp = calculate_global_mean(gdp_target_period, valid_mask)

            # GDP-weighted mean = area_weighted_mean(GDP * scaling) / area_weighted_mean(GDP)
            if total_gdp > 0:
                gdp_weighted_mean = total_weighted_scaling / total_gdp
            else:
                gdp_weighted_mean = np.nan

            # Calculate GDP-weighted median using user's algorithm
            # Create weights = cos(lat) * GDP and sort by scaling factor values
            area_weights = calculate_area_weights(lat_values)
            area_weights_2d = np.broadcast_to(area_weights[:, np.newaxis], scale_data.shape)

            # Flatten arrays and apply valid mask (convert xarray to numpy for flatten)
            scale_data_values = scale_data.values if hasattr(scale_data, 'values') else scale_data
            flat_scale = scale_data_values.flatten()
            flat_gdp = gdp_target_period.values.flatten()
            flat_area_weights = area_weights_2d.flatten()
            flat_valid = (valid_mask.values if hasattr(valid_mask, 'values') else valid_mask).flatten()

            # Keep only valid entries
            valid_indices = flat_valid & ~np.isnan(flat_scale) & ~np.isnan(flat_gdp)
            valid_scale = flat_scale[valid_indices]
            valid_gdp = flat_gdp[valid_indices]
            valid_area_weights = flat_area_weights[valid_indices]

            if len(valid_scale) > 0:
                # Column 0: weights (cos(lat) * GDP), Column 1: scaling factors
                gdp_area_weights = valid_area_weights * valid_gdp
                combined = np.column_stack([gdp_area_weights, valid_scale])

                # Sort by scaling factor values (column 1)
                sorted_indices = np.argsort(combined[:, 1])
                sorted_combined = combined[sorted_indices]

                # Calculate cumulative sum of weights and find median
                cumsum_weights = np.cumsum(sorted_combined[:, 0])
                total_weight = cumsum_weights[-1]
                half_weight = total_weight / 2.0

                if half_weight <= cumsum_weights[0]:
                    gdp_weighted_median = sorted_combined[0, 1]
                elif half_weight >= cumsum_weights[-1]:
                    gdp_weighted_median = sorted_combined[-1, 1]
                else:
                    gdp_weighted_median = np.interp(half_weight, cumsum_weights, sorted_combined[:, 1])
            else:
                gdp_weighted_median = np.nan

            # Calculate scaling factor max/min statistics
            scale_data_values = scale_data.values if hasattr(scale_data, 'values') else scale_data
            valid_mask_values = valid_mask.values if hasattr(valid_mask, 'values') else valid_mask
            valid_scaling = scale_data_values[valid_mask_values & np.isfinite(scale_data_values)]
            if len(valid_scaling) > 0:
                scaling_max = np.max(valid_scaling)
                scaling_min = np.min(valid_scaling)
            else:
                scaling_max = scaling_min = np.nan

            # Calculate objective function statistics
            error_data = optimization_errors[resp_idx, target_idx, :, :]  # [lat, lon]
            error_data_values = error_data.values if hasattr(error_data, 'values') else error_data
            valid_errors = error_data_values[valid_mask_values & np.isfinite(error_data_values)]

            if len(valid_errors) > 0:
                obj_func_max = np.max(valid_errors)
                obj_func_mean = np.mean(valid_errors)
                obj_func_std = np.std(valid_errors)
                obj_func_min = np.min(valid_errors)
            else:
                obj_func_max = obj_func_mean = obj_func_std = obj_func_min = np.nan

            # Get response function configuration for this scaling
            response_config = config['response_function_scalings'][resp_idx]

            # Extract the 12 scaling parameters, defaulting to 0.0 if not present
            scaling_params = {
                'y_tas1': response_config.get('y_tas1', 0.0),
                'y_tas2': response_config.get('y_tas2', 0.0),
                'k_tas1': response_config.get('k_tas1', 0.0),
                'k_tas2': response_config.get('k_tas2', 0.0),
                'tfp_tas1': response_config.get('tfp_tas1', 0.0),
                'tfp_tas2': response_config.get('tfp_tas2', 0.0),
                'y_pr1': response_config.get('y_pr1', 0.0),
                'y_pr2': response_config.get('y_pr2', 0.0),
                'k_pr1': response_config.get('k_pr1', 0.0),
                'k_pr2': response_config.get('k_pr2', 0.0),
                'tfp_pr1': response_config.get('tfp_pr1', 0.0),
                'tfp_pr2': response_config.get('tfp_pr2', 0.0)
            }

            # Get base economic parameters from config
            model_params = config.get('model_params', {})
            base_params = {
                's': model_params.get('s', 0.3),
                'alpha': model_params.get('alpha', 0.3),
                'delta': model_params.get('delta', 0.1),
                'tas0': model_params.get('tas0', 0.0),
                'pr0': model_params.get('pr0', 0.0)
            }

            # Get target configuration to determine if variability target and calculate g0, g1, g2
            target_config = next((t for t in config['gdp_targets'] if t['target_name'] == target_name), None)
            target_type = target_config.get('target_type', 'damage') if target_config else 'damage'

            # Calculate g0, g1, g2 based on target type
            if target_type == 'variability' and target_config:
                target_shape = target_config.get('target_shape', 'constant')

                if target_shape == 'constant':
                    g0 = target_config.get('global_mean_amount', 1.0)
                    g1 = 0.0
                    g2 = 0.0

                elif target_shape == 'linear':
                    global_mean_amount = target_config.get('global_mean_amount', 1.0)
                    zero_amount_temperature = target_config.get('zero_amount_temperature', 0.0)

                    # Calculate GDP-weighted mean temperature using historical period GDP
                    tas0_2d = all_data.get('tas0_2d')
                    if tas0_2d is not None:
                        tas0_2d_values = tas0_2d.values if hasattr(tas0_2d, 'values') else tas0_2d
                        valid_mask_values = valid_mask.values if hasattr(valid_mask, 'values') else valid_mask
                        gdp_hist_values = gdp_hist_period.values if hasattr(gdp_hist_period, 'values') else gdp_hist_period

                        total_gdp = np.sum(gdp_hist_values[valid_mask_values])
                        gdp_weighted_tas = np.sum(gdp_hist_values[valid_mask_values] * tas0_2d_values[valid_mask_values]) / total_gdp

                        g1 = global_mean_amount / (gdp_weighted_tas - zero_amount_temperature)
                        g0 = -g1 * zero_amount_temperature
                        g2 = 0.0
                    else:
                        g0, g1, g2 = 1.0, 0.0, 0.0

                elif target_shape == 'quadratic':
                    global_mean_amount = target_config.get('global_mean_amount', 1.0)
                    zero_amount_temperature = target_config.get('zero_amount_temperature', 0.0)
                    zero_derivative_temperature = target_config.get('zero_derivative_temperature', 0.0)

                    # Calculate GDP-weighted mean temperature and T^2 using historical period GDP
                    tas0_2d = all_data.get('tas0_2d')
                    if tas0_2d is not None:
                        tas0_2d_values = tas0_2d.values if hasattr(tas0_2d, 'values') else tas0_2d
                        valid_mask_values = valid_mask.values if hasattr(valid_mask, 'values') else valid_mask
                        gdp_hist_values = gdp_hist_period.values if hasattr(gdp_hist_period, 'values') else gdp_hist_period

                        total_gdp = np.sum(gdp_hist_values[valid_mask_values])
                        gdp_weighted_tas = np.sum(gdp_hist_values[valid_mask_values] * tas0_2d_values[valid_mask_values]) / total_gdp
                        gdp_weighted_tas2 = np.sum(gdp_hist_values[valid_mask_values] * tas0_2d_values[valid_mask_values]**2) / total_gdp

                        T0 = zero_amount_temperature
                        T_mean = gdp_weighted_tas
                        T2_mean = gdp_weighted_tas2

                        # Solve the system
                        det = (T_mean - T0) * 2 * T0 - (T2_mean - T0**2) * 1
                        g1 = (global_mean_amount * 2 * T0 - zero_derivative_temperature * (T2_mean - T0**2)) / det
                        g2 = (zero_derivative_temperature * (T_mean - T0) - global_mean_amount * 1) / det
                        g0 = -g1 * T0 - g2 * T0**2
                    else:
                        g0, g1, g2 = 1.0, 0.0, 0.0
                else:
                    g0, g1, g2 = 1.0, 0.0, 0.0
            else:
                # For damage targets, use default values
                g0, g1, g2 = 1.0, 0.0, 0.0

            variability_params = {
                'g0': g0,
                'g1': g1,
                'g2': g2
            }

            # Collect data for CSV
            csv_row = {
                'target_name': target_name,
                'response_function': resp_name,
                'gdp_weighted_mean': gdp_weighted_mean,
                'gdp_weighted_median': gdp_weighted_median,
                'scaling_max': scaling_max,
                'scaling_min': scaling_min,
                'obj_func_max': obj_func_max,
                'obj_func_mean': obj_func_mean,
                'obj_func_std': obj_func_std,
                'obj_func_min': obj_func_min
            }

            # Add the base economic parameters
            csv_row.update(base_params)

            # Add the GDP variability scaling parameters
            csv_row.update(variability_params)

            # Add the 12 climate response scaling parameters
            csv_row.update(scaling_params)

            # Add regression slope statistics if available
            slope_gdp_mean = np.nan
            slope_gdp_median = np.nan
            slope_max = np.nan
            slope_min = np.nan
            slope_std = np.nan

            if 'regression_slopes' in scaling_results:
                regression_data = scaling_results['regression_slopes']
                if resp_name in regression_data['slopes']:
                    if target_name in regression_data['slopes'][resp_name]:
                        # Get GDP-weighted mean from pre-computed value
                        slope_gdp_mean = regression_data['gdp_weighted_means'][resp_name][target_name]

                        # Get slope data array for this combination
                        slope_data = regression_data['slopes'][resp_name][target_name]
                        success_mask = regression_data['success_mask'][resp_name][target_name]

                        # Calculate GDP-weighted median using historical period GDP for slopes
                        valid_slope_indices = valid_mask & success_mask & np.isfinite(slope_data)

                        if np.sum(valid_slope_indices) > 0:
                            # Flatten and filter (convert xarray to numpy first)
                            slope_data_values = slope_data.values if hasattr(slope_data, 'values') else slope_data
                            flat_slope = slope_data_values.flatten()
                            flat_gdp_hist = gdp_hist_period.values.flatten()  # Use historical period GDP for slopes
                            flat_area_weights = area_weights_2d.flatten()
                            flat_valid = (valid_slope_indices.values if hasattr(valid_slope_indices, 'values') else valid_slope_indices).flatten()

                            valid_slope = flat_slope[flat_valid]
                            valid_gdp_slope = flat_gdp_hist[flat_valid]  # Historical period GDP
                            valid_area_weights_slope = flat_area_weights[flat_valid]

                            # GDP-weighted median using historical period GDP
                            gdp_area_weights = valid_area_weights_slope * valid_gdp_slope
                            combined_slope = np.column_stack([gdp_area_weights, valid_slope])
                            sorted_indices = np.argsort(combined_slope[:, 1])
                            sorted_combined = combined_slope[sorted_indices]
                            cumsum_weights = np.cumsum(sorted_combined[:, 0])
                            total_weight = cumsum_weights[-1]
                            half_weight = total_weight / 2.0
                            slope_gdp_median = np.interp(half_weight, cumsum_weights, sorted_combined[:, 1])

                            # Max/min/std
                            slope_max = np.max(valid_slope)
                            slope_min = np.min(valid_slope)
                            slope_std = np.std(valid_slope)

            csv_row['slope_gdp_mean'] = slope_gdp_mean
            csv_row['slope_gdp_median'] = slope_gdp_median
            csv_row['slope_max'] = slope_max
            csv_row['slope_min'] = slope_min
            csv_row['slope_std'] = slope_std

            csv_data.append(csv_row)

    # Write to CSV file if output_dir provided
    if output_dir and csv_data:

        json_id = config['run_metadata']['json_id']
        model_name = config['climate_model']['model_name']
        reference_ssp = config['ssp_scenarios']['reference_ssp']
        csv_filename = f"step3_{json_id}_{model_name}_{reference_ssp}_scaling_factors_summary.csv"
        csv_path = os.path.join(output_dir, csv_filename)

        # Create DataFrame and write to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False, float_format='%.6f')

        print(f"Scaling factors summary written to: {csv_path}")
        print()


def write_variability_calibration_summary(variability_results: Dict[str, Any], config: Dict[str, Any],
                                          all_data: Dict[str, Any], output_dir: str, response_scalings: list) -> None:
    """
    Write GDP-weighted summary of variability calibration results to CSV.

    Parameters
    ----------
    variability_results : Dict[str, Any]
        Results from calculate_variability_climate_response_parameters
    config : Dict[str, Any]
        Configuration dictionary
    all_data : Dict[str, Any]
        Pre-loaded NetCDF data containing GDP information
    output_dir : str
        Directory to write CSV file
    response_scalings : list
        List of response function configurations
    """
    print("\n" + "="*80)
    print("VARIABILITY CALIBRATION SUMMARY")
    print("="*80)

    # Extract data - now organized by response function
    all_regression_slopes = variability_results['all_regression_slopes']  # dict[response_name] -> [nlat, nlon]
    all_regression_success_masks = variability_results['all_regression_success_masks']  # dict[response_name] -> [nlat, nlon]
    response_function_names = variability_results['response_function_names']
    valid_mask = all_data['_metadata']['valid_mask']

    # Get reference SSP GDP data for weighting
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    gdp_data = get_ssp_data(all_data, reference_ssp, 'gdp')  # [time, lat, lon]

    # Calculate historical period GDP for weighting
    hist_period = config['time_periods']['historical_period']
    hist_start = hist_period['start_year']
    hist_end = hist_period['end_year']
    years = all_data['years']

    hist_start_idx = hist_start - years[0]
    hist_end_idx = hist_end - years[0] + 1

    # Average GDP over historical period for weighting (keep as xarray)
    gdp_hist_period = gdp_data.isel(time=slice(hist_start_idx, hist_end_idx)).mean(dim='time')  # [lat, lon]

    print(f"GDP-weighted global statistics for variability calibration (using {reference_ssp} GDP, {hist_start}-{hist_end}):")
    print(f"Valid grid cells: {np.sum(valid_mask)} of {valid_mask.size}")

    # Get coordinates from metadata
    metadata = get_grid_metadata(all_data)
    lat_values = metadata['lat']

    # Prepare list to collect CSV rows (one per response function)
    csv_data_list = []

    # Process each response function separately
    for response_name in response_function_names:
        print(f"\nProcessing response function: {response_name}")

        regression_slopes = all_regression_slopes[response_name]
        regression_success_mask = all_regression_success_masks[response_name]

        # Calculate GDP-weighted statistics for regression slopes
        valid_regression_mask = valid_mask & regression_success_mask
        successful_regressions = np.sum(valid_regression_mask)

        print(f"  Successful regressions: {successful_regressions}/{np.sum(valid_mask)}")

        # GDP-weighted mean
        gdp_weighted_slopes_data = gdp_hist_period * regression_slopes
        total_weighted_slopes = calculate_global_mean(gdp_weighted_slopes_data, valid_regression_mask)
        total_gdp = calculate_global_mean(gdp_hist_period, valid_regression_mask)
        gdp_weighted_mean_slope = total_weighted_slopes / total_gdp if total_gdp > 0 else np.nan

        # Calculate GDP-weighted median for slopes
        area_weights = calculate_area_weights(lat_values)
        area_weights_2d = np.broadcast_to(area_weights[:, np.newaxis], regression_slopes.shape)

        # Flatten and filter (convert xarray to numpy first)
        regression_slopes_values = regression_slopes.values if hasattr(regression_slopes, 'values') else regression_slopes
        flat_slopes = regression_slopes_values.flatten()
        flat_gdp = gdp_hist_period.values.flatten()
        flat_area_weights = area_weights_2d.flatten()
        flat_valid = (valid_regression_mask.values if hasattr(valid_regression_mask, 'values') else valid_regression_mask).flatten()

        valid_indices = flat_valid & ~np.isnan(flat_slopes) & ~np.isnan(flat_gdp)
        if np.sum(valid_indices) > 0:
            valid_slopes = flat_slopes[valid_indices]
            valid_gdp = flat_gdp[valid_indices]
            valid_area_weights = flat_area_weights[valid_indices]

            gdp_area_weights = valid_area_weights * valid_gdp
            combined = np.column_stack([gdp_area_weights, valid_slopes])
            sorted_indices = np.argsort(combined[:, 1])
            sorted_combined = combined[sorted_indices]
            cumsum_weights = np.cumsum(sorted_combined[:, 0])
            total_weight = cumsum_weights[-1]
            half_weight = total_weight / 2.0
            gdp_weighted_median_slope = np.interp(half_weight, cumsum_weights, sorted_combined[:, 1])

            # Min/max/std
            slope_min = np.min(valid_slopes)
            slope_max = np.max(valid_slopes)
            slope_std = np.std(valid_slopes)
        else:
            gdp_weighted_median_slope = slope_min = slope_max = slope_std = np.nan

        # Prepare CSV row for this response function
        csv_row = {
            'response_function': response_name,
            'gdp_weighted_mean_slope': gdp_weighted_mean_slope,
            'gdp_weighted_median_slope': gdp_weighted_median_slope,
            'slope_min': slope_min,
            'slope_max': slope_max,
            'slope_std': slope_std,
            'successful_regressions': successful_regressions,
            'valid_cells': variability_results['valid_cells']
        }

        csv_data_list.append(csv_row)

        print(f"  GDP-weighted mean slope: {gdp_weighted_mean_slope:.6f}")
        print(f"  GDP-weighted median slope: {gdp_weighted_median_slope:.6f}")
        print(f"  Range: {slope_min:.6f} to {slope_max:.6f}")
        print(f"  Std dev: {slope_std:.6f}")

    # Write to CSV (one row per response function)
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']
    csv_filename = f"step3_{json_id}_{model_name}_{reference_ssp}_variability_calibration_summary.csv"
    csv_path = str(Path(output_dir) / csv_filename)

    df = pd.DataFrame(csv_data_list)
    df.to_csv(csv_path, index=False, float_format='%.6f')

    print(f"\nVariability calibration summary written to: {csv_path}")
    print(f"  {len(csv_data_list)} rows (one per response function)")
    print()


def create_target_gdp_visualization(target_results: Dict[str, Any], config: Dict[str, Any],
                                   output_dir: str, reference_ssp: str, valid_mask: np.ndarray,
                                   all_data: Dict[str, Any]) -> str:
    """
    Create comprehensive visualization of target GDP reduction results.

    Generates a single-page PDF with:
    - Global maps showing spatial patterns of each target reduction type
    - Line plot showing response functions vs temperature (if coefficients available)

    Parameters
    ----------
    target_results : Dict[str, Any]
        Results from step1_calculate_target_gdp_changes()
    config : Dict[str, Any]
        Integrated configuration dictionary containing climate_model.model_name and run_metadata.json_id
    output_dir : str
        Output directory path
    reference_ssp : str
        Reference SSP scenario name
    valid_mask : np.ndarray
        Boolean mask for valid economic grid cells
    all_data : Dict[str, Any]
        Combined data structure containing climate and economic data

    Returns
    -------
    str
        Path to generated PDF file
    """

    # Extract configuration values for standardized naming
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']

    # Generate output filename using standardized pattern
    pdf_filename = f"step1_{json_id}_{model_name}_{reference_ssp}_target_gdp_visualization.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Extract metadata and coordinates
    metadata = target_results['_metadata']
    lat = metadata['lat']
    lon = metadata['lon']
    tas_ref = metadata['tas_ref']
    gdp_target = metadata['gdp_target']
    global_tas_ref = metadata['global_tas_ref']
    global_gdp_target = metadata['global_gdp_target']

    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Determine color scale using zero-biased range from all target reduction data
    all_response_values = []
    for target_name, result in target_results.items():
        if target_name != '_metadata':
            reduction_array = result['reduction_array']
            # Convert to numpy and extract valid values
            if hasattr(reduction_array, 'values'):
                reduction_data = reduction_array.values[valid_mask]
            else:
                reduction_data = reduction_array[valid_mask]
            all_response_values.extend(reduction_data.flatten())

    if len(all_response_values) > 0:
        vmin, vmax = calculate_zero_biased_range(all_response_values)
    else:
        vmin, vmax = -0.25, 0.25  # Fallback

    # Use standard blue-red colormap (blue=positive, red=negative, white=zero)
    cmap = plt.cm.RdBu_r

    # Get time series data
    years = all_data['years']
    tas_series = all_data[reference_ssp]['tas']
    gdp_series = all_data[reference_ssp]['gdp']
    area_weights = all_data['_metadata']['area_weights']

    # Calculate GDP-weighted temperature for target period (2080-2100)
    target_period_start = config['time_periods']['target_period']['start_year']
    target_period_end = config['time_periods']['target_period']['end_year']
    gdp_weighted_tas_target = calculate_gdp_weighted_mean(
        tas_series, gdp_series, area_weights, valid_mask, target_period_start, target_period_end
    )

    # Calculate GDP-weighted temperature for historical period (1861-2014)
    historical_period_start = config['time_periods']['historical_period']['start_year']
    historical_period_end = config['time_periods']['historical_period']['end_year']
    gdp_weighted_tas_historical = calculate_gdp_weighted_mean(
        tas_series, gdp_series, area_weights, valid_mask, historical_period_start, historical_period_end
    )

    # Extract reduction arrays and calculate statistics
    reduction_arrays = {}
    global_means = {}
    data_ranges = {}

    # Get all available targets (flexible for different configurations)
    target_names = [key for key in target_results.keys() if key != '_metadata']

    # Separate targets by type
    damage_targets = []
    variability_targets = []

    for target_name in target_names:
        target_config = next(t for t in config['gdp_targets'] if t['target_name'] == target_name)
        target_type = target_config.get('target_type', 'damage')

        if target_type == 'variability':
            variability_targets.append(target_name)
        else:
            damage_targets.append(target_name)

    for target_name in target_names:
        reduction_array = target_results[target_name]['reduction_array']

        reduction_arrays[target_name] = reduction_array

        # Calculate ranges using only valid cells
        if hasattr(reduction_array, 'values'):
            valid_response_data = reduction_array.values[valid_mask]
        else:
            valid_response_data = reduction_array[valid_mask]
        data_ranges[target_name] = {
            'min': float(np.min(valid_response_data)),
            'max': float(np.max(valid_response_data))
        }

    # Calculate GDP-weighted means for verification using time series
    gdp_weighted_means = {}
    for target_name in target_names:
        reduction_array = reduction_arrays[target_name]
        target_config = next(t for t in config['gdp_targets'] if t['target_name'] == target_name)
        target_type = target_config.get('target_type', 'damage')

        # Use appropriate time period based on target type
        if target_type == 'variability':
            period_start = historical_period_start
            period_end = historical_period_end
        else:
            period_start = target_period_start
            period_end = target_period_end

        # GDP-weighted mean calculation using time series: mean_over_time[sum(area×GDP×(1+reduction)) / sum(area×GDP)] - 1
        # Create a broadcast-compatible reduction array
        # Convert reduction_array to numpy if needed for broadcasting
        if hasattr(reduction_array, 'values'):
            reduction_values = reduction_array.values
        else:
            reduction_values = reduction_array
        # Broadcast (1 + reduction) across time dimension: [time, lat, lon]
        reduction_broadcast = xr.ones_like(tas_series) * (1 + reduction_values)[np.newaxis, :, :]
        gdp_weighted_mean = calculate_gdp_weighted_mean(
            reduction_broadcast, gdp_series, area_weights, valid_mask, period_start, period_end
        ) - 1
        gdp_weighted_means[target_name] = gdp_weighted_mean

    with PdfPages(pdf_path) as pdf:
        # Create separate pages for damage and variability targets
        for target_type, type_targets in [('Damage', damage_targets), ('Variability', variability_targets)]:
            if not type_targets:
                continue  # Skip if no targets of this type

            # Calculate color scale for this target type only
            type_response_values = []
            for target_name in type_targets:
                reduction_array = reduction_arrays[target_name]
                if hasattr(reduction_array, 'values'):
                    reduction_data = reduction_array.values[valid_mask]
                else:
                    reduction_data = reduction_array[valid_mask]
                type_response_values.extend(reduction_data.flatten())

            if len(type_response_values) > 0:
                type_vmin, type_vmax = calculate_zero_biased_range(type_response_values)
            else:
                type_vmin, type_vmax = -0.25, 0.25  # Fallback

            # Single page with 4 panels: 3 maps + 1 line plot (2x2 layout)
            fig = plt.figure(figsize=(16, 12))

            # Select period and temperature based on target type
            if target_type == 'Variability':
                period_start, period_end = historical_period_start, historical_period_end
                gdp_weighted_tas = gdp_weighted_tas_historical
            else:
                period_start, period_end = target_period_start, target_period_end
                gdp_weighted_tas = gdp_weighted_tas_target

            # Overall title with target type
            fig.suptitle(f'{target_type} Target GDP Response - {model_name} {reference_ssp.upper()}\n'
                        f'GDP-weighted Mean Temperature ({period_start}-{period_end}): {gdp_weighted_tas:.2f}°C',
                        fontsize=16, fontweight='bold')

            # Calculate layout for maps + line plot
            n_type_targets = len(type_targets)

            if n_type_targets <= 3:
                # Use 2x2 layout: 3 maps + 1 line plot
                subplot_rows, subplot_cols = 2, 2
                fig.set_size_inches(16, 12)
            else:
                # Use 3x2 layout: up to 5 maps + 1 line plot
                subplot_rows, subplot_cols = 3, 2
                fig.set_size_inches(18, 16)

            # Line plot will be in the last position
            line_plot_position = subplot_rows * subplot_cols

            for i, target_name in enumerate(type_targets):  # Show targets of this type
                reduction_array = reduction_arrays[target_name]
                gdp_weighted_mean = gdp_weighted_means[target_name]
                data_range = data_ranges[target_name]
                target_info = target_results[target_name]

                # Calculate subplot position for maps (1-indexed, avoiding last position)
                subplot_idx = i + 1
                ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)

                # Create map with zero-centered normalization (mask invalid cells)
                masked_response_array = np.where(valid_mask, reduction_array, np.nan)
                norm = mcolors.TwoSlopeNorm(vmin=type_vmin, vcenter=0.0, vmax=type_vmax)
                im = ax.pcolormesh(lon_grid, lat_grid, masked_response_array,
                                 cmap=cmap, norm=norm, shading='auto')

                # Format target name for display
                display_name = target_name.replace('_', ' ').title()

                # Get target type from configuration
                target_config = next(t for t in config['gdp_targets'] if t['target_name'] == target_name)
                target_shape = target_config.get('target_shape', 'unknown')

                ax.set_title(f'{display_name} ({target_shape})\n'
                            f'Range: {data_range["min"]:.4f} to {data_range["max"]:.4f}\n'
                            f'GDP-weighted: {gdp_weighted_mean:.6f}',
                            fontsize=12, fontweight='bold')
                ax.set_xlabel('Longitude', fontsize=10)
                ax.set_ylabel('Latitude', fontsize=10)
                ax.grid(True, alpha=0.3)

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.7)
                cbar.set_label('Fractional GDP\nReduction', rotation=270, labelpad=20, fontsize=10)

            # Line plot in last position
            ax4 = plt.subplot(subplot_rows, subplot_cols, line_plot_position)

            # Temperature range for plotting
            tas_range = np.linspace(-5, 35, 1000)

            # Plot each function
            colors = ['black', 'red', 'blue', 'green', 'orange', 'purple']

            for i, target_name in enumerate(type_targets):
                target_info = target_results[target_name]
                color = colors[i % len(colors)]

                # Get target type from configuration
                target_config = next(t for t in config['gdp_targets'] if t['target_name'] == target_name)
                target_shape = target_config.get('target_shape', 'unknown')

                if target_shape == 'constant':
                    # Constant function
                    gdp_targets = config['gdp_targets']
                    const_config = next(t for t in gdp_targets if t['target_name'] == target_name)
                    constant_value = const_config['global_mean_amount']
                    function_values = np.full_like(tas_range, constant_value)
                    label = f'Constant: {constant_value:.3f}'

                    # Horizontal line for constant
                    ax4.plot(tas_range, function_values, color=color, linewidth=2,
                            label=label, alpha=0.8)

                elif target_shape == 'linear':
                    coefficients = target_info['coefficients']
                    if coefficients:
                        # Linear function: reduction = a0 + a1 * T
                        a0, a1 = coefficients['a0'], coefficients['a1']
                        function_values = a0 + a1 * tas_range

                        ax4.plot(tas_range, function_values, color=color, linewidth=2,
                                label=f'Linear: {a0:.4f} + {a1:.4f}×T', alpha=0.8)

                        # Add zero anchor point from config
                        gdp_targets = config['gdp_targets']
                        linear_config = next(t for t in gdp_targets if t['target_name'] == target_name)
                        if 'zero_amount_temperature' in linear_config:
                            zero_tas = linear_config['zero_amount_temperature']
                            ax4.plot(zero_tas, 0.0, 'o', color=color, markersize=8,
                                    label=f'Linear anchor: {zero_tas}°C = 0.0')

                elif target_shape == 'quadratic':
                    coefficients = target_info['coefficients']
                    if coefficients:
                        # Quadratic function: reduction = a0 + a1*T + a2*T²
                        a0, a1, a2 = coefficients['a0'], coefficients['a1'], coefficients['a2']
                        function_values = a0 + a1 * tas_range + a2 * tas_range**2

                        ax4.plot(tas_range, function_values, color=color, linewidth=2,
                                label=f'Quadratic: {a0:.4f} + {a1:.4f}×T + {a2:.6f}×T²', alpha=0.8)

                        # Add calibration points from config
                        gdp_targets = config['gdp_targets']
                        quad_config = next(t for t in gdp_targets if t['target_name'] == target_name)

                        # Handle new derivative-based specification
                        zero_tas = quad_config['zero_amount_temperature']
                        derivative = quad_config['zero_derivative_temperature']
                        ax4.plot(zero_tas, 0, 's', color=color, markersize=8,
                                label=f'Quad zero: {zero_tas}°C = 0 (slope={derivative:.5f})')

            # Add reference lines
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

            # Format line plot
            ax4.set_xlabel('Temperature (°C)', fontsize=12)
            ax4.set_ylabel('Fractional GDP Reduction', fontsize=12)
            ax4.set_title('Target Functions vs Temperature', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=9, loc='best')

            # Set axis limits
            ax4.set_xlim(-5, 35)

            # Calculate y-axis limits from all function values
            all_y_values = []
            for target_name in type_targets:
                target_info = target_results[target_name]

                # Get target type from configuration
                target_config = next(t for t in config['gdp_targets'] if t['target_name'] == target_name)
                target_shape = target_config.get('target_shape', 'unknown')

                if target_shape == 'constant':
                    gdp_targets = config['gdp_targets']
                    const_config = next(t for t in gdp_targets if t['target_name'] == target_name)
                    constant_value = const_config['global_mean_amount']
                    all_y_values.extend([constant_value])

                elif target_shape in ['linear', 'quadratic'] and target_info['coefficients']:
                    coefficients = target_info['coefficients']
                    if target_shape == 'linear':
                        a0, a1 = coefficients['a0'], coefficients['a1']
                        values = a0 + a1 * tas_range
                    elif target_shape == 'quadratic':
                        a0, a1, a2 = coefficients['a0'], coefficients['a1'], coefficients['a2']
                        values = a0 + a1 * tas_range + a2 * tas_range**2
                    all_y_values.extend(values)

            if all_y_values:
                y_min, y_max = calculate_zero_biased_axis_range(all_y_values, padding_factor=0.1)
                ax4.set_ylim(y_min, y_max)

            # Adjust layout to prevent overlap
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # Make room for suptitle
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"Target GDP visualization saved to {pdf_path}")
    return pdf_path




def create_scaling_factors_visualization(scaling_results, config, output_dir, all_data):
    """
    Create comprehensive PDF visualization for Step 3 scaling factor results.

    Generates a multi-panel visualization with one map per response function × target combination.
    For typical case: 3 response functions × 2 targets = 6 small maps on one page.

    Parameters
    ----------
    scaling_results : dict
        Results from Step 3 scaling factor calculation
    config : dict
        Configuration dictionary
    output_dir : str
        Directory for output files
    model_name : str
        Climate model name for labeling

    Returns
    -------
    str
        Path to generated PDF file
    """

    # Extract configuration values for standardized naming
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']
    reference_ssp = config['ssp_scenarios']['reference_ssp']

    # Generate output filename using standardized pattern
    pdf_filename = f"step3_{json_id}_{model_name}_{reference_ssp}_scaling_factors_visualization.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Extract data arrays and metadata
    scaling_factors = scaling_results['scaling_factors']  # [response_func, target, lat, lon]
    valid_mask = all_data['_metadata']['valid_mask']  # [lat, lon]
    response_function_names = scaling_results['response_function_names']
    target_names = scaling_results['target_names']

    # Get coordinate information
    coordinates = scaling_results['_coordinates']
    lat = coordinates['lat']
    lon = coordinates['lon']

    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Get dimensions (scaling_factors is [response_func, target, lat, lon])
    n_response_functions, n_targets, nlat, nlon = scaling_factors.shape

    # Calculate adaptive layout based on number of targets
    subplot_rows, subplot_cols, fig_size = get_adaptive_subplot_layout(n_targets)

    # Calculate total pages (one page per response function)
    total_pages = n_response_functions

    print(f"Creating Step 3 visualization: {n_targets} targets per page across {total_pages} pages")
    print(f"  {n_response_functions} response functions")
    print(f"  Layout: {subplot_rows} rows × {subplot_cols} cols per page")

    # Create PDF with multi-page layout
    with PdfPages(pdf_path) as pdf:
        page_num = 0

        # Loop through response functions (each gets its own page)
        for response_idx, response_name in enumerate(response_function_names):
            response_config = config['response_function_scalings'][response_idx]

            # Create new page for this response function
            page_num += 1
            fig = plt.figure(figsize=fig_size)
            fig.suptitle(f'Step 3: Scaling Factors - {model_name} ({reference_ssp})\n'
                        f'Response Function: {response_config["scaling_name"]} - Page {page_num}/{total_pages}',
                        fontsize=16, fontweight='bold', y=0.98)

            # Plot all targets on this page
            for target_idx, target_name in enumerate(target_names):

                # Calculate subplot position (1-indexed)
                if subplot_cols == 1:
                    # Single column layout
                    subplot_idx = target_idx + 1
                    ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)
                else:
                    # Two column layout
                    row = target_idx // subplot_cols
                    col = target_idx % subplot_cols
                    subplot_idx = row * subplot_cols + col + 1
                    ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)

                # Extract scaling factor map for this combination
                sf_map = scaling_factors[response_idx, target_idx, :, :]

                # Mask invalid cells and ocean
                # Convert to numpy for masking operations
                sf_map_values = sf_map.values if hasattr(sf_map, 'values') else sf_map
                valid_mask_values = valid_mask.values if hasattr(valid_mask, 'values') else valid_mask

                sf_map_masked = np.copy(sf_map_values)
                sf_map_masked[~valid_mask_values] = np.nan

                # Calculate independent zero-biased range for this map
                valid_values = sf_map_values[valid_mask_values & np.isfinite(sf_map_values)]
                if len(valid_values) > 0:
                    vmin, vmax = calculate_zero_biased_range(valid_values)
                    actual_min = np.min(valid_values)
                    actual_max = np.max(valid_values)
                else:
                    vmin, vmax = -0.01, 0.01  # Default range
                    actual_min = actual_max = 0.0

                # Create map with proper zero-centered normalization
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
                im = ax.pcolormesh(lon_grid, lat_grid, sf_map_masked,
                                 cmap='RdBu_r', norm=norm, shading='auto')

                # Add coastlines (basic grid)
                ax.contour(lon_grid, lat_grid, valid_mask_values.astype(float),
                          levels=[0.5], colors='black', linewidths=0.5, alpha=0.3)

                # Labels and formatting (larger fonts for better visibility)
                ax.set_title(f'{response_name}\n{target_name}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Longitude', fontsize=12)
                ax.set_ylabel('Latitude', fontsize=12)
                ax.tick_params(labelsize=10)

                # Set aspect ratio and limits
                ax.set_xlim(lon.min(), lon.max())
                ax.set_ylim(lat.min(), lat.max())
                ax.set_aspect('equal')

                # Add max/min value box with coordinates
                add_extremes_info_box(ax, actual_min, actual_max,
                                     lat=lat, lon=lon, data_array=sf_map, valid_mask=valid_mask)

                # Add colorbar for each subplot (larger for better visibility)
                cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=12)
                cbar.set_label('Scaling Factor', rotation=270, labelpad=15, fontsize=12)
                cbar.ax.tick_params(labelsize=10)

            # Save this page after plotting all targets for this response function
            plt.tight_layout()
            plt.subplots_adjust(top=0.90, bottom=0.05)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"Scaling factors visualization saved to {pdf_path} ({total_pages} pages, {n_targets} targets per page)")
    return pdf_path




def create_objective_function_visualization(scaling_results, config, output_dir, all_data):
    """
    Create comprehensive PDF visualization for Step 3 objective function values.

    Generates one page per 3 maps (same layout as scaling factors) showing optimization
    objective function values across grid cells. Lower values indicate better constraint
    satisfaction for each response function × target combination.

    Parameters
    ----------
    scaling_results : dict
        Results from Step 3 scaling factor calculation
    config : dict
        Configuration dictionary
    output_dir : str
        Directory for output files
    model_name : str
        Climate model name for labeling

    Returns
    -------
    str
        Path to generated PDF file
    """

    # Extract configuration values for standardized naming
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']
    reference_ssp = config['ssp_scenarios']['reference_ssp']

    # Generate output filename using standardized pattern
    pdf_filename = f"step3_{json_id}_{model_name}_{reference_ssp}_objective_function_visualization.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Extract data arrays and metadata
    optimization_errors = scaling_results['optimization_errors']  # [response_func, target, lat, lon]
    valid_mask = all_data['_metadata']['valid_mask']  # [lat, lon]
    response_function_names = scaling_results['response_function_names']
    target_names = scaling_results['target_names']

    # Get coordinate information
    coordinates = scaling_results['_coordinates']
    lat = coordinates['lat']
    lon = coordinates['lon']

    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Get dimensions
    n_response_functions, n_targets, nlat, nlon = optimization_errors.shape

    # Calculate adaptive layout based on number of targets
    subplot_rows, subplot_cols, fig_size = get_adaptive_subplot_layout(n_targets)

    # Calculate total pages (one page per response function)
    total_pages = n_response_functions

    print(f"Creating Step 3 objective function visualization: {n_targets} targets per page across {total_pages} pages")
    print(f"  {n_response_functions} response functions")
    print(f"  Layout: {subplot_rows} rows × {subplot_cols} cols per page")

    # Create PDF with multi-page layout
    with PdfPages(pdf_path) as pdf:
        page_num = 0

        # Loop through response functions (each gets its own page)
        for response_idx, response_name in enumerate(response_function_names):
            response_config = config['response_function_scalings'][response_idx]

            # Create new page for this response function
            page_num += 1
            fig = plt.figure(figsize=fig_size)
            fig.suptitle(f'Step 3: Objective Function Values - {model_name} ({reference_ssp})\n'
                        f'Response Function: {response_config["scaling_name"]} - Page {page_num}/{total_pages}',
                        fontsize=16, fontweight='bold', y=0.98)

            # Plot all targets on this page
            for target_idx, target_name in enumerate(target_names):

                # Calculate subplot position (1-indexed)
                if subplot_cols == 1:
                    # Single column layout
                    subplot_idx = target_idx + 1
                    ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)
                else:
                    # Two column layout
                    row = target_idx // subplot_cols
                    col = target_idx % subplot_cols
                    subplot_idx = row * subplot_cols + col + 1
                    ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)

                # Extract objective function map for this combination
                obj_map = optimization_errors[response_idx, target_idx, :, :]

                # Convert to numpy for masking operations
                obj_map_values = obj_map.values if hasattr(obj_map, 'values') else obj_map
                valid_mask_values = valid_mask.values if hasattr(valid_mask, 'values') else valid_mask

                # Mask invalid cells and ocean
                obj_map_masked = np.copy(obj_map_values)
                obj_map_masked[~valid_mask_values] = np.nan

                # Calculate range for this map (objective function values are always >= 0)
                valid_values = obj_map_values[valid_mask_values & np.isfinite(obj_map_values)]
                if len(valid_values) > 0:
                    actual_min = np.min(valid_values)
                    actual_max = np.max(valid_values)
                else:
                    actual_min = actual_max = 0.0

                # Apply log10 transformation for visualization
                # Set minimum threshold to avoid log(0) issues
                min_threshold = 1e-14
                obj_map_log = np.copy(obj_map_masked)

                # Replace values below threshold with threshold, and zeros/negatives with threshold
                valid_finite_mask = valid_mask_values & np.isfinite(obj_map_values) & (obj_map_values > 0)
                obj_map_log[valid_finite_mask] = np.maximum(obj_map_values[valid_finite_mask], min_threshold)
                obj_map_log[~valid_finite_mask] = np.nan

                # Take log10
                obj_map_log[valid_finite_mask] = np.log10(obj_map_log[valid_finite_mask])

                # Set fixed log10 range: 1e-14 to 1
                vmin_log = np.log10(min_threshold)  # log10(1e-14) = -14
                vmax_log = np.log10(1.0)            # log10(1) = 0

                # Create map with log10 color scaling (viridis good for objective functions)
                cmap = plt.cm.viridis  # Dark = low error (good), bright = high error (poor)
                im = ax.pcolormesh(lon_grid, lat_grid, obj_map_log,
                                 cmap=cmap, vmin=vmin_log, vmax=vmax_log, shading='auto')

                # Add coastlines (basic grid)
                ax.contour(lon_grid, lat_grid, valid_mask_values.astype(float),
                          levels=[0.5], colors='white', linewidths=0.5, alpha=0.7)

                # Labels and formatting (larger fonts for better visibility)
                ax.set_title(f'{response_name}\n{target_name}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Longitude', fontsize=12)
                ax.set_ylabel('Latitude', fontsize=12)
                ax.tick_params(labelsize=10)

                # Set aspect ratio and limits
                ax.set_xlim(lon.min(), lon.max())
                ax.set_ylim(lat.min(), lat.max())
                ax.set_aspect('equal')

                # Add max/min value box with coordinates
                add_extremes_info_box(ax, actual_min, actual_max,
                                     lat=lat, lon=lon, data_array=obj_map, valid_mask=valid_mask)

                # Add colorbar for each subplot (larger for better visibility)
                cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=12)
                cbar.set_label('log₁₀(Objective Function Value)\n(Lower = Better Fit)', rotation=270, labelpad=15, fontsize=12)
                cbar.ax.tick_params(labelsize=10)

            # Save this page after plotting all targets for this response function
            plt.tight_layout()
            plt.subplots_adjust(top=0.90, bottom=0.05)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"Objective function visualization saved to {pdf_path} ({total_pages} pages, {n_targets} targets per page)")
    return pdf_path


def create_regression_slopes_visualization(scaling_results, config, output_dir, all_data):
    """
    Create comprehensive PDF visualization of weather-GDP regression slopes.

    Generates one page per response function showing maps of regression slopes
    from GDP_variability ~ temperature_variability analysis over historical period.

    Parameters
    ----------
    scaling_results : Dict[str, Any]
        Results from Step 3 containing regression slope data
    config : Dict[str, Any]
        Configuration dictionary containing model information
    output_dir : str
        Output directory path

    Returns
    -------
    str
        Path to generated PDF file
    """
    if 'regression_slopes' not in scaling_results:
        print("⚠️  No regression slope data available - skipping visualization")
        return None

    # Extract configuration values for standardized naming
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']
    reference_ssp = config['ssp_scenarios']['reference_ssp']

    # Generate output filename
    pdf_filename = f"step3_{json_id}_{model_name}_{reference_ssp}_regression_slopes_visualization.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Extract regression data
    regression_data = scaling_results['regression_slopes']
    response_function_names = scaling_results['response_function_names']
    target_names = scaling_results['target_names']
    valid_mask = all_data['_metadata']['valid_mask']

    # Get grid coordinates from scaling results
    coordinates = scaling_results['_coordinates']
    lat = coordinates['lat']
    lon = coordinates['lon']
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Use RdBu_r colormap (red=negative, blue=positive, white=zero)
    cmap = plt.cm.RdBu_r

    # Calculate adaptive layout based on number of response functions
    n_response_funcs = len(response_function_names)
    subplot_rows, subplot_cols, fig_size = get_adaptive_subplot_layout(n_response_funcs)

    print(f"Creating regression slopes visualization: {n_response_funcs} response functions")
    print(f"  Layout: {subplot_rows} rows × {subplot_cols} cols")

    # Create the PDF - one page per target
    with PdfPages(pdf_path) as pdf_pages:
        for target_idx, target_name in enumerate(target_names):
            target_config = config['gdp_targets'][target_idx]
            target_type_label = target_config.get('target_type', 'unknown').upper()

            print(f"  Processing target: {target_name} [{target_type_label}]")

            # Create figure with adaptive layout
            fig = plt.figure(figsize=fig_size)
            fig.suptitle(f'Weather-GDP Regression Slopes - {model_name.upper()} {reference_ssp.upper()}\n'
                       f'Target: {target_name} [{target_type_label}] | Historical Period Regression Analysis',
                       fontsize=16, fontweight='bold', y=0.98)

            # Plot each response function
            for response_idx, response_name in enumerate(response_function_names):
                # Check if data exists for this response-target combination
                if response_name not in regression_data['slopes']:
                    continue
                if target_name not in regression_data['slopes'][response_name]:
                    continue

                # Calculate subplot position (1-indexed)
                if subplot_cols == 1:
                    subplot_idx = response_idx + 1
                else:
                    row = response_idx // subplot_cols
                    col = response_idx % subplot_cols
                    subplot_idx = row * subplot_cols + col + 1

                # Extract regression slope data
                slope_data = regression_data['slopes'][response_name][target_name]
                success_mask = regression_data['success_mask'][response_name][target_name]

                # Get GDP-weighted mean from nested dict
                gdp_weighted_mean = regression_data['gdp_weighted_means'][response_name][target_name]

                # Create subplot
                ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)

                # Convert to numpy for boolean indexing operations
                slope_data_values = slope_data.values if hasattr(slope_data, 'values') else slope_data
                valid_mask_values = valid_mask.values if hasattr(valid_mask, 'values') else valid_mask
                success_mask_values = success_mask.values if hasattr(success_mask, 'values') else success_mask

                # Create plot data (set invalid/unsuccessful cells to NaN for white color)
                plot_data = np.full_like(slope_data_values, np.nan)
                combined_mask = valid_mask_values & success_mask_values
                plot_data[combined_mask] = slope_data_values[combined_mask]

                # Calculate panel-specific color scale
                valid_slopes = slope_data_values[combined_mask]
                if len(valid_slopes) > 0:
                    data_min, data_max = np.min(valid_slopes), np.max(valid_slopes)
                    vmin, vmax = calculate_zero_biased_range(valid_slopes)
                else:
                    data_min = data_max = 0.0
                    vmin, vmax = -0.1, 0.1

                # Create panel-specific normalization
                panel_norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

                # Create the map with panel-specific color scale
                mesh = ax.pcolormesh(lon_grid, lat_grid, plot_data, cmap=cmap, norm=panel_norm, shading='auto')

                # Add coastlines
                ax.contour(lon_grid, lat_grid, valid_mask, levels=[0.5], colors='black', linewidths=0.5, alpha=0.7)

                # Formatting
                ax.set_xlabel('Longitude', fontsize=12)
                ax.set_ylabel('Latitude', fontsize=12)

                # Add min/max/mean info box with coordinates
                add_extremes_info_box(ax, data_min, data_max, gdp_weighted_mean,
                                     lat=lat, lon=lon, data_array=slope_data, valid_mask=valid_mask & success_mask)

                ax.set_title(f'{response_name}', fontsize=14, fontweight='bold')

                # Add colorbar
                cbar = plt.colorbar(mesh, ax=ax, shrink=0.6, aspect=12)
                cbar.set_label('Regression Slope (GDP/Temperature variability)', rotation=270, labelpad=15, fontsize=12)
                cbar.ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1, alpha=0.8)

                # Set map aspect and limits
                ax.set_xlim(lon.min(), lon.max())
                ax.set_ylim(lat.min(), lat.max())
                ax.set_aspect('equal')

            plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space for suptitle at top
            pdf_pages.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close(fig)

    print(f"✅ Regression slopes visualization saved: {pdf_path}")
    return pdf_path


def create_baseline_tfp_visualization(tfp_results, config, output_dir, all_data):
    """
    Create comprehensive PDF visualization for Step 2 baseline TFP results.

    Generates one page per forward simulation SSP, each with 3-panel visualization:
    1. Map of mean TFP for reference period
    2. Map of mean TFP for target period
    3. Time series percentile plot (min, 10%, 25%, 50%, 75%, 90%, max)

    Parameters
    ----------
    tfp_results : dict
        Results from Step 2 baseline TFP calculation containing:
        - '_metadata': Coordinate and data information
        - SSP scenarios with TFP time series data
    config : dict
        Configuration dictionary containing time periods, SSP information, climate_model.model_name, and run_metadata.json_id
    output_dir : str
        Directory for output files

    Returns
    -------
    str
        Path to generated PDF file
    """

    # Extract configuration values for standardized naming
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']
    forward_ssps = config['ssp_scenarios']['forward_simulation_ssps']

    # Generate output filename using standardized pattern
    pdf_filename = f"step2_{json_id}_{model_name}_baseline_tfp_visualization.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Extract metadata and coordinates
    metadata = tfp_results['_metadata']
    lat = metadata['lat']
    lon = metadata['lon']
    years = all_data['years']  # Years stored at top level in all_data

    # Get time period information
    ref_start = config['time_periods']['reference_period']['start_year']
    ref_end = config['time_periods']['reference_period']['end_year']
    target_start = config['time_periods']['target_period']['start_year']
    target_end = config['time_periods']['target_period']['end_year']

    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Find time indices for reference and target periods
    ref_mask = (years >= ref_start) & (years <= ref_end)
    target_mask = (years >= target_start) & (years <= target_end)

    # Create PDF with multiple pages (one per forward SSP)
    with PdfPages(pdf_path) as pdf:
        for ssp_idx, viz_ssp in enumerate(forward_ssps):
            print(f"Creating TFP visualization page for {viz_ssp} ({ssp_idx+1}/{len(forward_ssps)})")

            # Extract TFP data for this SSP
            tfp_timeseries = tfp_results[viz_ssp]['tfp_baseline']  # Shape: [time, lat, lon]

            # Calculate period means (axis=0 for time dimension in [time, lat, lon])
            tfp_ref_mean = np.mean(tfp_timeseries[ref_mask], axis=0)  # [lat, lon]
            tfp_target_mean = np.mean(tfp_timeseries[target_mask], axis=0)  # [lat, lon]

            # Use pre-computed valid mask from TFP results (computed once during data loading)
            valid_mask = all_data['_metadata']['valid_mask']
            print(f"  Using pre-computed valid mask: {np.sum(valid_mask)} valid cells")

            # If no valid cells found, use fallback
            if np.sum(valid_mask) == 0:
                print(f"  WARNING: No valid cells found for {viz_ssp} - using sample cells for visualization")
                valid_mask = np.zeros_like(tfp_ref_mean, dtype=bool)
                # Set a few cells as valid for basic visualization
                valid_mask[32, 64] = True  # Single test cell
                valid_mask[16, 32] = True  # Another test cell

            # Calculate percentiles across valid grid cells for time series
            percentiles = [0, 10, 25, 50, 75, 90, 100]  # min, 10%, 25%, 50%, 75%, 90%, max
            percentile_labels = ['Min', '10%', '25%', 'Median', '75%', '90%', 'Max']
            percentile_colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']

            # Extract time series for valid cells only
            tfp_percentiles = np.zeros((len(percentiles), len(years)))

            for t_idx, year in enumerate(years):
                tfp_slice = tfp_timeseries[t_idx]  # [lat, lon]
                if hasattr(tfp_slice, 'values'):
                    valid_values = tfp_slice.values[valid_mask]
                else:
                    valid_values = tfp_slice[valid_mask]

                # Diagnostic output for first few time steps
                if t_idx < 3:
                    print(f"  DEBUG: t_idx={t_idx}, year={year}")
                    print(f"    tfp_slice shape: {tfp_slice.shape}, range: {np.nanmin(tfp_slice):.6e} to {np.nanmax(tfp_slice):.6e}")
                    print(f"    valid_mask sum: {np.sum(valid_mask)}")
                    print(f"    valid_values shape: {valid_values.shape}, range: {np.nanmin(valid_values):.6e} to {np.nanmax(valid_values):.6e}")
                    if len(valid_values) > 0:
                        calculated_percentiles = np.percentile(valid_values, percentiles)
                        print(f"    calculated percentiles: {calculated_percentiles}")
                        print(f"    percentile spread: {calculated_percentiles[-1] - calculated_percentiles[0]:.6e}")

                if len(valid_values) > 0:
                    tfp_percentiles[:, t_idx] = np.percentile(valid_values, percentiles)
                else:
                    tfp_percentiles[:, t_idx] = np.nan

            # Diagnostic output for percentile results
            print(f"  DEBUG: PERCENTILE CALCULATION COMPLETE for {viz_ssp}")
            print(f"    tfp_percentiles shape: {tfp_percentiles.shape}")
            print(f"    tfp_percentiles range: {np.nanmin(tfp_percentiles):.6e} to {np.nanmax(tfp_percentiles):.6e}")
            print(f"    tfp_percentiles NaN count: {np.sum(np.isnan(tfp_percentiles))}")
            print(f"    Sample percentiles for year 0: {tfp_percentiles[:, 0]}")
            print(f"    Sample percentiles for year 50: {tfp_percentiles[:, 50] if tfp_percentiles.shape[1] > 50 else 'N/A'}")

            # Check if all percentiles are identical (explaining overlapping lines)
            for p_idx, percentile_name in enumerate(percentile_labels):
                percentile_timeseries = tfp_percentiles[p_idx, :]
                percentile_range = np.nanmax(percentile_timeseries) - np.nanmin(percentile_timeseries)
                print(f"    {percentile_name} percentile range over time: {percentile_range:.6e}")

            # Determine color scale for maps (TFP values are always positive, use log scale)
            tfp_ref_valid = tfp_ref_mean.values[valid_mask] if hasattr(tfp_ref_mean, 'values') else tfp_ref_mean[valid_mask]
            tfp_target_valid = tfp_target_mean.values[valid_mask] if hasattr(tfp_target_mean, 'values') else tfp_target_mean[valid_mask]
            all_tfp_values = np.concatenate([tfp_ref_valid, tfp_target_valid])
            vmin = np.percentile(all_tfp_values, 5)  # Use 5th percentile for log scale lower bound
            vmax = np.percentile(all_tfp_values, 95)  # Use 95th percentile to handle outliers

            # Create colormap for TFP (use viridis - good for scientific data)
            cmap = plt.cm.viridis

            # Use log normalization for color scale
            norm = LogNorm(vmin=vmin, vmax=vmax)

            # Create page layout for this SSP
            fig = plt.figure(figsize=(18, 10))

            # Overall title
            fig.suptitle(f'Baseline Total Factor Productivity - {model_name} {viz_ssp.upper()}\n'
                        f'Reference Period: {ref_start}-{ref_end} | Target Period: {target_start}-{target_end}',
                        fontsize=16, fontweight='bold')

            # Panel 1: Reference period mean TFP map
            ax1 = plt.subplot(2, 3, (1, 2))  # Top left, spans 2 columns
            im1 = ax1.pcolormesh(lon_grid, lat_grid, tfp_ref_mean,
                                cmap=cmap, norm=norm, shading='auto')
            ax1.set_title(f'Mean TFP: Reference Period ({ref_start}-{ref_end})',
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')

            # Add coastlines if available
            ax1.set_xlim(lon.min(), lon.max())
            ax1.set_ylim(lat.min(), lat.max())

            # Colorbar for reference map
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, aspect=20)
            cbar1.set_label('TFP', rotation=270, labelpad=15)

            # Panel 2: Target period mean TFP map
            ax2 = plt.subplot(2, 3, (4, 5))  # Bottom left, spans 2 columns
            im2 = ax2.pcolormesh(lon_grid, lat_grid, tfp_target_mean,
                                cmap=cmap, norm=norm, shading='auto')
            ax2.set_title(f'Mean TFP: Target Period ({target_start}-{target_end})',
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')

            ax2.set_xlim(lon.min(), lon.max())
            ax2.set_ylim(lat.min(), lat.max())

            # Colorbar for target map
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, aspect=20)
            cbar2.set_label('TFP', rotation=270, labelpad=15)

            # Panel 3: Time series percentiles
            ax3 = plt.subplot(1, 3, 3)  # Right side, full height

            for i, (percentile, label, color) in enumerate(zip(percentiles, percentile_labels, percentile_colors)):
                ax3.plot(years, tfp_percentiles[i], color=color, linewidth=2,
                        label=label, alpha=0.8)

            ax3.set_xlabel('Year', fontsize=12)
            ax3.set_ylabel('Total Factor Productivity (log scale)', fontsize=12)
            ax3.set_title('TFP Percentiles Across Valid Grid Cells', fontsize=14, fontweight='bold')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=10, loc='best')

            # Add reference lines for time periods
            ax3.axvspan(ref_start, ref_end, alpha=0.2, color='blue', label='Reference Period')
            ax3.axvspan(target_start, target_end, alpha=0.2, color='red', label='Target Period')

            # Set reasonable axis limits
            ax3.set_xlim(years.min(), years.max())

            # Set y-axis limits using 90th percentile to avoid outlier distortion
            if np.all(np.isnan(tfp_percentiles)):
                print("  WARNING: All TFP percentile values are NaN - using default y-axis limits")
                ax3.set_ylim(0, 1)
            else:
                # Use 90th percentile (index 5) for max, 0 for min to avoid outlier distortion
                percentile_90_max = np.nanmax(tfp_percentiles[5, :])  # 90th percentile line maximum
                global_min = np.nanmin(tfp_percentiles)
                global_max = np.nanmax(tfp_percentiles)

                # Find coordinates of global min and max values in the full timeseries data
                global_min_full = np.nanmin(tfp_timeseries)
                global_max_full = np.nanmax(tfp_timeseries)

                # Find indices of min and max values (first occurrence if multiple)
                min_indices = np.unravel_index(np.nanargmin(tfp_timeseries), tfp_timeseries.shape)
                max_indices = np.unravel_index(np.nanargmax(tfp_timeseries), tfp_timeseries.shape)
                min_t, min_lat, min_lon = min_indices
                max_t, max_lat, max_lon = max_indices

                # Convert time index to year
                min_year = years[min_t]
                max_year = years[max_t]

                if np.isfinite(percentile_90_max) and percentile_90_max > 0:
                    # Set y-axis from 0 to 90th percentile max with small buffer
                    y_max = percentile_90_max * 1.1
                    ax3.set_ylim(0, y_max)

                    # Add text annotation showing global min/max ranges with coordinates
                    annotation_text = (f'Global range: {global_min_full:.3f} to {global_max_full:.1f}\n'
                                     f'Min: year {min_year}, lat[{min_lat}], lon[{min_lon}]\n'
                                     f'Max: year {max_year}, lat[{max_lat}], lon[{max_lon}]')
                    ax3.text(0.02, 0.98, annotation_text,
                            transform=ax3.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                            fontsize=9)

                    # Create CSV file with time series for top 3 and bottom 3 grid points
                    csv_filename = f"step2_{json_id}_{model_name}_{viz_ssp}_baseline_tfp_extremes.csv"
                    csv_path = os.path.join(output_dir, csv_filename)

                    # Get coordinate arrays (same source as lat/lon used for plotting)
                    lat_coords = lat
                    lon_coords = lon

                    # Find mean log(TFP) over time for each grid cell
                    # Use log to avoid overweighting high TFP values at end of time series
                    log_tfp_timeseries = np.log(tfp_timeseries)
                    mean_log_tfp_spatial = np.nanmean(log_tfp_timeseries, axis=0)  # [lat, lon]

                    # Get valid mask
                    valid_mask = all_data['_metadata']['valid_mask']

                    # Flatten and find top 3, bottom 3, and median 3 indices among valid cells
                    if hasattr(mean_log_tfp_spatial, 'values'):
                        valid_mean_log_tfp = mean_log_tfp_spatial.values[valid_mask]
                    else:
                        valid_mean_log_tfp = mean_log_tfp_spatial[valid_mask]
                    valid_indices = np.where(valid_mask)

                    # Calculate median log(TFP) value
                    median_log_tfp = np.nanmedian(valid_mean_log_tfp)

                    # Get indices of 3 lowest and 3 highest mean log(TFP) values
                    bottom_3_sorted_idx = np.argsort(valid_mean_log_tfp)[:3]
                    top_3_sorted_idx = np.argsort(valid_mean_log_tfp)[-3:][::-1]  # Reverse to get highest first

                    # Get indices of 3 cells closest to median log(TFP)
                    distance_from_median = np.abs(valid_mean_log_tfp - median_log_tfp)
                    median_3_sorted_idx = np.argsort(distance_from_median)[:3]

                    # Map back to 2D coordinates
                    bottom_3_coords = [(valid_indices[0][i], valid_indices[1][i]) for i in bottom_3_sorted_idx]
                    top_3_coords = [(valid_indices[0][i], valid_indices[1][i]) for i in top_3_sorted_idx]
                    median_3_coords = [(valid_indices[0][i], valid_indices[1][i]) for i in median_3_sorted_idx]

                    # Extract data from pre-loaded all_data
                    ssp_data = all_data[viz_ssp]

                    # Build DataFrame with time series for all 9 grid cells
                    extremes_data = {'year': years}

                    # Add bottom 3 (lowest TFP)
                    for i, (lat_idx, lon_idx) in enumerate(bottom_3_coords, 1):
                        extremes_data[f'min{i}_lat'] = lat_coords[lat_idx]
                        extremes_data[f'min{i}_lon'] = lon_coords[lon_idx]
                        extremes_data[f'min{i}_tfp'] = tfp_timeseries[:, lat_idx, lon_idx]
                        extremes_data[f'min{i}_pop'] = ssp_data['pop'][:, lat_idx, lon_idx]
                        extremes_data[f'min{i}_gdp'] = ssp_data['gdp'][:, lat_idx, lon_idx]

                    # Add median 3 (closest to median TFP)
                    for i, (lat_idx, lon_idx) in enumerate(median_3_coords, 1):
                        extremes_data[f'med{i}_lat'] = lat_coords[lat_idx]
                        extremes_data[f'med{i}_lon'] = lon_coords[lon_idx]
                        extremes_data[f'med{i}_tfp'] = tfp_timeseries[:, lat_idx, lon_idx]
                        extremes_data[f'med{i}_pop'] = ssp_data['pop'][:, lat_idx, lon_idx]
                        extremes_data[f'med{i}_gdp'] = ssp_data['gdp'][:, lat_idx, lon_idx]

                    # Add top 3 (highest TFP)
                    for i, (lat_idx, lon_idx) in enumerate(top_3_coords, 1):
                        extremes_data[f'max{i}_lat'] = lat_coords[lat_idx]
                        extremes_data[f'max{i}_lon'] = lon_coords[lon_idx]
                        extremes_data[f'max{i}_tfp'] = tfp_timeseries[:, lat_idx, lon_idx]
                        extremes_data[f'max{i}_pop'] = ssp_data['pop'][:, lat_idx, lon_idx]
                        extremes_data[f'max{i}_gdp'] = ssp_data['gdp'][:, lat_idx, lon_idx]

                    df = pd.DataFrame(extremes_data)
                    df.to_csv(csv_path, index=False)
                    print(f"  Extremes CSV saved: {csv_path}")
                else:
                    print("  WARNING: Invalid 90th percentile range - using default y-axis limits")
                    ax3.set_ylim(0, 1)

            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # Make room for suptitle

            # Save this page to PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"Baseline TFP visualization saved to {pdf_path} ({len(forward_ssps)} pages)")
    return pdf_path



