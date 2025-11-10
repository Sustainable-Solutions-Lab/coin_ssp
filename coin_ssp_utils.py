import copy
import json
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import xarray as xr
from datetime import datetime
from typing import Dict, Any, List
from coin_ssp_models import ScalingParams
from coin_ssp_math_utils import apply_loess_subtract
# Import all moved functions from refactored modules
from coin_ssp_netcdf import (
    write_all_loaded_data_netcdf, save_step4_results_netcdf_split,
    load_step3_results_from_netcdf, save_step1_results_netcdf,
    save_step2_results_netcdf, save_step3_results_netcdf,
    create_serializable_config, extract_year_coordinate, interpolate_to_annual_grid,
    resolve_netcdf_filepath
)
from coin_ssp_math_utils import (
    apply_loess_subtract, calculate_zero_biased_range, calculate_zero_biased_axis_range,
    calculate_area_weights, calculate_time_means, calculate_global_mean
)
from coin_ssp_target_calculations import (
    calculate_constant_target_response, calculate_linear_target_response,
    calculate_quadratic_target_response, calculate_all_target_responses
)

def filter_scaling_params(scaling_config):
    allowed_keys = set(ScalingParams.__dataclass_fields__.keys())
    return {k: v for k, v in scaling_config.items() if k in allowed_keys}


def load_and_concatenate_climate_data(config, ssp_name, data_type):
    """
    Load and concatenate historical and SSP-specific climate data files.

    For temperature and precipitation, loads from:
    1. CLIMATE_{model_name}_historical.nc
    2. CLIMATE_{model_name}_{ssp_name}.nc
    Then concatenates along time dimension.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing climate_model.model_name and file patterns
    ssp_name : str
        SSP scenario name
    data_type : str
        'tas' or 'pr'

    Returns
    -------
    xr.DataArray
        Concatenated data with time, lat, lon coordinates
    """

    climate_model = config['climate_model']
    input_dir = climate_model['input_directory']
    model_name = climate_model['model_name']

    # Get file prefix and variable name from new configuration structure
    if data_type == 'tas':
        prefix = climate_model['file_prefixes']['tas_file_prefix']
        var_name = climate_model['variable_names']['tas_var_name']
    elif data_type == 'pr':
        prefix = climate_model['file_prefixes']['pr_file_prefix']
        var_name = climate_model['variable_names']['pr_var_name']
    else:
        raise ValueError(f"Unsupported data_type for climate data: {data_type}")

    # Historical file
    hist_filename = f"{prefix}_{model_name}_historical.nc"
    hist_filepath = os.path.join(input_dir, hist_filename)

    # SSP-specific file
    ssp_filename = f"{prefix}_{model_name}_{ssp_name}.nc"
    ssp_filepath = os.path.join(input_dir, ssp_filename)

    print(f"    Loading historical: {hist_filename}")
    print(f"    Loading SSP: {ssp_filename}")

    # Load historical data
    hist_ds = xr.open_dataset(hist_filepath, decode_times=False)
    hist_years, hist_valid_mask = extract_year_coordinate(hist_ds)

    # Load SSP data
    ssp_ds = xr.open_dataset(ssp_filepath, decode_times=False)
    ssp_years, ssp_valid_mask = extract_year_coordinate(ssp_ds)

    # Extract DataArrays and apply time masks
    hist_data = hist_ds[var_name].isel(time=hist_valid_mask)
    ssp_data = ssp_ds[var_name].isel(time=ssp_valid_mask)

    # Assign year coordinates
    hist_data = hist_data.assign_coords(time=hist_years)
    ssp_data = ssp_data.assign_coords(time=ssp_years)

    # Convert temperature from Kelvin to Celsius
    if data_type == 'tas':
        hist_data = hist_data - 273.15
        ssp_data = ssp_data - 273.15

    # Concatenate along time dimension
    concatenated_data = xr.concat([hist_data, ssp_data], dim='time')

    print(f"    Historical: {len(hist_years)} years ({hist_years[0]}-{hist_years[-1]})")
    print(f"    SSP: {len(ssp_years)} years ({ssp_years[0]}-{ssp_years[-1]})")
    print(f"    Concatenated: {len(concatenated_data.time)} years ({concatenated_data.time[0].values}-{concatenated_data.time[-1].values})")

    hist_ds.close()
    ssp_ds.close()

    return concatenated_data


def load_and_concatenate_pop_data(config, ssp_name):
    """
    Load and concatenate historical and SSP-specific population data files.

    Loads from:
    1. POP_{model_name}_hist.nc
    2. POP_{model_name}_{short_ssp_name}.nc
    Then concatenates along time dimension.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing climate_model.model_name and file patterns
    ssp_name : str
        SSP scenario name (e.g., 'ssp245')

    Returns
    -------
    xr.DataArray
        Concatenated population data with time, lat, lon coordinates
    """

    climate_model = config['climate_model']
    input_dir = climate_model['input_directory']
    model_name = climate_model['model_name']
    prefix = climate_model['file_prefixes']['pop_file_prefix']
    var_name = climate_model['variable_names']['pop_var_name']

    # Truncate SSP name to short form (e.g., 'ssp245' -> 'ssp2')
    if ssp_name.startswith('ssp') and len(ssp_name) >= 4:
        short_ssp = ssp_name[:4]  # Keep 'ssp' + first digit
    else:
        short_ssp = ssp_name

    # Historical file
    hist_filename = f"{prefix}_{model_name}_hist.nc"
    hist_filepath = os.path.join(input_dir, hist_filename)

    # SSP-specific file
    ssp_filename = f"{prefix}_{model_name}_{short_ssp}.nc"
    ssp_filepath = os.path.join(input_dir, ssp_filename)

    print(f"    Loading historical: {hist_filename}")
    print(f"    Loading SSP: {ssp_filename}")

    # Load historical data
    hist_ds = xr.open_dataset(hist_filepath, decode_times=False)
    hist_years, hist_valid_mask = extract_year_coordinate(hist_ds)
    hist_data = hist_ds[var_name].isel(time=hist_valid_mask)
    hist_data = hist_data.assign_coords(time=hist_years)

    # Load SSP data
    ssp_ds = xr.open_dataset(ssp_filepath, decode_times=False)
    ssp_years, ssp_valid_mask = extract_year_coordinate(ssp_ds)
    ssp_data = ssp_ds[var_name].isel(time=ssp_valid_mask)
    ssp_data = ssp_data.assign_coords(time=ssp_years)

    # Concatenate along time dimension
    concatenated_data = xr.concat([hist_data, ssp_data], dim='time')

    print(f"    Historical: {len(hist_years)} years ({hist_years[0]}-{hist_years[-1]})")
    print(f"    SSP: {len(ssp_years)} years ({ssp_years[0]}-{ssp_years[-1]})")
    print(f"    Concatenated: {len(concatenated_data.time)} years ({concatenated_data.time[0].values}-{concatenated_data.time[-1].values})")

    hist_ds.close()
    ssp_ds.close()

    return concatenated_data


def load_gridded_data(config, case_name):
    """
    Load all NetCDF files and return as a temporally-aligned xarray Dataset.

    All variables are interpolated to annual resolution and aligned to the
    same common year range that all variables share after interpolation.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing climate_model.model_name, file patterns, and time_periods.prediction_period.start_year
    case_name : str
        SSP scenario name

    Returns
    -------
    xr.Dataset
        Dataset containing temporally-aligned DataArrays:
        - 'tas': temperature data (time, lat, lon) - annual, °C
        - 'pr': precipitation data (time, lat, lon) - annual, mm/day
        - 'gdp': GDP data (time, lat, lon) - annual
        - 'pop': population data (time, lat, lon) - annual
    """

    # Extract configuration values
    model_name = config['climate_model']['model_name']
    prediction_year = config['time_periods']['prediction_period']['start_year']

    print(f"Loading and aligning NetCDF data for {model_name} {case_name}...")

    # Load temperature data (concatenate historical + SSP)
    print("  Loading temperature data...")
    tas_raw = load_and_concatenate_climate_data(config, case_name, 'tas')

    # Load precipitation data (concatenate historical + SSP)
    print("  Loading precipitation data...")
    pr_raw = load_and_concatenate_climate_data(config, case_name, 'pr')

    # Load GDP data (single file with short SSP name, no concatenation)
    print("  Loading GDP data...")
    climate_model = config['climate_model']
    input_dir = climate_model['input_directory']
    gdp_prefix = climate_model['file_prefixes']['gdp_file_prefix']
    gdp_var_name = climate_model['variable_names']['gdp_var_name']

    # Truncate SSP name to short form (e.g., 'ssp245' -> 'ssp2')
    if case_name.startswith('ssp') and len(case_name) >= 4:
        short_ssp = case_name[:4]  # Keep 'ssp' + first digit
    else:
        short_ssp = case_name

    gdp_filename = f"{gdp_prefix}_{model_name}_{short_ssp}.nc"
    gdp_file = os.path.join(input_dir, gdp_filename)
    print(f"    Loading: {gdp_filename}")

    gdp_ds = xr.open_dataset(gdp_file, decode_times=False)
    gdp_years_raw, gdp_valid_mask = extract_year_coordinate(gdp_ds)
    gdp_raw = gdp_ds[gdp_var_name].isel(time=gdp_valid_mask)
    gdp_raw = gdp_raw.assign_coords(time=gdp_years_raw)
    gdp_ds.close()

    # Load population data (concatenate historical + SSP)
    print("  Loading population data...")
    pop_raw = load_and_concatenate_pop_data(config, case_name)

    print(f"  Original time ranges:")
    print(f"    Temperature: {int(tas_raw.time.min())}-{int(tas_raw.time.max())} ({len(tas_raw.time)} points)")
    print(f"    Precipitation: {int(pr_raw.time.min())}-{int(pr_raw.time.max())} ({len(pr_raw.time)} points)")
    print(f"    GDP: {int(gdp_raw.time.min())}-{int(gdp_raw.time.max())} ({len(gdp_raw.time)} points)")
    print(f"    Population: {int(pop_raw.time.min())}-{int(pop_raw.time.max())} ({len(pop_raw.time)} points)")

    # Create annual time grids for interpolation
    print("  Interpolating to annual grids...")
    tas_annual_years = np.arange(int(tas_raw.time.min()), int(tas_raw.time.max()) + 1)
    pr_annual_years = np.arange(int(pr_raw.time.min()), int(pr_raw.time.max()) + 1)
    gdp_annual_years = np.arange(int(gdp_raw.time.min()), int(gdp_raw.time.max()) + 1)
    pop_annual_years = np.arange(int(pop_raw.time.min()), int(pop_raw.time.max()) + 1)

    # Interpolate each variable to annual grid using xarray's built-in interpolation
    tas_annual = tas_raw.interp(time=tas_annual_years, method='linear')
    pr_annual = pr_raw.interp(time=pr_annual_years, method='linear')
    gdp_annual = gdp_raw.interp(time=gdp_annual_years, method='linear')
    pop_annual = pop_raw.interp(time=pop_annual_years, method='linear')

    # Find common year range (intersection of all ranges)
    common_start = max(tas_annual_years.min(), pr_annual_years.min(),
                      gdp_annual_years.min(), pop_annual_years.min())
    common_end = min(tas_annual_years.max(), pr_annual_years.max(),
                    gdp_annual_years.max(), pop_annual_years.max())

    print(f"  Common year range: {common_start}-{common_end}")

    # Subset all variables to common years using coordinate-based selection
    tas_aligned = tas_annual.sel(time=slice(common_start, common_end))
    pr_aligned = pr_annual.sel(time=slice(common_start, common_end))
    gdp_aligned = gdp_annual.sel(time=slice(common_start, common_end))
    pop_aligned = pop_annual.sel(time=slice(common_start, common_end))

    print(f"  ✅ All variables aligned to {len(tas_aligned.time)} common years")

    # Apply exponential growth modification for GDP and population before prediction year
    common_years = tas_aligned.time.values

    # Note: by going to historical end year, the historical gdp and pop is the same regardless of which ssp we are using
    idx_historical_end_year = np.where(common_years == config['time_periods']['historical_period']['end_year'])[0][0]
    idx_prediction_year = np.where(common_years == prediction_year)[0][0]

    print(f"  Applying exponential growth modification for years {int(common_years[0])}-{prediction_year}")

    # Convert to numpy for in-place modification (xarray doesn't support this efficiently)
    gdp_values = gdp_aligned.values
    pop_values = pop_aligned.values

    # For each grid cell, modify GDP and population using exponential interpolation
    for lat_idx in range(gdp_values.shape[1]):
        for lon_idx in range(gdp_values.shape[2]):

            # GDP exponential growth
            gdp_first = gdp_values[0, lat_idx, lon_idx]
            gdp_prediction = gdp_values[idx_prediction_year, lat_idx, lon_idx]

            # Population exponential growth
            pop_first = pop_values[0, lat_idx, lon_idx]
            pop_prediction = pop_values[idx_prediction_year, lat_idx, lon_idx]

            if gdp_first > 0 and gdp_prediction > 0:
                for idx in range(1, idx_historical_end_year):
                    gdp_values[idx, lat_idx, lon_idx] = gdp_first * (gdp_prediction / gdp_first) ** (idx / idx_historical_end_year)
                    pop_values[idx, lat_idx, lon_idx] = pop_first * (pop_prediction / pop_first) ** (idx / idx_historical_end_year)

    # Update DataArrays with modified values
    gdp_aligned.values = gdp_values
    pop_aligned.values = pop_values

    # Create and return Dataset
    ds = xr.Dataset({
        'tas': tas_aligned,
        'pr': pr_aligned,
        'gdp': gdp_aligned,
        'pop': pop_aligned
    })

    return ds







# =============================================================================
# Target GDP Reduction Calculation Functions
# Extracted from calculate_target_gdp_amounts.py for reuse in integrated workflow
# =============================================================================









# =============================================================================
# Centralized NetCDF Data Loading Functions
# For efficient loading of all SSP scenario data upfront
# =============================================================================

def load_all_data(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Load all NetCDF files for all SSP scenarios at start of processing.
    
    This function loads all gridded data upfront to avoid repeated file I/O
    operations during processing steps. Since NetCDF files are small, this 
    approach optimizes performance by loading everything into memory once.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Integrated configuration dictionary containing:
        - climate_model: model name and file patterns
        - ssp_scenarios: reference_ssp and forward_simulation_ssps
        - time_periods: reference and target period specifications
    output_dir : str, optional
        Output directory path. If provided, writes NetCDF file with all loaded data
    
    Returns
    -------
    Dict[str, Any]
        Nested dictionary organized as:
        data[ssp_name][data_type] = array
        
        Structure:
        {
            'ssp245': {
                'tas': np.array([time, lat, lon]),  # °C
                'pr': np.array([time, lat, lon]), # mm/day
                'gdp': np.array([time, lat, lon]),          # economic units
                'pop': np.array([time, lat, lon])    # people
            },
            'ssp585': { ... },
            '_metadata': {
                'lat': np.array([lat]),           # latitude coordinates
                'lon': np.array([lon]),           # longitude coordinates  
                'time_periods': time_periods,     # reference and target periods
                'ssp_list': ['ssp245', 'ssp585', ...],  # all loaded SSPs
                'grid_shape': (nlat, nlon),       # spatial dimensions
                'time_shape': ntime               # temporal dimension
            }
        }
    """
    
    print("\n" + "="*60)
    print("LOADING ALL NETCDF DATA")  
    print("="*60)
    
    # Extract configuration
    model_name = config['climate_model']['model_name']
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    forward_ssps = config['ssp_scenarios']['forward_simulation_ssps']
    time_periods = config['time_periods']

    # Create comprehensive SSP list (reference + forward, deduplicated)
    all_ssps = list(set([reference_ssp] + forward_ssps))
    all_ssps.sort()  # Consistent ordering

    print(f"Climate model: {model_name}")
    print(f"Reference SSP: {reference_ssp}")
    print(f"All SSPs to load: {all_ssps}")
    print(f"Total SSP scenarios: {len(all_ssps)}")

    # Log prediction period for exponential growth assumption
    prediction_start = time_periods['prediction_period']['start_year']
    print(f"Exponential growth assumption before year: {prediction_start}")
    
    # Initialize data structure
    all_data = {
        '_metadata': {
            'ssp_list': all_ssps,
            'time_periods': time_periods,
            'model_name': model_name
        }
    }
    
    # Load data for each SSP scenario
    for i, ssp_name in enumerate(all_ssps):
        print(f"\nLoading SSP scenario: {ssp_name} ({i+1}/{len(all_ssps)})")
        
        try:
            # Resolve file paths for this SSP
            tas_file = resolve_netcdf_filepath(config, 'tas', ssp_name)
            pr_file = resolve_netcdf_filepath(config, 'pr', ssp_name)
            gdp_file = resolve_netcdf_filepath(config, 'gdp', ssp_name)
            pop_file = resolve_netcdf_filepath(config, 'pop', ssp_name)
            
            print(f"  Temperature: {os.path.basename(tas_file)}")
            print(f"  Precipitation: {os.path.basename(pr_file)}")
            print(f"  GDP: {os.path.basename(gdp_file)}")
            print(f"  Population: {os.path.basename(pop_file)}")
            
            # Load gridded data for this SSP using existing function
            ssp_ds = load_gridded_data(config, ssp_name)

            # Store Dataset directly (much cleaner than extracting arrays)
            all_data[ssp_name] = {
                'tas': ssp_ds['tas'],
                'pr': ssp_ds['pr'],
                'gdp': ssp_ds['gdp'],
                'pop': ssp_ds['pop']
            }

            # Store metadata from first SSP (coordinates same for all)
            if i == 0:
                all_data['_metadata'].update({
                    'lat': ssp_ds.lat.values,
                    'lon': ssp_ds.lon.values,
                    'grid_shape': (len(ssp_ds.lat), len(ssp_ds.lon)),
                    'time_shape': len(ssp_ds.time)
                })
                # Store years at top level for easy access
                all_data['years'] = ssp_ds.time.values
            
            print(f"  Data shape: {ssp_ds['tas'].shape}")
            print(f"  ✅ Successfully loaded {ssp_name}")
            
        except Exception as e:
            print(f"  ❌ Failed to load {ssp_name}: {e}")
            raise RuntimeError(f"Could not load data for {ssp_name}: {e}")
    
    # Summary information
    nlat, nlon = all_data['_metadata']['grid_shape']
    ntime = all_data['_metadata']['time_shape']
    total_grid_cells = nlat * nlon
    
    print(f"\n📊 Data Loading Summary:")
    print(f"  Grid dimensions: {nlat} × {nlon} = {total_grid_cells} cells")
    print(f"  Time dimension: {ntime} years")
    print(f"  SSP scenarios loaded: {len(all_ssps)}")
    print(f"  Total data arrays: {len(all_ssps) * 4} (4 variables × {len(all_ssps)} SSPs)")
    
    # Estimate memory usage
    bytes_per_array = nlat * nlon * ntime * 8  # 8 bytes per float64
    total_arrays = len(all_ssps) * 4  # 4 data types per SSP
    total_bytes = bytes_per_array * total_arrays
    total_mb = total_bytes / (1024 * 1024)
    
    print(f"  Estimated memory usage: {total_mb:.1f} MB")
    print("  ✅ All NetCDF data loaded successfully")

    # Create global valid mask (check all time points for economic activity)
    print("Computing global valid grid cell mask...")

    # Use reference SSP for validity checking
    ref_gdp = all_data[reference_ssp]['gdp']  # xr.DataArray [time, lat, lon]
    ref_pop = all_data[reference_ssp]['pop']  # xr.DataArray [time, lat, lon]

    # Check if GDP and population are positive at ALL time points (vectorized)
    valid_mask = (ref_gdp > 0).all(dim='time') & (ref_pop > 0).all(dim='time')

    # Count valid cells
    final_valid_count = int(valid_mask.sum())
    total_cells = valid_mask.size

    print(f"  Grid cell validation results:")
    print(f"    Total cells: {total_cells}")
    print(f"    Valid economic grid cells (non-zero GDP and population for all years): {final_valid_count} / {total_cells} ({100*final_valid_count/total_cells:.1f}%)")

    # Add valid mask to metadata (as xarray DataArray)
    all_data['_metadata']['valid_mask'] = valid_mask
    all_data['_metadata']['valid_count'] = final_valid_count

    # Create area weights (cosine of latitude) as a 2D array [lat, lon]
    # This makes it general for any grid structure
    lat_coords = ref_gdp.lat
    lon_coords = ref_gdp.lon
    lat_values = lat_coords.values
    lat_rad = np.deg2rad(lat_values)

    # Create 2D area weights array by broadcasting cos(lat) across longitude
    area_weights_2d = np.cos(lat_rad)[:, np.newaxis] * np.ones(len(lon_coords))

    # Store as xarray DataArray with proper coordinates
    area_weights = xr.DataArray(
        area_weights_2d,
        dims=['lat', 'lon'],
        coords={'lat': lat_coords, 'lon': lon_coords}
    )
    all_data['_metadata']['area_weights'] = area_weights

    print("✅ Area weights computed and stored")

    # Write NetCDF file with all loaded data if output directory is provided
    if output_dir is not None:
        write_all_loaded_data_netcdf(all_data, config, output_dir)

    return all_data





def calculate_tfp_coin_ssp(pop, gdp, params):
    """
    Calculate total factor productivity time series using the Solow-Swan growth model.

    Parameters
    ----------
    pop : array-like
        Time series of pop (L) in people
    gdp : array-like
        Time series of gross domestic product (Y) in $/yr
    params : dict or ModelParams
        Model parameters containing:
        - 's': savings rate (dimensionless)
        - 'alpha': elasticity of output with respect to capital (dimensionless)
        - 'delta': depreciation rate in 1/yr
        
    Returns
    -------
    a : numpy.ndarray
        Total factor productivity time series, normalized to year 0 (A(t)/A(0))
    k : numpy.ndarray
        Capital stock time series, normalized to year 0 (K(t)/K(0))
        
    Notes
    -----
    Assumes system is in steady-state at year 0 with normalized values of 1.
    Uses discrete time integration with 1-year time steps.
    """
    y = gdp/gdp[0] # output normalized to year 0
    l = pop/pop[0] # population normalized to year 0
    k = np.copy(y) # capital stock normalized to year 0
    a = np.copy(y) # total factor productivity normalized to year 0
    s = params.s # savings rate
    alpha = params.alpha # elasticity of output with respect to capital
    delta = params.delta # depreciation rate in units of 1/yr

    # Let's assume that at year 0, the system is in steady-state, do d k / dt = 0 at year 0, and a[0] = 1.
    # 0 == s * y[0] - delta * k[0]
    k[0] = (s/delta) # everything is non0dimensionalized to 1 at year 0
    # y[0] ==  a[0] * k[0]**alpha * l[0]**(1-alpha)

    a[0] = k[0]**(-alpha) # nondimensionalized Total Factor Productivity is 0 in year 0

    # since we are assuming steady state, the capital stock will be the same at the start of year 1

    for t in range(len(y)-1):
        # I want y(t+1) ==  a(t+1) * k(t+1)**alpha * l(t)**(1-alpha)
        #
        # so this means that a(t+1) = y(t + 1) / (k(t+1)**alpha * l(t+1)**(1-alpha))

        dkdt = s * y[t] - delta *k[t]
        k[t+1] = k[t] + dkdt  # assumed time step is one year

        a[t+1] = y[t+1] / (k[t+1]**alpha * l[t+1]**(1-alpha))

    return a, k


# =============================================================================
# Output Writing Functions
# =============================================================================









def calculate_weather_vars(all_data, config):
    """
    Calculate weather (filtered) variables and reference baselines for all SSPs.

    Applies 30-year LOESS filtering to temperature and precipitation data relative to
    reference period mean, creating tas_weather and pr_weather arrays for each SSP.
    Also computes reference climate baselines (tas0_2d, pr0_2d) once and stores them.

    Parameters
    ----------
    all_data : dict
        Data structure containing xarray DataArrays for all SSPs
    config : dict
        Configuration containing time period definitions

    Returns
    -------
    dict
        Updated all_data with tas_weather, pr_weather, tas0_2d, and pr0_2d added
    """

    print("Computing weather variables (filtered climate data)...")

    # Get reference period
    time_periods = config['time_periods']
    ref_start_year = time_periods['reference_period']['start_year']
    ref_end_year = time_periods['reference_period']['end_year']
    ref_period_slice = slice(ref_start_year, ref_end_year)

    filter_width = 30  # years (consistent with existing code)

    # Process each SSP scenario - include reference_ssp and all forward_simulation_ssps
    ssp_scenarios = config['ssp_scenarios']
    reference_ssp = ssp_scenarios['reference_ssp']
    forward_ssps = ssp_scenarios['forward_simulation_ssps']

    # Take union to ensure reference_ssp is always included
    all_ssps_for_weather = list(set([reference_ssp] + forward_ssps))

    for ssp_name in all_ssps_for_weather:

        ssp_data = all_data[ssp_name]

        # Get climate data DataArrays [time, lat, lon]
        tas_data = ssp_data['tas']
        pr_data = ssp_data['pr']

        print(f"  Processing {ssp_name}: {len(tas_data.lat)}x{len(tas_data.lon)} grid cells...")

        # Apply LOESS filtering using xr.apply_ufunc for vectorization
        def apply_loess_to_grid(data_array, filter_width, ref_slice):
            """
            Apply LOESS to each grid cell in parallel.

            Returns xarray DataArray with dimensions [time, lat, lon] to maintain
            consistency with input dimension order.
            """
            result = xr.apply_ufunc(
                lambda ts: apply_loess_subtract(
                    xr.DataArray(ts, dims=['time'], coords={'time': data_array.time}),
                    filter_width,
                    ref_slice
                ).values,
                data_array,
                input_core_dims=[['time']],
                output_core_dims=[['time']],
                vectorize=True
            )
            # xr.apply_ufunc with output_core_dims places time dimension LAST: [lat, lon, time]
            # Transpose to standard order [time, lat, lon] for consistency
            result = result.transpose('time', 'lat', 'lon')
            return result

        # Compute weather variables first for all years
        tas_weather = apply_loess_to_grid(tas_data, filter_width, ref_period_slice)
        pr_weather = apply_loess_to_grid(pr_data, filter_width, ref_period_slice)

        # Now replace the historical period weather with weather that does not see the future
        # so that the weather in the historical period is the same for all ssp/rcp scenarios
        idx_historical_end_year = np.where(tas_data.time == config['time_periods']['historical_period']['end_year'])[0][0]
        tas_weather_historical = apply_loess_to_grid(
            tas_data.isel(time=slice(0, idx_historical_end_year+1)), filter_width, ref_period_slice
        )
        pr_weather_historical = apply_loess_to_grid(
            pr_data.isel(time=slice(0, idx_historical_end_year+1)), filter_width, ref_period_slice
        )

        # Assign using .values to avoid coordinate conflicts
        # Now both arrays have [time, lat, lon] order, so slice first dimension
        tas_weather.values[:idx_historical_end_year+1, :, :] = tas_weather_historical.values
        pr_weather.values[:idx_historical_end_year+1, :, :] = pr_weather_historical.values

        # Add weather variables to SSP data as DataArrays
        ssp_data['tas_weather'] = tas_weather
        ssp_data['pr_weather'] = pr_weather

        print(f"  ✅ {ssp_name} weather variables computed")

    print("✅ All weather variables computed")

    # Compute reference climate baselines using reference SSP
    print("Computing reference climate baselines...")
    reference_ssp = config['ssp_scenarios']['reference_ssp']

    # Get reference SSP data
    ref_ssp_data = all_data[reference_ssp]
    tas_data = ref_ssp_data['tas']
    pr_data = ref_ssp_data['pr']

    # Calculate reference period means using coordinate-based selection
    tas0_2d = tas_data.sel(time=ref_period_slice).mean(dim='time')
    pr0_2d = pr_data.sel(time=ref_period_slice).mean(dim='time')

    # Store reference baselines in all_data for easy access
    all_data['tas0_2d'] = tas0_2d
    all_data['pr0_2d'] = pr0_2d

    print("✅ Reference climate baselines computed and stored")
    return all_data


def get_ssp_data(all_data: Dict[str, Any], ssp_name: str, data_type: str) -> xr.DataArray:
    """
    Extract specific data array from loaded NetCDF data structure.

    Parameters
    ----------
    all_data : Dict[str, Any]
        Result from load_all_data()
    ssp_name : str
        SSP scenario name (e.g., 'ssp245')
    data_type : str
        Data type ('tas', 'pr', 'gdp', 'pop', 'tas_weather', 'pr_weather')

    Returns
    -------
    xr.DataArray
        Data array with time, lat, lon coordinates
    """
    if ssp_name not in all_data:
        raise KeyError(f"SSP scenario '{ssp_name}' not found in loaded data. Available: {all_data['_metadata']['ssp_list']}")

    if data_type not in all_data[ssp_name]:
        raise KeyError(f"Data type '{data_type}' not found for {ssp_name}. Available: {list(all_data[ssp_name].keys())}")

    return all_data[ssp_name][data_type]


def get_grid_metadata(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract grid metadata from loaded NetCDF data structure.

    Parameters
    ----------
    all_data : Dict[str, Any]
        Result from load_all_data()

    Returns
    -------
    Dict[str, Any]
        Metadata dictionary containing coordinates and dimensions
    """
    return {
        'lat': all_data['_metadata']['lat'],
        'lon': all_data['_metadata']['lon'],
        'nlat': len(all_data['_metadata']['lat']),
        'nlon': len(all_data['_metadata']['lon']),
    }
