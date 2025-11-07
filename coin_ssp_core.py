# The purpose of this code is to determine a total factor productivity time series based on 
# the following information:
#
# <gdp? a time series of gdp
# <pop> a time series of population
# 
# <params> a dictionary of parameters
# <params[deelta]> depreciation rate
# <alpha> elasticity of output with respect to capital
# 
#  We have as our core equation:
# 
#  Y(t) = A(t) K(t)**alpha  L(t)**(1-alpha)
#
#  d K(t) / d t = s * Y(t) - delta * K(t)
#
# where Y(t) is gross production, A(t) is total factor productivity, K(t) is capital, and
#  L(t) is population.
#
# In dimension units:
#
# Y(t) -- $/yr
# K(t) -- $
# L(t) -- people
# A(t) -- ($/yr) ($)**(-alpha) * (people)**(alpha-1)
#
# Howeever, this can be non-dimensionalized by dividing each year's value by the value at the first
# year, for example:
#
#  y(t) = Y(t)/Y(0)   where 0 is a stand-in for the reference year
#  k(t) = K(t)/K(0)
#  l(t) = L(t)/L(0)
#  a(t) - A(t)/A(0)
#
#  This makes it so that all of the year(0) values are known.
#
import numpy as np
import xarray as xr
from dataclasses import dataclass
import copy
from scipy.optimize import minimize
from scipy import stats
from coin_ssp_utils import filter_scaling_params, get_ssp_data, get_grid_metadata
from coin_ssp_math_utils import apply_loess_divide, calculate_time_means
from coin_ssp_models import ScalingParams, ModelParams

# Small epsilon to prevent division by zero in ratio calculations
RATIO_EPSILON = 1e-20

def create_scaled_params(params, scaling, scale_factor):
    """
    Create scaled parameters from base parameters, scaling template, and scale factor.
    Computed once, used many times.
    """
    params_scaled = copy.copy(params)
    params_scaled.k_tas1   = scale_factor * scaling.k_tas1
    params_scaled.k_tas2   = scale_factor * scaling.k_tas2
    params_scaled.tfp_tas1 = scale_factor * scaling.tfp_tas1
    params_scaled.tfp_tas2 = scale_factor * scaling.tfp_tas2
    params_scaled.y_tas1   = scale_factor * scaling.y_tas1
    params_scaled.y_tas2   = scale_factor * scaling.y_tas2
    params_scaled.k_pr1    = scale_factor * scaling.k_pr1
    params_scaled.k_pr2    = scale_factor * scaling.k_pr2
    params_scaled.tfp_pr1  = scale_factor * scaling.tfp_pr1
    params_scaled.tfp_pr2  = scale_factor * scaling.tfp_pr2
    params_scaled.y_pr1    = scale_factor * scaling.y_pr1
    params_scaled.y_pr2    = scale_factor * scaling.y_pr2
    return params_scaled

def calculate_coin_ssp_forward_model(tfp, pop, tas, pr, params: ModelParams):
    # Convert xarray DataArrays to numpy for consistent indexing
    tfp = np.asarray(tfp)
    pop = np.asarray(pop)
    tas = np.asarray(tas)
    pr = np.asarray(pr)

    # This function calculates the forward model for the COIN-SSP economic model.
    # It takes in total factor productivity (tfp), population (pop), 
    # temperature (tas), and a set of model parameters (params).
    # The function returns the adjusted total factor productivity (tfp_adj),
    # capital stock (k), and output (y) time series.

    # Extract parameters from the ModelParams dataclass
    s = params.s  # savings rate
    alpha = params.alpha  # elasticity of output with respect to capital
    delta = params.delta  # depreciation rate in 1/yr
    # Note: The following parameters default to 0 if not provided
    tas0 = params.tas0  # reference temperature for temperature (tas) response
    pr0 = params.pr0  # reference precipitation for precipitation (pr) response
    k_tas1 = params.k_tas1  # linear temperature sensitivity for capital loss
    k_tas2 = params.k_tas2  # quadratic temperature sensitivity for capital loss        
    k_pr1 = params.k_pr1  # linear precipitation sensitivity for capital loss
    k_pr2 = params.k_pr2  # quadratic precipitation sensitivity for capital loss
    tfp_tas1 = params.tfp_tas1  # linear temperature sensitivity for TFP loss
    tfp_tas2 = params.tfp_tas2  # quadratic temperature sensitivity for TFP loss
    tfp_pr1 = params.tfp_pr1  # linear precipitation sensitivity for TFP loss
    tfp_pr2 = params.tfp_pr2  # quadratic precipitation sensitivity for TFP loss
    y_tas1 = params.y_tas1  # linear temperature sensitivity for output loss
    y_tas2 = params.y_tas2  # quadratic temperature sensitivity for output loss
    y_pr1 = params.y_pr1  # linear precipitation sensitivity for output loss
    y_pr2 = params.y_pr2  # quadratic precipitation sensitivity for output loss
    g0 = params.g0  # GDP variability scaling constant term
    g1 = params.g1  # GDP variability scaling linear term
    g2 = params.g2  # GDP variability scaling quadratic term

    # Clipping bounds to prevent extreme values and overflow
    clip_low = 0
    clip_high = 1e12

    # convert TFP into interannual fractional increase in TFP
    tfp_growth = tfp[1:]/tfp[:-1] # note that this is one element shorter than the other vectors

    # non-dimensionalize the input data
    y = np.ones_like(pop)  # output normalized to 1 at year 0
    l = pop/pop[0] # population normalized to year 0
    k = np.copy(y) # capital stock normalized to year 0
    a = np.copy(y) # total factor productivity normalized to year 0

    # assume that at year 0, the system is in steady-state, do d k / dt = 0 at year 0, and a[0] = 1.
    # 0 == s * y[0] - delta * k[0]
    k[0] = (s/delta) # everything is non0dimensionalized to 1 at year 0
    # y[0] ==  a[0] * k[0]**alpha * l[0]**(1-alpha)
    a0 = k[0]**(-alpha) # nondimensionalized Total Factor Productivity in year 0 in steady state without climate impacts

    # compute climate effect on capital stock, tfp growth rate, and output
    #note that these are all defined so a positive number means a positive economic impact

    # Calculate variability scaling factors for current conditions and reference
    # Note: For "damage" targets, g0=1.0, g1=0.0, g2=0.0, so g_scaling = g_ref_scaling = 1.0
    # The g(T) scaling is only used for "variability" targets to scale GDP-weather relationships by temperature
    g_scaling = g0 + g1 * tas + g2 * tas**2
    g_ref_scaling = g0 + g1 * tas0 + g2 * tas0**2

    # Define climate response functions f_y, f_k, f_tfp
    def f_y(T, P):
        return (y_tas1 * T + y_tas2 * T**2) + (y_pr1 * P + y_pr2 * P**2)

    def f_k(T, P):
        return (k_tas1 * T + k_tas2 * T**2) + (k_pr1 * P + k_pr2 * P**2)

    def f_tfp(T, P):
        return (tfp_tas1 * T + tfp_tas2 * T**2) + (tfp_pr1 * P + tfp_pr2 * P**2)

    # Calculate climate response factors using cleaner formulation
    y_climate = 1.0 + g_scaling * f_y(tas, pr) - g_ref_scaling * f_y(tas0, pr0)
    k_climate = 1.0 + g_scaling * f_k(tas, pr) - g_ref_scaling * f_k(tas0, pr0)
    tfp_climate = 1.0 + g_scaling * f_tfp(tas, pr) - g_ref_scaling * f_tfp(tas0, pr0)  


    a[0] = a0 * tfp_climate[0] # initial TFP adjusted for climate in year 0

    for t in range(len(y)-1):

        # compute climate responses
        # Note that the climate response is computed at the start of year t, and then applied
        # to the change in capital stock and TFP over year t to year t+1
        
        # in year t, we are assume that the damage to capital stock occurs before production occurs
        # so that production in year t is based on the capital stock after climate damage
        # and before investment occurs
        y[t] = a[t] * np.maximum(0, k[t]*k_climate[t])**alpha * l[t]**(1-alpha) * y_climate[t]

        # Clip output to prevent extreme values
        y[t] = np.clip(y[t], clip_low, clip_high)

        # capital stock is then updated based on savings, depereciation, and climate damage
        k[t+1] = (k[t] * k_climate[t]) + s * y[t] - delta * k[t]

        # Clip capital to prevent extreme values
        k[t+1] = np.clip(k[t+1], clip_low, clip_high) 

        # apply climate effect to tfp growth rate
        a[t+1] = a[t] * tfp_growth[t] * tfp_climate[t+1]  # tfp is during the year t to t+1

        # Clip TFP to prevent extreme values
        a[t+1] = np.clip(a[t+1], clip_low, clip_high)

    # compute the last year's output
    t = len(y)-1
    y[t] = a[t] * np.maximum(0, k[t]*k_climate[t])**alpha * l[t]**(1-alpha) * y_climate[t]

    # Clip final output to prevent extreme values
    y[t] = np.clip(y[t], clip_low, clip_high)

    return y, a, k, y_climate, tfp_climate, k_climate


def optimize_climate_response_scaling(
        gridcell_data, params: ModelParams, scaling: ScalingParams,
        config, gdp_target):
    """
    Optimize the scaling factor with adaptive bounds expansion.

    Performs initial optimization within specified bounds. If the result hits the bounds
    (within tolerance), expands search in the indicated direction by factor of 5 and
    retries optimization. Avoids re-searching the original parameter space.
    Returns the better of the two optimization results.

    Returns (optimal_scale, final_error, scaled_params).

    Parameters
    ----------
    gridcell_data : dict
        Grid cell time series data
    params : ModelParams
        Model parameters
    scaling : ScalingParams
        Scaling parameters
    config : dict
        Configuration dictionary containing time_periods and other optimization settings
    gdp_target : dict
        Current GDP target configuration containing target_type and other target-specific settings

    Notes
    -----
    The adaptive bounds expansion uses efficient directional search:
    - Hit lower bound: new search region is [10×lower, old_lower]
    - Hit upper bound: new search region is [old_upper, 10×upper]
    This avoids re-searching the original bounds while expanding in the promising direction.
    """
    # Define optimization parameters
    x0 = -0.001  # starting guess for the scale
    bounds = (-0.1, 0.1)  # initial bounds for optimization (will be expanded if needed)
    maxiter = 500  # maximum iterations per optimization attempt
    tol = 1e-8  # tolerance for optimization

    # Ensure starting guess is inside bounds
    lo, hi = bounds
    x0 = float(np.clip(x0, lo, hi))

    # Extract configuration parameters
    target_period = config['time_periods']['target_period']
    target_type = gdp_target['target_type']
    historical_end_year = config['time_periods']['historical_period']['end_year']

    # Precompute target period indices
    start_year = target_period['start_year']
    end_year = target_period['end_year']
    years = gridcell_data['years']

    target_mask = (years >= start_year) & (years <= end_year)
    target_indices = np.where(target_mask)[0]

    if len(target_indices) == 0:
        raise ValueError(f"No years found in target period {start_year}-{end_year}")


    def objective_damage(xarr):
        # xarr is a length-1 array because we're using scipy.optimize.minimize
        scale = float(xarr[0])

        # Create scaled parameters using helper function
        pc = create_scaled_params(params, scaling, scale)

        # Climate run
        y_climate, *_ = calculate_coin_ssp_forward_model(
            gridcell_data['tfp_baseline'], gridcell_data['pop'],
            gridcell_data['tas'], gridcell_data['pr'], pc
        )

        # Weather (baseline) run
        y_weather, *_ = calculate_coin_ssp_forward_model(
            gridcell_data['tfp_baseline'], gridcell_data['pop'],
            gridcell_data['tas_weather'], gridcell_data['pr_weather'], pc
        )

        # Calculate ratios for all years in target period
        ratios = y_climate[target_indices] / (y_weather[target_indices] + RATIO_EPSILON)
        mean_ratio = np.mean(ratios)

        # Clip extreme ratios to prevent overflow in optimization
        clipped_ratio = np.clip(mean_ratio, -1e6, 1e6)

        target = 1.0 + gdp_target['global_mean_amount']
        objective_value = (clipped_ratio - target) ** 2

        # Additional safety: cap objective function at large finite value
        return min(objective_value, 1e20)


    # Initial optimization with original bounds
    res = minimize(
        objective_damage,
        x0=[x0],
        bounds=[bounds],                # keeps search inside [lo, hi]
        method="L-BFGS-B",              # supports bounds + numeric gradient
        options={"maxiter": maxiter},
        tol=tol
    )

    optimal_scale = float(res.x[0])
    final_error = float(res.fun)

    # Check if optimization hit the bounds and retry with expanded bounds if needed
    lo, hi = bounds
    bound_tolerance = 1e-6  # Consider "at bounds" if within this tolerance

    if (optimal_scale <= lo + bound_tolerance) or (optimal_scale >= hi - bound_tolerance):
        # Hit bounds - expand by factor of 10 and retry
        expansion_factor = 10.0

        # Diagnostic: track bounds expansion
        bound_type = "lower" if optimal_scale <= lo + bound_tolerance else "upper"

        if optimal_scale <= lo + bound_tolerance:
            # Hit lower bound - search lower region, avoid re-searching upper region
            new_lo = lo * expansion_factor if lo < 0 else lo / expansion_factor  # Expand downward 5x
            new_hi = lo  # Upper bound becomes the old lower bound
        else:
            # Hit upper bound - search higher region, avoid re-searching lower region
            new_lo = hi  # Lower bound becomes the old upper bound
            new_hi = hi * expansion_factor if hi > 0 else hi / expansion_factor  # Expand upward 5x

        expanded_bounds = (new_lo, new_hi)

        # Retry optimization with expanded bounds
        res_expanded = minimize(
            objective_damage,
            x0=[optimal_scale],  # Start from previous result
            bounds=[expanded_bounds],
            method="L-BFGS-B",
            options={"maxiter": maxiter},
            tol=tol
        )

        # Use expanded result if it's better (lower error)
        if res_expanded.success and res_expanded.fun < final_error:
            old_scale = optimal_scale
            old_error = final_error
            optimal_scale = float(res_expanded.x[0])
            final_error = float(res_expanded.fun)

            # Bounds expansion improved result
        else:
            # Expansion didn't help, keep original result
            pass

        # Optional: Could add another round of expansion if still hitting bounds

    scaled_params = create_scaled_params(params, scaling, optimal_scale)
    return optimal_scale, final_error, scaled_params


def process_response_target_optimization(
    target_idx, gdp_target, target_results, response_scalings,
    tas_data, pr_data, pop_data, gdp_data,
    reference_tfp, valid_mask, tfp_baseline, years, config,
    scaling_factors, optimization_errors, convergence_flags, scaled_parameters,
    total_grid_cells, successful_optimizations,
    tas_weather_data, pr_weather_data
):
    """
    Process optimization for a single damage target across all response functions and grid cells.

    This function encapsulates the nested loops and per-grid-cell optimization that was
    previously inline in main.py (lines 622-700). It handles all response functions for
    a single GDP target.

    Parameters
    ----------
    target_idx : int
        Index of current GDP target
    gdp_target : dict
        GDP target configuration
    target_results : dict
        Target GDP results containing reduction arrays
    response_scalings : list
        List of damage scaling configurations
    tas_data, pr_data, pop_data, gdp_data : np.ndarray
        Climate and economic data arrays [time, lat, lon]
    reference_tfp, valid_mask, tfp_baseline : np.ndarray
        TFP reference data
    years : np.ndarray
        Years array
    config : dict
        Configuration dictionary
    scaling_factors, optimization_errors, convergence_flags, scaled_parameters : np.ndarray
        Output arrays to populate (modified in place)
    tas_weather_data, pr_weather_data : np.ndarray
        Pre-computed weather variables [time, lat, lon]
    total_grid_cells, successful_optimizations : int
        Counters (modified in place via list trick)

    Returns
    -------
    dict
        Updated counters: {'total_grid_cells': int, 'successful_optimizations': int}
    """

    target_name = gdp_target['target_name']
    target_response_array = target_results[target_name]['reduction_array']  # [lat, lon]

    print(f"\nProcessing GDP target: {target_name} ({target_idx+1}/?)")

    # Calculate reference period from config
    time_periods = config['time_periods']
    ref_start_year = time_periods['reference_period']['start_year']
    ref_end_year = time_periods['reference_period']['end_year']

    # Get coordinate values
    lat_coords = valid_mask.coords['lat'].values
    lon_coords = valid_mask.coords['lon'].values
    n_response_functions = len(response_scalings)

    for response_idx, response_scaling in enumerate(response_scalings):
        scaling_name = response_scaling['scaling_name']
        print(f"  Response function: {scaling_name} ({response_idx+1}/{n_response_functions})")

        # Create ScalingParams for this response function
        scaling_config = filter_scaling_params(response_scaling)
        scaling_params = ScalingParams(**scaling_config)

        for lat_idx, lat_val in enumerate(lat_coords):
            # Progress indicator: print dot for each latitude band
            print(".", end="", flush=True)

            for lon_idx, lon_val in enumerate(lon_coords):

                # Check if grid cell is valid (has economic activity)
                if not valid_mask.sel(lat=lat_val, lon=lon_val):
                    continue

                total_grid_cells += 1

                # Extract time series for this grid cell using coordinate-based selection
                cell_tas = tas_data.sel(lat=lat_val, lon=lon_val).values
                cell_pr = pr_data.sel(lat=lat_val, lon=lon_val).values
                cell_pop = pop_data.sel(lat=lat_val, lon=lon_val).values
                cell_gdp = gdp_data.sel(lat=lat_val, lon=lon_val).values
                cell_tfp_baseline = tfp_baseline.sel(lat=lat_val, lon=lon_val).values

                # Get target reduction for this grid cell
                target_response = target_response_array.sel(lat=lat_val, lon=lon_val)

                # Get weather (filtered) time series from pre-computed arrays
                cell_tas_weather = tas_weather_data.sel(lat=lat_val, lon=lon_val).values
                cell_pr_weather = pr_weather_data.sel(lat=lat_val, lon=lon_val).values

                # Create parameters for this grid cell using factory
                # Calculate reference means using coordinate-based selection
                tas_ref_mean = float(tas_data.sel(time=slice(ref_start_year, ref_end_year), lat=lat_val, lon=lon_val).mean().values)
                pr_ref_mean = float(pr_data.sel(time=slice(ref_start_year, ref_end_year), lat=lat_val, lon=lon_val).mean().values)

                params_cell = config['model_params_factory'].create_for_step(
                    "grid_cell_optimization",
                    tas0=tas_ref_mean,
                    pr0=pr_ref_mean
                )

                # Create cell data dictionary matching gridcell_data structure
                cell_data = {
                    'years': years,
                    'pop': cell_pop,
                    'gdp': cell_gdp,
                    'tas': cell_tas,
                    'pr': cell_pr,
                    'tas_weather': cell_tas_weather,
                    'pr_weather': cell_pr_weather,
                    'tfp_baseline': cell_tfp_baseline
                }

                # Run per-grid-cell optimization
                optimal_scale, final_error, params_scaled = optimize_climate_response_scaling(
                    cell_data, params_cell, scaling_params, config, gdp_target
                )

                # Store results using coordinate-based assignment
                scaling_factors.loc[dict(response_func=scaling_name, target=target_name, lat=lat_val, lon=lon_val)] = optimal_scale
                optimization_errors.loc[dict(response_func=scaling_name, target=target_name, lat=lat_val, lon=lon_val)] = final_error
                convergence_flags.loc[dict(response_func=scaling_name, target=target_name, lat=lat_val, lon=lon_val)] = True

                # Store scaled response function parameters using coordinate-based assignment
                scaled_parameters.loc[dict(response_func=scaling_name, target=target_name, param='k_tas1', lat=lat_val, lon=lon_val)] = params_scaled.k_tas1
                scaled_parameters.loc[dict(response_func=scaling_name, target=target_name, param='k_tas2', lat=lat_val, lon=lon_val)] = params_scaled.k_tas2
                scaled_parameters.loc[dict(response_func=scaling_name, target=target_name, param='k_pr1', lat=lat_val, lon=lon_val)] = params_scaled.k_pr1
                scaled_parameters.loc[dict(response_func=scaling_name, target=target_name, param='k_pr2', lat=lat_val, lon=lon_val)] = params_scaled.k_pr2
                scaled_parameters.loc[dict(response_func=scaling_name, target=target_name, param='tfp_tas1', lat=lat_val, lon=lon_val)] = params_scaled.tfp_tas1
                scaled_parameters.loc[dict(response_func=scaling_name, target=target_name, param='tfp_tas2', lat=lat_val, lon=lon_val)] = params_scaled.tfp_tas2
                scaled_parameters.loc[dict(response_func=scaling_name, target=target_name, param='tfp_pr1', lat=lat_val, lon=lon_val)] = params_scaled.tfp_pr1
                scaled_parameters.loc[dict(response_func=scaling_name, target=target_name, param='tfp_pr2', lat=lat_val, lon=lon_val)] = params_scaled.tfp_pr2
                scaled_parameters.loc[dict(response_func=scaling_name, target=target_name, param='y_tas1', lat=lat_val, lon=lon_val)] = params_scaled.y_tas1
                scaled_parameters.loc[dict(response_func=scaling_name, target=target_name, param='y_tas2', lat=lat_val, lon=lon_val)] = params_scaled.y_tas2
                scaled_parameters.loc[dict(response_func=scaling_name, target=target_name, param='y_pr1', lat=lat_val, lon=lon_val)] = params_scaled.y_pr1
                scaled_parameters.loc[dict(response_func=scaling_name, target=target_name, param='y_pr2', lat=lat_val, lon=lon_val)] = params_scaled.y_pr2

                successful_optimizations += 1

        # Newline after each response function completes its latitude bands
        print()

    return {
        'total_grid_cells': total_grid_cells,
        'successful_optimizations': successful_optimizations
    }


def calculate_reference_climate_baselines(all_data, config):
    """
    Calculate reference climate baselines (tas0, pr0) as 2D arrays for all grid cells.

    This computes the reference period mean temperature and precipitation for each grid cell,
    following the same approach as the per-grid-cell optimization but computed once for reuse.

    Parameters
    ----------
    all_data : dict
        Complete data structure containing all SSP climate data
    config : dict
        Configuration dictionary containing time_periods and ssp_scenarios

    Returns
    -------
    tas0_2d, pr0_2d : np.ndarray
        Reference baselines as 2D arrays [lat, lon]
    """
    # Extract data from all_data structure
    # Get reference SSP from config
    reference_ssp = config['ssp_scenarios']['reference_ssp']

    tas_data = get_ssp_data(all_data, reference_ssp, 'tas')
    pr_data = get_ssp_data(all_data, reference_ssp, 'pr')

    # Get reference period
    time_periods = config['time_periods']
    ref_start_year = time_periods['reference_period']['start_year']
    ref_end_year = time_periods['reference_period']['end_year']

    # Calculate reference period means using coordinate-based selection
    tas0_2d = tas_data.sel(time=slice(ref_start_year, ref_end_year)).mean(dim='time')
    pr0_2d = pr_data.sel(time=slice(ref_start_year, ref_end_year)).mean(dim='time')

    return tas0_2d, pr0_2d


def calculate_weather_gdp_regression_slopes(
    all_data, config, response_scalings, reference_tfp, scaling_results
):
    """
    Calculate regression slopes of GDP variability vs temperature variability over historical period.

    This analysis runs forward economic projections for each response function and computes
    the regression slope of (GDP_ratio ~ tas_weather) over the historical period, where:
    - GDP_ratio = actual_GDP / LOESS_smoothed_GDP_trend
    - tas_weather = weather component of temperature (LOESS-filtered)

    Parameters
    ----------
    all_data : Dict[str, Any]
        Pre-loaded data containing climate and economic variables
    config : Dict[str, Any]
        Configuration dictionary with time periods and model parameters
    response_scalings : List[Dict]
        Response function scaling configurations
    reference_tfp : Dict[str, np.ndarray]
        Baseline TFP results from Step 2
    scaling_results : Dict[str, Any]
        Scaling factor results from Step 3

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing regression slopes for each response function and target:
        - slopes[response_name][target_name]: 2D array [lat, lon] of regression slopes
        - success_mask[response_name][target_name]: 2D boolean array of successful regressions
        - gdp_weighted_means[response_name][target_name]: GDP-weighted mean slope
    """
    print("\n=== Weather-GDP Regression Analysis ===")
    print("Computing regression slopes: GDP_variability ~ temperature_variability")

    # Extract time period indices
    hist_period = config['time_periods']['historical_period']
    hist_start_year = hist_period['start_year']
    hist_end_year = hist_period['end_year']

    # Extract data arrays using get_ssp_data helper
    reference_ssp = config['ssp_scenarios']['reference_ssp']

    # Get data using helper function
    tas_data = get_ssp_data(all_data, reference_ssp, 'tas')
    pr_data = get_ssp_data(all_data, reference_ssp, 'pr')
    gdp_data = get_ssp_data(all_data, reference_ssp, 'gdp')
    pop_data = get_ssp_data(all_data, reference_ssp, 'pop')

    # Get dimensions
    nlat, nlon = len(tas_data.lat), len(tas_data.lon)
    valid_mask = all_data['_metadata']['valid_mask']

    # Extract weather data (already computed in all_data)
    tas_weather_data = all_data[reference_ssp]['tas_weather']
    pr_weather_data = all_data[reference_ssp]['pr_weather']

    # Get reference climate baselines
    tas0_2d = all_data['tas0_2d']
    pr0_2d = all_data['pr0_2d']

    # Initialize results storage
    regression_results = {
        'slopes': {},
        'success_mask': {},
        'gdp_weighted_means': {}
    }

    # Get target names and response function names
    target_names = [target['target_name'] for target in config['gdp_targets']]
    response_names = [scaling['scaling_name'] for scaling in response_scalings]

    # Process each response function
    for response_idx, response_config in enumerate(response_scalings):
        response_name = response_config['scaling_name']
        print(f"Processing response function: {response_name} ({response_idx+1}/{len(response_scalings)})")

        regression_results['slopes'][response_name] = {}
        regression_results['success_mask'][response_name] = {}
        regression_results['gdp_weighted_means'][response_name] = {}

        # Process each target
        for target_idx, target_name in enumerate(target_names):
            print(f"  Target: {target_name}")

            # Initialize arrays for this response-target combination as xarray
            lat_coords = valid_mask.coords['lat'].values
            lon_coords = valid_mask.coords['lon'].values

            regression_slopes = xr.DataArray(
                np.zeros((nlat, nlon)),
                coords={'lat': lat_coords, 'lon': lon_coords},
                dims=['lat', 'lon']
            )
            regression_success_mask = xr.DataArray(
                np.zeros((nlat, nlon), dtype=bool),
                coords={'lat': lat_coords, 'lon': lon_coords},
                dims=['lat', 'lon']
            )
            error_count = 0  # Track errors for debugging

            # Extract scaling factors for this response-target combination
            scaling_factors = scaling_results['scaling_factors']  # xarray DataArray

            # Process each grid cell
            successful_regressions = 0
            for lat_val in lat_coords:
                for lon_val in lon_coords:
                    if not valid_mask.sel(lat=lat_val, lon=lon_val):
                        continue

                    scaling_factor = float(scaling_factors.sel(response_func=response_name, target=target_name, lat=lat_val, lon=lon_val))
                    if not np.isfinite(scaling_factor):
                        continue

                    # Create scaled parameters for this grid cell
                    # Start with base parameters from config and scale the response function parameters
                    base_params = config['model_params']
                    scaled_params = ModelParams(
                        s=base_params['s'],
                        alpha=base_params['alpha'],
                        delta=base_params['delta'],
                        tas0=float(tas0_2d.sel(lat=lat_val, lon=lon_val)),
                        pr0=float(pr0_2d.sel(lat=lat_val, lon=lon_val)),
                        k_tas1=float(response_config.get('k_tas1', 0.0) * scaling_factor),
                        k_tas2=float(response_config.get('k_tas2', 0.0) * scaling_factor),
                        k_pr1=float(response_config.get('k_pr1', 0.0) * scaling_factor),
                        k_pr2=float(response_config.get('k_pr2', 0.0) * scaling_factor),
                        tfp_tas1=float(response_config.get('tfp_tas1', 0.0) * scaling_factor),
                        tfp_tas2=float(response_config.get('tfp_tas2', 0.0) * scaling_factor),
                        tfp_pr1=float(response_config.get('tfp_pr1', 0.0) * scaling_factor),
                        tfp_pr2=float(response_config.get('tfp_pr2', 0.0) * scaling_factor),
                        y_tas1=float(response_config.get('y_tas1', 0.0) * scaling_factor),
                        y_tas2=float(response_config.get('y_tas2', 0.0) * scaling_factor),
                        y_pr1=float(response_config.get('y_pr1', 0.0) * scaling_factor),
                        y_pr2=float(response_config.get('y_pr2', 0.0) * scaling_factor)
                    )

                    # Extract baseline TFP for this cell
                    cell_tfp_baseline = reference_tfp[reference_ssp]['tfp_baseline'].sel(lat=lat_val, lon=lon_val).values
                    cell_pop = pop_data.sel(lat=lat_val, lon=lon_val).values
                    cell_tas = tas_data.sel(lat=lat_val, lon=lon_val).values
                    cell_pr = pr_data.sel(lat=lat_val, lon=lon_val).values

                    # Run forward model to get GDP projections
                    try:
                        forward_results = calculate_coin_ssp_forward_model(
                            cell_tfp_baseline, cell_pop, cell_tas, cell_pr, scaled_params
                        )
                        gdp_forward, _, _, _, _, _ = forward_results

                        # Extract historical period data using coordinate-based selection
                        tas_weather_hist = tas_weather_data.sel(
                            time=slice(hist_start_year, hist_end_year), lat=lat_val, lon=lon_val
                        ).values

                        # gdp_forward is already numpy array from forward model
                        # Select historical period
                        years = all_data['years']
                        hist_mask = (years >= hist_start_year) & (years <= hist_end_year)
                        gdp_forward_hist = gdp_forward[hist_mask]

                        # Compute GDP ratio: actual GDP / LOESS smoothed trend
                        gdp_ratio = apply_loess_divide(gdp_forward_hist, 30)

                        # Remove invalid values
                        valid_data_mask = np.isfinite(gdp_ratio) & np.isfinite(tas_weather_hist)

                        if np.sum(valid_data_mask) < 10:  # Need sufficient data points
                            continue

                        gdp_ratio_valid = gdp_ratio[valid_data_mask]
                        tas_weather_valid = tas_weather_hist[valid_data_mask]

                        # Linear regression: GDP_ratio ~ tas_weather
                        if np.std(tas_weather_valid) > 1e-6:  # Check for sufficient variation
                            slope, intercept = np.polyfit(tas_weather_valid, gdp_ratio_valid, 1)
                            regression_slopes.loc[dict(lat=lat_val, lon=lon_val)] = slope
                            regression_success_mask.loc[dict(lat=lat_val, lon=lon_val)] = True
                            successful_regressions += 1

                            # Debug first successful regression
                            if successful_regressions == 1:
                                print(f"      First successful regression at lat={lat_val}, lon={lon_val}:")
                                print(f"        Slope: {slope:.6f}")
                                print(f"        TAS weather std: {np.std(tas_weather_valid):.6f}")
                                print(f"        GDP ratio std: {np.std(gdp_ratio_valid):.6f}")

                    except Exception as e:
                        # Skip cells where forward model fails
                        if error_count < 5:  # Print first 5 errors for debugging
                            print(f"      Error in cell [lat={lat_val}, lon={lon_val}]: {type(e).__name__}: {e}")
                            error_count += 1
                        continue

            print(f"    Successful regressions: {successful_regressions}/{np.sum(valid_mask)}")

            # Calculate GDP-weighted mean slope
            if successful_regressions > 0:
                # Convert xarray DataArrays to numpy for boolean indexing
                gdp_weights = gdp_data.sel(time=slice(hist_start_year, hist_end_year)).mean(dim='time').values
                regression_slopes_values = regression_slopes.values
                regression_success_mask_values = regression_success_mask.values
                valid_mask_values = valid_mask.values

                weighted_slopes = regression_slopes_values * regression_success_mask_values * gdp_weights
                total_weight = np.sum(gdp_weights[valid_mask_values] * regression_success_mask_values[valid_mask_values])
                gdp_weighted_mean = np.sum(weighted_slopes[valid_mask_values]) / total_weight if total_weight > 0 else 0.0

                # Debug GDP-weighted mean calculation
                print(f"    GDP-weighted mean slope: {gdp_weighted_mean:.6f}")
                print(f"    Slope range: {np.min(regression_slopes_values[regression_success_mask_values]):.6f} to {np.max(regression_slopes_values[regression_success_mask_values]):.6f}")
            else:
                gdp_weighted_mean = 0.0

            # Store results
            regression_results['slopes'][response_name][target_name] = regression_slopes
            regression_results['success_mask'][response_name][target_name] = regression_success_mask
            regression_results['gdp_weighted_means'][response_name][target_name] = gdp_weighted_mean

    print(f"Weather-GDP regression analysis complete")
    print(f"Regression results created for {len(regression_results['gdp_weighted_means'])} response functions:")
    for resp_name, targets in regression_results['gdp_weighted_means'].items():
        print(f"  {resp_name}: {len(targets)} targets")
    return regression_results


def calculate_variability_climate_response_parameters(
    all_data, config, reference_tfp, response_scalings
):
    """
    Calculate climate response parameters for variability targets using a 4-step calibration process.

    This function determines the climate response parameters that will be used for variability
    scaling by establishing the relationship between weather variability and economic impacts.

    ALGORITHM OVERVIEW:
    ==================

    Step 1: OPTIMIZATION FOR UNIFORM 10% GDP LOSS
    ---------------------------------------------
    - Run optimization to find scaling factors that produce uniform 10% GDP loss in target period
    - This establishes the baseline strength of climate-economy relationship needed for target impact
    - Uses dummy target with 10% constant reduction across all grid cells
    - Outputs: scaling factors for each response function parameter

    Step 2: FORWARD MODEL SIMULATIONS WITH SCALED PARAMETERS
    -------------------------------------------------------
    - Take parameters from Step 1 optimization, scaled by the found factors
    - Run forward model simulations using WEATHER COMPONENTS (tas_weather, pr_weather)
    - This isolates weather variability effects from long-term climate trends
    - Generate economic projections over the full time period (historical + future)
    - Outputs: time series of economic variables (GDP, capital, TFP) for each grid cell

    Step 3: WEATHER VARIABILITY REGRESSION ANALYSIS
    -----------------------------------------------
    - For each grid cell, compute regression: (GDP / LOESS_smoothed_GDP) ~ tas_weather over historical period
    - GDP = actual GDP from forward model simulation
    - LOESS_smoothed_GDP = 30-year LOESS smoothed trend of GDP
    - tas_weather = weather component of temperature (detrended, LOESS-filtered climate signal)
    - Regression slope = fractional change in GDP per degree C of weather variability
    - This quantifies the actual historical relationship between weather and economic fluctuations

    Step 4: PARAMETER NORMALIZATION BY REGRESSION SLOPE
    --------------------------------------------------
    - Divide all climate response parameters from Phase 1 by the regression slope from Phase 3
    - This normalizes parameters so they represent the correct strength per degree of variability
    - Final parameters capture both the target impact magnitude AND the observed weather sensitivity
    - Result: climate parameters calibrated for variability target applications

    Parameters
    ----------
    all_data : dict
        Complete data structure containing all SSP climate and economic data
    config : dict
        Configuration dictionary containing ssp_scenarios and time_periods
    reference_tfp : dict
        TFP reference data containing valid_mask and tfp_baseline
    response_scalings : list
        List of damage scaling configurations (for optimization)

    Returns
    -------
    baseline_climate_parameters : np.ndarray
        Climate response parameters [lat, lon, n_params] calibrated for variability targets.
        Parameters are normalized by weather-GDP regression slopes from historical data.
    """

    print("Computing climate response parameters for variability targets...")
    print("Using 4-step calibration process:")
    print("  1. Optimization for uniform 10% GDP loss")
    print("  2. Forward model simulations with scaled parameters")
    print("  3. Weather variability regression analysis")
    print("  4. Parameter normalization by regression slope")

    # =================================================================================
    # STEP 1: OPTIMIZATION FOR UNIFORM 10% GDP LOSS
    # =================================================================================
    print("\n--- Step 1: Optimization for uniform 10% GDP loss ---")

    # Extract data from all_data structure
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    tas_data = get_ssp_data(all_data, reference_ssp, 'tas')
    pr_data = get_ssp_data(all_data, reference_ssp, 'pr')
    pop_data = get_ssp_data(all_data, reference_ssp, 'pop')
    gdp_data = get_ssp_data(all_data, reference_ssp, 'gdp')
    tas_weather_data = all_data[reference_ssp]['tas_weather']
    pr_weather_data = all_data[reference_ssp]['pr_weather']
    years = all_data['years']
    tas0_2d = all_data['tas0_2d']
    pr0_2d = all_data['pr0_2d']

    # Extract TFP data
    valid_mask = all_data['_metadata']['valid_mask']
    tfp_baseline = reference_tfp['tfp_baseline']  # reference_tfp is already SSP-specific
    nlat = len(valid_mask.coords['lat'])
    nlon = len(valid_mask.coords['lon'])

    # Create dummy GDP target for optimization (uniform 10% loss)
    dummy_gdp_target = {
        'target_type': 'damage',
        'target_shape': 'constant',
        'global_mean_amount': -0.10,
        'target_name': 'variability_reference'
    }

    # Create constant response as xarray DataArray
    constant_response = xr.DataArray(
        np.full((nlat, nlon), -0.10),
        coords={'lat': valid_mask.coords['lat'], 'lon': valid_mask.coords['lon']},
        dims=['lat', 'lon']
    )
    dummy_target_results = {
        'variability_reference': {
            'reduction_array': constant_response
        }
    }

    # Initialize arrays for optimization
    n_response_functions = len(response_scalings)
    n_targets = 1
    n_params = 12

    # Extract coordinate lists
    lat_coords = valid_mask.coords['lat']
    lon_coords = valid_mask.coords['lon']
    response_func_names = [rs['scaling_name'] for rs in response_scalings]
    target_names_list = ['variability_reference']
    param_names = ['y_tas1', 'y_tas2', 'y_pr1', 'y_pr2', 'k_tas1', 'k_tas2', 'k_pr1', 'k_pr2',
                   'tfp_tas1', 'tfp_tas2', 'tfp_pr1', 'tfp_pr2']

    # Create xarray DataArrays with named dimensions
    scaling_factors = xr.DataArray(
        np.zeros((n_response_functions, n_targets, nlat, nlon)),
        coords={'response_func': response_func_names, 'target': target_names_list,
                'lat': lat_coords, 'lon': lon_coords},
        dims=['response_func', 'target', 'lat', 'lon']
    )
    optimization_errors = xr.DataArray(
        np.zeros((n_response_functions, n_targets, nlat, nlon)),
        coords={'response_func': response_func_names, 'target': target_names_list,
                'lat': lat_coords, 'lon': lon_coords},
        dims=['response_func', 'target', 'lat', 'lon']
    )
    convergence_flags = xr.DataArray(
        np.zeros((n_response_functions, n_targets, nlat, nlon), dtype=bool),
        coords={'response_func': response_func_names, 'target': target_names_list,
                'lat': lat_coords, 'lon': lon_coords},
        dims=['response_func', 'target', 'lat', 'lon']
    )
    scaled_parameters = xr.DataArray(
        np.zeros((n_response_functions, n_targets, n_params, nlat, nlon)),
        coords={'response_func': response_func_names, 'target': target_names_list,
                'param': param_names, 'lat': lat_coords, 'lon': lon_coords},
        dims=['response_func', 'target', 'param', 'lat', 'lon']
    )

    # Run optimization to find scaling factors for uniform 10% GDP loss
    # Uses full climate data (tas_data, pr_data) for optimization
    reference_results = process_response_target_optimization(
        0, dummy_gdp_target, dummy_target_results, response_scalings,
        tas_data, pr_data, pop_data, gdp_data, reference_tfp, valid_mask, tfp_baseline,
        years, config, scaling_factors, optimization_errors, convergence_flags,
        scaled_parameters, 0, 0, tas_weather_data, pr_weather_data
    )

    # Extract optimized parameters (Step 1 results) for ALL response functions
    # Store parameters for each response function as [lat, lon, param] xarray DataArrays
    all_step1_parameters = {}
    for response_config in response_scalings:
        response_name = response_config['scaling_name']
        # Select and transpose: [param, lat, lon] -> [lat, lon, param]
        params_for_response = scaled_parameters.sel(response_func=response_name, target='variability_reference')
        all_step1_parameters[response_name] = params_for_response.transpose('lat', 'lon', 'param')

    valid_cells = np.sum(valid_mask)
    print(f"Phase 1 complete: {valid_cells}/{valid_cells} valid grid cells for {n_response_functions} response functions")

    # Get time period (needed for all response functions)
    time_periods = config['time_periods']
    hist_start_year = time_periods['historical_period']['start_year']
    hist_end_year = time_periods['historical_period']['end_year']

    # =================================================================================
    # PHASES 2-4: PROCESS EACH RESPONSE FUNCTION SEPARATELY
    # =================================================================================
    # Store results for each response function
    all_regression_slopes = {}
    all_regression_success_masks = {}
    all_final_parameters = {}
    all_final_success_masks = {}

    for resp_idx, response_config in enumerate(response_scalings):
        response_name = response_config['scaling_name']
        print(f"\n{'='*80}")
        print(f"Processing response function: {response_name} ({resp_idx+1}/{n_response_functions})")
        print(f"{'='*80}")

        step1_parameters = all_step1_parameters[response_name]

        # =================================================================================
        # PHASE 2: FORWARD MODEL SIMULATIONS WITH SCALED PARAMETERS
        # =================================================================================
        print(f"\n--- Phase 2: Forward model simulations for {response_name} ---")

        # Initialize arrays for forward model results
        gdp_forward = np.zeros((len(years), nlat, nlon))

        print(f"Running forward model for {np.sum(valid_mask)} grid cells...")

        # Run forward model for each valid grid cell using coordinate-based indexing
        lat_coords = valid_mask.coords['lat']
        lon_coords = valid_mask.coords['lon']

        for lat_idx, lat_val in enumerate(lat_coords):
            for lon_idx, lon_val in enumerate(lon_coords):
                if not valid_mask.sel(lat=lat_val, lon=lon_val):
                    continue

                # Extract model parameters for this cell from Step 1 using coordinate selection
                cell_params = step1_parameters.sel(lat=lat_val, lon=lon_val)

                # Create ModelParams object with optimized climate response parameters
                # Convert xarray scalars to float
                model_params = ModelParams(
                    s=config['model_params']['s'],
                    alpha=config['model_params']['alpha'],
                    delta=config['model_params']['delta'],
                    tas0=float(tas0_2d.sel(lat=lat_val, lon=lon_val)),
                    pr0=float(pr0_2d.sel(lat=lat_val, lon=lon_val)),
                    k_tas1=float(cell_params.sel(param='k_tas1')),
                    k_tas2=float(cell_params.sel(param='k_tas2')),
                    k_pr1=float(cell_params.sel(param='k_pr1')),
                    k_pr2=float(cell_params.sel(param='k_pr2')),
                    tfp_tas1=float(cell_params.sel(param='tfp_tas1')),
                    tfp_tas2=float(cell_params.sel(param='tfp_tas2')),
                    tfp_pr1=float(cell_params.sel(param='tfp_pr1')),
                    tfp_pr2=float(cell_params.sel(param='tfp_pr2')),
                    y_tas1=float(cell_params.sel(param='y_tas1')),
                    y_tas2=float(cell_params.sel(param='y_tas2')),
                    y_pr1=float(cell_params.sel(param='y_pr1')),
                    y_pr2=float(cell_params.sel(param='y_pr2'))
                )

                # Extract time series for this grid cell using coordinate selection
                # Use WEATHER COMPONENTS for forward simulation to isolate variability effects
                cell_tas_weather = tas_weather_data.sel(lat=lat_val, lon=lon_val).values
                cell_pr_weather = pr_weather_data.sel(lat=lat_val, lon=lon_val).values
                cell_pop = pop_data.sel(lat=lat_val, lon=lon_val).values
                cell_tfp_baseline = tfp_baseline.sel(lat=lat_val, lon=lon_val).values

                # Run forward model with weather components
                y_forward, _, _, _, _, _ = calculate_coin_ssp_forward_model(
                    cell_tfp_baseline, cell_pop, cell_tas_weather, cell_pr_weather, model_params
                )

                gdp_forward[:, lat_idx, lon_idx] = y_forward

        print(f"Phase 2 complete: Forward model simulations generated for {response_name}")

        # =================================================================================
        # PHASE 3: WEATHER VARIABILITY REGRESSION ANALYSIS
        # =================================================================================
        print(f"\n--- Phase 3: Weather variability regression analysis for {response_name} ---")

        # Initialize regression slope array
        regression_slopes = np.zeros((nlat, nlon))
        regression_success_mask = np.zeros((nlat, nlon), dtype=bool)

        print(f"Computing y_weather ~ tas_weather regression for {np.sum(valid_mask)} cells...")

        for lat_idx, lat_val in enumerate(lat_coords):
            for lon_idx, lon_val in enumerate(lon_coords):
                if not valid_mask.sel(lat=lat_val, lon=lon_val):
                    continue

                # Extract weather variables for historical period
                tas_weather_hist = tas_weather_data.sel(time=slice(hist_start_year, hist_end_year), lat=lat_val, lon=lon_val).values
                # gdp_forward is numpy array, use boolean mask for historical period
                hist_mask = (years >= hist_start_year) & (years <= hist_end_year)
                gdp_forward_hist = gdp_forward[hist_mask, lat_idx, lon_idx]

                # Compute GDP ratio: actual GDP divided by 30-year LOESS smoothed GDP trend
                # Use the new apply_loess_divide function for clean, mnemonic operation
                gdp_ratio = apply_loess_divide(gdp_forward_hist, 30)

                # Remove any invalid values
                valid_data_mask = np.isfinite(gdp_ratio) & np.isfinite(tas_weather_hist)

                # Compute regression: (GDP / LOESS_smoothed_GDP) ~ tas_weather
                gdp_ratio_valid = gdp_ratio[valid_data_mask]
                tas_weather_valid = tas_weather_hist[valid_data_mask]

                # Linear regression
                if np.std(tas_weather_valid) > 1e-6:  # Check for sufficient variation
                    slope, _ = np.polyfit(tas_weather_valid, gdp_ratio_valid, 1)
                    regression_slopes[lat_idx, lon_idx] = slope
                    regression_success_mask[lat_idx, lon_idx] = True

        regression_success = np.sum(regression_success_mask)
        print(f"Phase 3 complete: {regression_success} successful regressions for {response_name}")

        # =================================================================================
        # PHASE 4: PARAMETER NORMALIZATION BY REGRESSION SLOPE
        # =================================================================================
        print(f"\n--- Phase 4: Parameter normalization for {response_name} ---")

        # Initialize final normalized parameters as xarray DataArray with same structure as step1_parameters
        final_parameters = step1_parameters.copy()
        # Convert regression masks to xarray for proper broadcasting
        regression_success_xr = xr.DataArray(
            regression_success_mask,
            coords={'lat': lat_coords, 'lon': lon_coords},
            dims=['lat', 'lon']
        )
        final_success_mask = valid_mask & regression_success_xr

        for lat_idx, lat_val in enumerate(lat_coords):
            for lon_idx, lon_val in enumerate(lon_coords):
                if not final_success_mask.sel(lat=lat_val, lon=lon_val):
                    continue

                slope = regression_slopes[lat_idx, lon_idx]

                # Divide all parameters by regression slope
                final_parameters.loc[dict(lat=lat_val, lon=lon_val)] = step1_parameters.sel(lat=lat_val, lon=lon_val) / slope

        final_success = np.sum(final_success_mask)
        print(f"Phase 4 complete: {final_success} final calibrated parameters for {response_name}")

        # Store results for this response function
        all_regression_slopes[response_name] = regression_slopes
        all_regression_success_masks[response_name] = regression_success_mask
        all_final_parameters[response_name] = final_parameters
        all_final_success_masks[response_name] = final_success_mask

    print(f"\n4-phase calibration summary:")
    print(f"  Valid grid cells: {valid_cells}")
    print(f"  Response functions processed: {n_response_functions}")
    for response_name in all_regression_slopes.keys():
        resp_success = np.sum(all_regression_success_masks[response_name])
        print(f"    {response_name}: {resp_success} successful regressions")

    # Return comprehensive results dictionary with per-response-function data
    return {
        'all_final_parameters': all_final_parameters,  # dict[response_name] -> [nlat, nlon, n_params]
        'all_step1_parameters': all_step1_parameters,  # dict[response_name] -> [nlat, nlon, n_params]
        'all_regression_slopes': all_regression_slopes,  # dict[response_name] -> [nlat, nlon]
        'all_regression_success_masks': all_regression_success_masks,  # dict[response_name] -> [nlat, nlon]
        'all_final_success_masks': all_final_success_masks,  # dict[response_name] -> [nlat, nlon]
        'response_function_names': [rf['scaling_name'] for rf in response_scalings],
        'valid_cells': valid_cells
    }


def calculate_variability_scaling_parameters(
    variability_calibration_results, gdp_target, target_idx,
    all_data, config, response_scalings,
    scaling_factors, optimization_errors, convergence_flags, scaled_parameters
):
    """
    Calculate GDP variability scaling parameters (g0, g1, g2) for variability targets.

    Computes the g0, g1, g2 parameters that define how GDP climate sensitivity varies with
    local temperature according to the target shape (constant, linear, quadratic).
    These parameters are used in the forward model as: g(T) = g0 + g1*T + g2*T²

    For different target shapes:
    - constant: g0 = target_amount, g1 = 0, g2 = 0
    - linear: g0, g1 computed from global_mean_amount and zero_amount_temperature
    - quadratic: g0, g1, g2 computed from global_mean_amount, zero_amount_temperature, etc.

    Parameters
    ----------
    variability_calibration_results : dict
        Results from calculate_variability_climate_response_parameters containing 'all_final_parameters' dict
    gdp_target : dict
        Target configuration with variability parameters and target_shape
    target_idx : int
        Target index for result storage
    all_data : dict
        Complete data structure containing all SSP climate and economic data
    config : dict
        Configuration dictionary containing time periods and SSP scenarios
    response_scalings : list
        Damage scaling configurations
    scaling_factors, optimization_errors, convergence_flags, scaled_parameters : np.ndarray
        Output arrays to populate (modified in place)

    Returns
    -------
    dict
        Dictionary containing:
        - 'g0_array': np.ndarray [lat, lon] - g0 values for each grid cell
        - 'g1_array': np.ndarray [lat, lon] - g1 values for each grid cell
        - 'g2_array': np.ndarray [lat, lon] - g2 values for each grid cell
        - 'total_grid_cells': int - number of processed grid cells
        - 'successful_optimizations': int - number of successful calculations
    """
    # Extract data from all_data and config
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    time_periods = config['time_periods']

    # Get climate and economic data
    tas_data = get_ssp_data(all_data, reference_ssp, 'tas')
    gdp_data = get_ssp_data(all_data, reference_ssp, 'gdp')
    years = all_data['years']
    tas0_2d = all_data['tas0_2d']

    # Get historical period for GDP weighting
    hist_start_year = time_periods['historical_period']['start_year']
    hist_end_year = time_periods['historical_period']['end_year']

    nlat, nlon = tas0_2d.shape
    n_response_functions = len(response_scalings)
    valid_mask = all_data['_metadata']['valid_mask']

    target_name = gdp_target['target_name']
    target_shape = gdp_target['target_shape']

    print(f"\nProcessing GDP target: {target_name} ({target_idx+1}/?) - Shape: {target_shape}")

    # Step 1: Calculate mean GDP by grid cell over historical period (only for valid cells)
    mean_gdp_per_cell = gdp_data.sel(time=slice(hist_start_year, hist_end_year)).mean(dim='time').values  # [lat, lon]

    # Calculate GDP variability scaling parameters (g0, g1, g2) based on target shape
    if target_shape == 'constant':
        # For constant targets: g(T) = g0, g1 = 0, g2 = 0
        g0 = gdp_target.get('global_mean_amount', 1.0)
        g1 = 0.0
        g2 = 0.0
        print(f"  Constant GDP variability scaling: g0={g0}")

    elif target_shape == 'linear':
        # Linear case: g(T) = a0 + a1 * T, so g0 = a0, g1 = a1, g2 = 0
        # Constraints: g(T_zero) = 0 and mean(g(T)) = global_mean_amount
        global_mean_amount = gdp_target['global_mean_amount']
        zero_amount_temperature = gdp_target['zero_amount_temperature']

        print(f"  Linear target: global_mean={global_mean_amount}, zero_temp={zero_amount_temperature}")

        # Calculate GDP-weighted mean temperature using only valid economic cells
        total_gdp = np.sum(mean_gdp_per_cell[valid_mask.values])
        tas0_2d_values = tas0_2d.values
        gdp_weighted_tas = np.sum(mean_gdp_per_cell[valid_mask.values] * tas0_2d_values[valid_mask.values]) / total_gdp

        # Calculate coefficients for linear relationship: g(T) = a0 + a1 * T
        # From constraints: g(T_zero) = 0 => a0 + a1*T_zero = 0 => a0 = -a1*T_zero
        # mean(g(T)) = global_mean => a0 + a1*mean(T) = global_mean
        # Substituting: -a1*T_zero + a1*mean(T) = global_mean
        # => a1*(mean(T) - T_zero) = global_mean
        g1 = global_mean_amount / (gdp_weighted_tas - zero_amount_temperature)
        g0 = -g1 * zero_amount_temperature
        g2 = 0.0

        print(f"  Linear coefficients: g0={g0:.6f}, g1={g1:.6f}")
        print(f"  GDP-weighted mean temperature: {gdp_weighted_tas:.2f}°C")

    elif target_shape == 'quadratic':
        # Quadratic case: g(T) = a0 + a1*T + a2*T^2, so g0 = a0, g1 = a1, g2 = a2
        global_mean_amount = gdp_target['global_mean_amount']
        zero_amount_temperature = gdp_target['zero_amount_temperature']
        zero_derivative_temperature = gdp_target['zero_derivative_temperature']

        print(f"  Quadratic target: global_mean={global_mean_amount}, zero_temp={zero_amount_temperature}, deriv_at_zero={zero_derivative_temperature}")

        # Calculate GDP-weighted mean temperature using only valid economic cells
        total_gdp = np.sum(mean_gdp_per_cell[valid_mask.values])
        tas0_2d_values = tas0_2d.values
        gdp_weighted_tas = np.sum(mean_gdp_per_cell[valid_mask.values] * tas0_2d_values[valid_mask.values]) / total_gdp
        gdp_weighted_tas2 = np.sum(mean_gdp_per_cell[valid_mask.values] * tas0_2d_values[valid_mask.values]**2) / total_gdp

        T0 = zero_amount_temperature
        T_mean = gdp_weighted_tas
        T2_mean = gdp_weighted_tas2

        # From constraints:
        # a1 + 2*a2*T0 = zero_derivative_temperature
        # a0 + a1*T0 + a2*T0^2 = 0
        # a0 + a1*T_mean + a2*T2_mean = global_mean_amount

        # Solve the system: eliminate a0 and solve for a1, a2
        # From first two: a0 = -a1*T0 - a2*T0^2
        # Substitute into third: -a1*T0 - a2*T0^2 + a1*T_mean + a2*T2_mean = global_mean_amount
        # a1*(T_mean - T0) + a2*(T2_mean - T0^2) = global_mean_amount
        # a1 + 2*a2*T0 = zero_derivative_temperature

        # Matrix form: [T_mean-T0, T2_mean-T0^2] [a1] = [global_mean_amount]
        #              [1,         2*T0         ] [a2]   [zero_derivative_temperature]

        det = (T_mean - T0) * 2 * T0 - (T2_mean - T0**2) * 1
        g1 = (global_mean_amount * 2 * T0 - zero_derivative_temperature * (T2_mean - T0**2)) / det
        g2 = (zero_derivative_temperature * (T_mean - T0) - global_mean_amount * 1) / det
        g0 = -g1 * T0 - g2 * T0**2

        print(f"  Quadratic coefficients: g0={g0:.6f}, g1={g1:.6f}, g2={g2:.6f}")
        print(f"  GDP-weighted mean temperature: {gdp_weighted_tas:.2f}°C")

    else:
        raise ValueError(f"Unknown target_shape: {target_shape}")

    # Compute GDP variability scaling factors at reference temperature for each grid cell
    target_scaling_factors_array = g0 + g1 * tas0_2d + g2 * tas0_2d**2

    print(f"  Variability scaling at reference temperature range: {np.nanmin(target_scaling_factors_array):.6f} to {np.nanmax(target_scaling_factors_array):.6f}")

    # Initialize counters to match process_response_target_optimization return structure
    total_grid_cells = 0
    successful_optimizations = 0

    # Extract all_final_parameters dict from variability calibration results
    all_baseline_parameters = variability_calibration_results['all_final_parameters']

    # Loop over response functions to match process_response_target_optimization structure
    for response_idx, response_scaling in enumerate(response_scalings):
        scaling_name = response_scaling['scaling_name']
        print(f"  Response function: {scaling_name} ({response_idx+1}/{n_response_functions})")

        # Get response-function-specific baseline parameters (keep as xarray)
        baseline_climate_parameters = all_baseline_parameters[scaling_name]

        # Get coordinate values for iteration
        lat_coords = valid_mask.coords['lat']
        lon_coords = valid_mask.coords['lon']

        # Progress indicator and counting: iterate over coordinates
        for lat_idx, lat_val in enumerate(lat_coords):
            print(".", end="", flush=True)

            for lon_idx, lon_val in enumerate(lon_coords):
                # Count valid cells where we have finite scaling factors
                scaling_val = target_scaling_factors_array.sel(lat=lat_val, lon=lon_val)
                baseline_val = baseline_climate_parameters.sel(lat=lat_val, lon=lon_val, param='y_tas1')
                if np.isfinite(scaling_val) and np.isfinite(baseline_val):
                    total_grid_cells += 1
                    successful_optimizations += 1

        # Calculate scaling factors by applying target scaling to reference scaled parameters
        # scaling_factor = target_scaling_factor (no optimization, direct application)
        scaling_factors[response_idx, target_idx, :, :] = target_scaling_factors_array.values

        # For variability targets, set optimization error to zero (no optimization performed)
        optimization_errors[response_idx, target_idx, :, :] = 0.0

        # Mark as converged where we have valid baseline parameters and finite scaling factors
        convergence_flags[response_idx, target_idx, :, :] = (
            np.isfinite(baseline_climate_parameters.sel(param='y_tas1').values) &
            np.isfinite(target_scaling_factors_array.values)
        )

        # Store scaled parameters by applying target scaling to baseline parameters
        # For variability targets, we scale the baseline parameters by the target scaling factor
        for param_name in baseline_climate_parameters.coords['param'].values:
            param_idx = list(baseline_climate_parameters.coords['param'].values).index(param_name)
            scaled_parameters[response_idx, target_idx, param_idx, :, :] = (
                baseline_climate_parameters.sel(param=param_name).values * target_scaling_factors_array.values
            )

        # Newline after each response function completes its latitude bands
        print()

    # Summary statistics
    valid_cells = np.sum(np.isfinite(baseline_climate_parameters.sel(param='y_tas1').values))
    applied_cells = np.sum(np.isfinite(target_scaling_factors_array.values))

    print(f"  Applied to {applied_cells}/{valid_cells} valid grid cells")
    if applied_cells > 0:
        # Convert to numpy for boolean indexing
        scaling_values = target_scaling_factors_array.values if hasattr(target_scaling_factors_array, 'values') else target_scaling_factors_array
        scaling_range = scaling_values[np.isfinite(scaling_values)]
        print(f"  Scaling factors range: {np.min(scaling_range):.6f} to {np.max(scaling_range):.6f}")

    return {
        'g0': g0,
        'g1': g1,
        'g2': g2,
        'total_grid_cells': total_grid_cells,
        'successful_optimizations': successful_optimizations
    }