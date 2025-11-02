# COIN_SSP: Gridded Climate-Economic Impact Model

A spatially-explicit implementation of the Solow-Swan growth model for assessing climate impacts on economic growth at the grid cell level.

## Table of Contents

1. [Quick Start with Concrete Examples](#quick-start-with-concrete-examples)
2. [Overview](#overview)
3. [Complete Methods Documentation](#complete-methods-documentation)
   - [Overview](#1-overview)
   - [The COIN-SSP Model](#2-the-coin-ssp-model)
   - [Historical Calibration](#3-historical-calibration-of-reference-case)
   - [SSP Calibration](#4-calibration-of-reference-case-total-factor-productivity-for-ssp-scenarios)
   - [Combined Historical and SSP Reference Cases](#5-combined-historical-and-ssp-reference-cases)
   - [Climate Response Functions](#6-climate-response-functions)
   - [Model Calibration](#7-model-calibration)
   - [Cases Considered](#8-cases-considered)
   - [Data Sources](#9-data-sources)
   - [References](#references)
4. [Function Calling Tree](#function-calling-tree)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Output Structure](#output-structure)
8. [Key Features](#key-features)
9. [Documentation](#documentation)
10. [Next Steps](#next-steps)
11. [Contributing](#contributing)
12. [License](#license)
13. [Citation](#citation)

---

## Quick Start with Concrete Examples

### Linear Response Functions (Ignoring Quadratic Terms)

For a complete analysis using only linear temperature response parameters (y_tas1, k_tas1, tfp_tas1):

```bash
python workflow_manager.py \
  coin_ssp_config_linear_parameter_sensitivity.json \
  coin_ssp_config_response_functions_linear_template.json
```

**What this does:**
- **Stage 1**: Runs individual parameter sensitivity for y_tas1, k_tas1, and tfp_tas1
- **Stage 2**: Analyzes GDP-weighted results and generates multi-variable response functions
- **Stage 3**: Executes combined simulations with parameter combinations like:
  - `output_linear` (y_tas1 only)
  - `capital_linear` (k_tas1 only)
  - `tfp_linear` (tfp_tas1 only)
  - `y+k_linear` (y_tas1 + k_tas1)
  - `y+tfp_linear` (y_tas1 + tfp_tas1)
  - `y+k+tfp_linear` (all three combined)

**Typical runtime**: ~15-25 minutes (depends on grid size and number of targets)

**Output location**: `data/output/output_CanESM5_<timestamp>/`

### Full Response Functions (Including Quadratic Terms)

For a comprehensive analysis including both linear and quadratic temperature and precipitation effects:

```bash
python workflow_manager.py \
  coin_ssp_config_parameter_sensitivity.json \
  coin_ssp_config_response_functions_template.json
```

**What this does:**
- **Stage 1**: Runs individual parameter sensitivity for all 12 climate response parameters:
  - Temperature: y_tas1, y_tas2, k_tas1, k_tas2, tfp_tas1, tfp_tas2
  - Precipitation: y_pr1, y_pr2, k_pr1, k_pr2, tfp_pr1, tfp_pr2
- **Stage 2**: Generates multi-variable configurations combining optimal parameter ratios
- **Stage 3**: Executes final projections with complex response function combinations

**Typical runtime**: ~30-45 minutes (processes 12 individual parameters in Stage 1)

**Output location**: `data/output/output_CanESM5_<timestamp>/`

### Single Configuration Run (Advanced)

To run a specific configuration directly without the multi-stage workflow:

```bash
# Run full 5-step integrated workflow
python main.py coin_ssp_config_0008.json

# Skip Step 3 optimization using pre-computed results (faster for testing visualizations)
python main.py coin_ssp_config_0008.json --step3-file data/output/previous_run/step3_results.nc
```

### Workflow Manager Advanced Options

```bash
# Start from Stage 2 (if Stage 1 already completed)
python workflow_manager.py \
  config_sensitivity.json config_template.json \
  --start-stage 2 \
  --stage1-output ./data/output/output_CanESM5_20251101_174657/

# Start from Stage 3 (if Stage 2 config already generated)
python workflow_manager.py \
  config_sensitivity.json config_template.json \
  --start-stage 3 \
  --stage2-config ./configs/coin_ssp_config_stage2_generated_20251101_174657.json
```

---

## Overview

COIN_SSP processes gridded NetCDF climate and economic data to quantify how climate change affects economic growth through response functions applied to capital stock, productivity, and output.

### Core Model
- **Economic Framework**: Solow-Swan growth model with DICE-derived parameters (Barrage & Nordhaus 2024)
- **Climate Integration**: Temperature and precipitation response functions (linear/quadratic)
- **Variability Scaling**: Temperature-dependent climate sensitivity with g(T) = g0 + g1*T + g2*T²
- **Spatial Processing**: Grid cell-level optimization and forward modeling
- **Scenario Support**: Multiple SSP economic scenarios and climate projections

---

## Complete Methods Documentation

## 1. Overview

We developed a variant of a Solow-Swann growth model of an economy, with the capability of responding to temperature and precipitation changes. We call this model COIN-SSP.

We tuned this model so that it could approximate both historical GDP growth and GDP growth as represented in the Shared Socioeconomic Pathways. We then specify various known climate response functions and simulate both the historical and future economic growth both with and without climate change, taking these known climate response functions into account.

Various econometric methods have been applied to GDP and weather variability in the real world, and used to project the economic response. In further work, we will apply some of these methods to the COIN-SSP results for the historical period, and test their predictive skill at estimating the climate response in the target period (typically, years 2080 to 2100).

The implementation includes a multi-stage workflow system (`workflow_manager.py`) that systematically applies the approach described in Section 7.4 for developing multi-variable response functions through: (1) individual parameter assessment, (2) GDP-weighted analysis of results, and (3) generation and execution of combined response function configurations.

## 2. The COIN-SSP Model

The core of the COIN-SSP model is a variant Solow-Swann growth model with a Cobb-Douglas production function, with flexible climate response specification that can be applied to output, capital stock, and/or the growth rate in total factor productivity.

The core model, in the absence of climate or weather effects, can be described by:

```
Y(t) = A(t) K(t)^α L(t)^(1-α)     (1)
```

and

```
dK(t)/dt = sY(t) - δK(t)     (2)
```

For capital elasticity in the production function (α) and the depreciation rate of capital (δ), we use values from Barrage and Nordhaus (2024), namely α=0.3 and δ=0.1 yr⁻¹. We use a savings rate of s = α = 0.3.

## 3. Historical Calibration of Reference Case

The actual GDP record contains the influence of weather variability and climate change. We do not want to introduce this unknown climate signal into our model calibration. Therefore, we generate a stylized proxy for economic growth in each model grid cell with no influence of climate or weather.

We calibrate the model by assuming that in our reference case that capital stock (K) grows at a constant exponential rate, and that Y(t) is equal to the historical GDP values both in the first year that data is available (t_init) and in year 2015 (t_2015):

```
K_hist(t) = K_hist(t_init) e^(k_K(t-t_init))     (3)
```

Taking the derivative of equation (3) and substituting into equation (2), and solving for k_K and K_hist(t_init), we have:

```
k_K = Log(GDP_2015/GDP_init)/(2015-t_init)     (4)
```

and

```
K_hist(t_init) = s/(δ+k_K) GDP_init     (5)
```

Because we do not want to have actual historical weather influencing our calculation, we further assume that Y(t) increases exponentially from the initial value of GDP_init to the year 2015 value. Following the logic above, this means:

```
Y_hist(t) = Y_hist(t_init) e^(k_K(t-t_init))     (6)
```

From equation (1), we then have as total factor productivity in the historical reference case:

```
A_hist(t) = Y_hist(t) / (K_hist(t)^α L_hist(t)^(1-α))     (7)
```

## 4. Calibration of Reference Case Total Factor Productivity for SSP Scenarios

If we assume at the beginning of the SSP scenario:

```
K_SSP(t_2015) = K_hist(t_2015)     (8)
```

and consider that Y_SSP(t) is provided by the SSP scenario, we can evolve K_SSP(t) through time using equation (2).

For the SSP scenarios, considering equation (1), we then have:

```
A_SSP(t) = Y_SSP(t) / (K_SSP(t)^α L_SSP(t)^(1-α))     (9)
```

## 5. Combined Historical and SSP Reference Cases

We then have for our reference cases:

```
A_ref(t) = A_hist(t) for t < 2015     (10a)
```

and

```
A_ref(t) = A_SSP(t) for t ≥ 2015     (10b)
```

Trivially,

```
K_ref(t_init) = K_hist(t_init)     (11)
```

We then use these reference values to drive our model taking into consideration weather variability and climate change, as described below.

## 6. Climate Response Functions

We consider climate response functions that involve functions of temperature and precipitation. (In practice, in the simulations described here, all of the precipitation-related variables have been set to zero.)

We consider cases in which climate can damage output, capital stock and/or the growth rate in total factor productivity.

For any experimental case, exp, where exp might be a climate-change case, or a weather-only case, we have, for each climate model grid cell, climate response factors that modify the economic components directly.

The climate response factors are calculated as:

```
y_climate = 1.0 + g(T) * f_y(T, P) - g(T_ref) * f_y(T_ref, P_ref)     (12)
```

```
k_climate = 1.0 + g(T) * f_k(T, P) - g(T_ref) * f_k(T_ref, P_ref)     (13)
```

```
tfp_climate = 1.0 + g(T) * f_tfp(T, P) - g(T_ref) * f_tfp(T_ref, P_ref)     (14)
```

where the climate response functions f_y, f_k, and f_tfp are defined as:

```
f_y(T, P) = (y_tas1 * T + y_tas2 * T²) + (y_pr1 * P + y_pr2 * P²)     (15)
```

```
f_k(T, P) = (k_tas1 * T + k_tas2 * T²) + (k_pr1 * P + k_pr2 * P²)     (16)
```

```
f_tfp(T, P) = (tfp_tas1 * T + tfp_tas2 * T²) + (tfp_pr1 * P + tfp_pr2 * P²)     (17)
```

where g(T) is a scaling function:

```
g(T(t)) = g0 + g1*T(t) + g2*T(t)²     (18)
```

This user-defined climate response scaling factor function, g(T), has units of fraction of output per degree Celsius. It indicates as a function of temperature the desired slope of an ordinary least squares fit of Y_weather as a function of T_weather; this slope is the correlation coefficient times the ratio of the standard deviations. The parameters g0, g1 and g2 may be chosen to examine various cases of interest. Positive values of g(T) would indicate a positive correlation between Y_weather and T_weather (i.e., climate benefit) and negative values would indicate a negative correlation (i.e., climate losses).

**Note**: For "damage" target simulations, g(T) = 1.0 (i.e., g0 = 1.0, g1 = 0.0, g2 = 0.0), meaning the climate response functions are applied uniformly without temperature-dependent scaling. The g(T) scaling is only used for "variability" target simulations where the goal is to scale historical GDP-weather variability relationships as a function of temperature.

The climate response parameters in the implementation are:
- **Output responses**: y_tas1, y_tas2 (linear and quadratic temperature), y_pr1, y_pr2 (linear and quadratic precipitation)
- **Capital responses**: k_tas1, k_tas2 (linear and quadratic temperature), k_pr1, k_pr2 (linear and quadratic precipitation)
- **TFP responses**: tfp_tas1, tfp_tas2 (linear and quadratic temperature), tfp_pr1, tfp_pr2 (linear and quadratic precipitation)

## 7. Model Calibration

### 7.1 Conceptual Framework

Conceptually, to examine various combinations of climate response pathways, we introduce parameters indicating the relative values of the coefficients for the output, capital stock, and total-factor productivity growth climate response functions: r_Y, r_K and r_A.

To calibrate our model, for each combination of r_Y, r_K and r_A considered, at each grid cell, we determine the value of f0, such that when the climate response parameters are scaled appropriately, the model produces the desired economic impact targets.

This procedure assures that the relationships between output variability and temperature variability are similar across cases considering different climate response pathways.

### 7.2 Implementation: Damage Target Calibration

In practice, for damage targets, this is implemented through an optimization process that finds a scale factor α for each grid cell such that the ratio of climate-affected GDP to weather-only GDP in the target period equals (1 + target_reduction).

The optimization process:
1. For each response function scaling configuration, determines which climate parameters are non-zero
2. Uses scipy.optimize.minimize to find the optimal scale factor
3. Applies the scale factor to create scaled model parameters
4. Runs forward model simulations with both climate and weather-only forcing
5. Computes the objective function as the squared difference between the achieved and target GDP ratios

### 7.3 Implementation: Variability Target Calibration

For variability targets, the calibration uses a sophisticated four-step process:

**Step 1: Optimization for Uniform 10% GDP Loss**
- Run optimization to find scaling factors that produce uniform 10% GDP loss in target period
- Establishes baseline strength of climate-economy relationship needed for target impact

**Step 2: Forward Model Simulations with Scaled Parameters**
- Take parameters from Step 1, scaled by found factors
- Run forward model simulations for each grid cell using scaled parameters
- Generate economic projections over full time period (historical + future)

**Step 3: Weather Variability Regression Analysis**
- For each grid cell: compute regression `log(y_weather) ~ tas_weather` over historical period
- `y_weather` = weather component of GDP (detrended, LOESS-filtered economic signal)
- `tas_weather` = weather component of temperature (detrended, LOESS-filtered climate signal)
- Regression slope = fractional change in GDP per degree C of weather variability

**Step 4: Parameter Normalization by Regression Slope**
- Divide all climate response parameters from Phase 1 by regression slope from Phase 3
- Normalizes parameters to represent correct strength per degree of variability
- Final parameters capture both target impact magnitude AND observed weather sensitivity

### 7.4 Multi-Stage Workflow for Developing Multi-Variable Response Functions

The implementation includes a systematic three-stage workflow for developing response functions involving multiple climate-economy pathways. This approach allows for principled combination of individual parameter effects while maintaining target impact levels.

#### Stage 1: Individual Parameter Assessment

All initial optimizations operate on single parameters for each grid cell, determining the scaling factor required to achieve a specified climate response (typically 10% GDP reduction in the target period).

For each of the 12 core climate response parameters (y_tas1, y_tas2, k_tas1, k_tas2, tfp_tas1, tfp_tas2, y_pr1, y_pr2, k_pr1, k_pr2, tfp_pr1, tfp_pr2), we run individual optimization simulations to find the parameter value that produces the target economic impact in each grid cell.

The Stage 1 results provide GDP-weighted global median values for each parameter, representing the typical parameter magnitude needed to generate the target climate damage. For "damage" simulations we use the target period for GDP weighting; for "variability" simulations we use the historical period for GDP weightings.

#### Stage 2: Multi-Variable Response Function Generation

Stage 2 combines individual parameter baselines from Stage 1 with user-specified response function templates to generate multi-variable parameter combinations.

The algorithm operates as follows:

1. **Load Stage 1 baselines**: Extract GDP-weighted median scaling factors for each of the 12 parameters from Stage 1 results
2. **Apply template ratios**: For each response function specification in `coin_ssp_config_response_functions_template.json`, multiply the template ratio by the corresponding Stage 1 baseline and convert sign convention:
   ```
   raw_value[i] = template_ratio[i] × stage1_baseline[i] × (-1)
   ```
   The sign conversion transforms negative scaling factors from Stage 1 optimization into positive parameter values representing the magnitude of climate response for intuitive interpretation. A positive climate response indicates that GDP increases with temperature.
3. **Normalize by sum**: Divide each raw value by the sum of all non-zero raw values to maintain target impact magnitude:
   ```
   final_value[i] = raw_value[i] / sum(raw_values)
   ```

This normalization ensures that multi-variable response functions produce approximately the same target economic impact (10% GDP reduction) while respecting the relative importance ratios specified in the template.

#### Stage 3: Multi-Variable Simulations

Stage 3 executes the complete COIN-SSP pipeline using the multi-variable response functions generated in Stage 2, producing final economic projections that incorporate the combined effects of multiple climate-economy pathways.

#### Template-Based Response Function Design

The template file `coin_ssp_config_response_functions_template.json` allows researchers to specify relative importance ratios between different climate response pathways. For example:

- **Output-focused**: `y_tas1: 2.0, k_tas1: 1.0, tfp_tas1: 0.0` emphasizes temperature effects on output over capital or TFP
- **Balanced linear**: `y_tas1: 1.0, k_tas1: 1.0, tfp_tas1: 1.0` gives equal weight to all linear temperature effects
- **Comprehensive**: Non-zero values across multiple pathways and polynomial terms for complex response functions

This systematic approach enables exploration of different hypotheses about climate-economy interaction mechanisms while maintaining consistent impact magnitudes for comparative analysis.

As described below, for each model grid cell, we have data specifying the temperature in the historical cases (T_hist) and for each of the Shared Socioeconomic Pathway cases (T_SSP).

For each specification of climate response parameters (y_tas1, y_tas2, k_tas1, k_tas2, tfp_tas1, tfp_tas2, etc.) and T_ref, we then perform a simulation of the historical case using the forward model equations, yielding a time series of Y_hist for each model grid point. This is the data used to train the various econometric methods.

## 8. Cases Considered

### 8.1 Consideration of Different Relationships Between Output Variability and Temperature Change

We choose three target patterns for the slope of the linear regression of Y_weather against T_weather.

**Constant sensitivity case:** we consider g0 = -0.01 °C⁻¹, g1 = 0 °C⁻², and g2 = 0 °C⁻³, indicating a constant slope of 1% per degree Celsius.

**Linear sensitivity case:** we consider a case where there is a slope of zero at 10°C and 2% at 20°C, which yields g0 = 0.02 °C⁻¹, g1 = -0.002 °C⁻², and g2 = 0 °C⁻³.

**Quadratic sensitivity case:** we consider a case where there is a slope of zero at 10°C, 2% at 20°C, and a slope of 0.001 °C⁻¹ which yields g0 = 0 °C⁻¹, g1 = 0.001 °C⁻², and g2 = -0.0001 °C⁻³.

We consider three patterns of target losses at each grid cell, each designed to predict a global average GDP loss of about 10% in the year 2080 to 2100 time interval under SSP2-4.5.

**Uniform targets:** each grid cell loses 10% of its output

**Linear targets:** GDP losses are chosen such that, under SSP2-4.5, GDP losses scale linearly with T_ref and are 25% for grid cells with T_ref = 30°C, with the global mean GDP loss at 10%.

**Quadratic targets:** GDP losses are chosen such that, under SSP2-4.5, GDP losses scale quadratically with T_ref and are 75% for grid cells with T_ref = 30°C, no net-gains or losses for grid cells with T_ref = 13.5°C, and a global mean GDP loss at 10%.

### 8.2 Climate Response Function Cases

For each SSP scenario and climate model considered, we define response function scaling configurations that specify which climate response parameters are non-zero. Each scaling configuration represents a different pathway through which climate affects the economy.

The response function scalings are defined in the model configuration and typically include:

- **Output linear temperature**: y_tas1 = 1.0 (all others = 0)
- **Output quadratic temperature**: y_tas2 = 1.0 (all others = 0)
- **Capital linear temperature**: k_tas1 = 1.0 (all others = 0)
- **Capital quadratic temperature**: k_tas2 = 1.0 (all others = 0)
- **TFP linear temperature**: tfp_tas1 = 1.0 (all others = 0)
- **TFP quadratic temperature**: tfp_tas2 = 1.0 (all others = 0)

In this way, we consider independently linear and quadratic influences of temperature on output, capital stock, and the growth rate in total factor productivity.

For each grid cell and each SSP/climate-model combination, we calculate the scale factor for each response function configuration that produces the target economic impact, and we then use those scaled parameters in our simulations.

Additional scaling configurations may combine multiple response pathways, where the values from individual cases are combined and then scaled by a common factor to produce the target impact.

The number of response function cases combined with the target climate-response patterns provides multiple scenarios for each SSP/climate-model combination.

## 9. Data Sources

The COIN-SSP model requires several gridded datasets that are specified through configuration files and loaded from NetCDF format files.

### 9.1 Gridded Historical Population
Historical population density data is provided at grid cell resolution, typically covering the period from 1861 to 2014. The data is stored in NetCDF files with the variable name "pop_density" (or as specified in the configuration). Population data is used to calculate the labor input L(t) in the production function and is normalized to the initial year value for each grid cell.

### 9.2 Gridded Historical GDP
Historical GDP density data is provided at matching grid cell resolution for the same time period as population. The data is stored in NetCDF files with the variable name "gdp_density" (or as specified in the configuration). GDP data serves as the economic output Y(t) that the model calibrates to reproduce in its reference case simulations.

### 9.3 Gridded SSP Population
Future population projections under different Shared Socioeconomic Pathway (SSP) scenarios, typically covering 2015-2100. These projections are provided for multiple SSP scenarios (e.g., SSP2-4.5, SSP5-8.5) and are used to drive the forward model simulations for future economic projections.

### 9.4 Gridded SSP GDP
Future GDP projections under different SSP scenarios, matching the temporal and spatial resolution of the SSP population data. These provide the target economic trajectories that the model reproduces in its reference case before applying climate response functions.

### 9.5 Gridded Climate Data
Climate data includes both temperature and precipitation:

**Temperature (tas)**: Surface air temperature data in Kelvin or Celsius, provided for both historical and future periods under different SSP scenarios. Temperature anomalies relative to a reference period (typically 1861-1910) are used to drive climate response functions.

**Precipitation (pr)**: Precipitation data in appropriate units (kg m⁻² s⁻¹ or mm/day), provided for matching periods. While precipitation parameters are included in the model structure, they are typically set to zero in current simulations as noted in Section 6.

Climate data is obtained from climate model output (e.g., CanESM5) and provides the forcing for the climate response functions in the economic model.

### 9.6 Data Processing Requirements
All datasets must:
- Use consistent grid resolution and spatial coverage
- Follow standardized NetCDF conventions with time, latitude, and longitude dimensions
- Include appropriate metadata and coordinate information
- Be accessible through the file naming convention specified in the model configuration (e.g., CLIMATE_CanESM5_historical.nc, GDP_CanESM5_ssp245.nc)

## References

Barrage L, Nordhaus W. 2024. Policies, projections, and the social cost of carbon: results from the DICE-2023 model. PNAS 121:(13):e2312030121

---

## Function Calling Tree

The COIN_SSP pipeline follows a structured calling hierarchy organized into five main processing steps. Understanding this structure is essential for navigating the codebase.

### Pipeline Entry Points

**`main.py`** - Single configuration execution
- `run_pipeline()` → Main execution orchestrator that coordinates all 5 processing steps

**`workflow_manager.py`** - Multi-stage workflow orchestration
- `WorkflowManager` → Three-stage pipeline manager for parameter sensitivity analysis
  - `run_stage1()` → Individual response function assessments (6-12 separate runs)
  - `analyze_stage1_results()` → Extract GDP-weighted parameter means from Stage 1 CSV outputs
  - `generate_stage2_config()` → Create multi-variable configuration combining best parameters
  - `run_stage3()` → Execute final simulations with combined response functions

### Data Loading and Preprocessing

Called at pipeline initialization before Step 1:

**`load_all_data()`** → **[coin_ssp_utils.py]**
- Loads and concatenates all NetCDF input files (climate, GDP, population)
- Returns unified `all_data` dictionary used throughout pipeline
- Sub-functions:
  - `load_and_concatenate_climate_data()` → Loads temperature/precipitation from historical + SSP files
  - `load_and_concatenate_pop_data()` → Loads population from historical + SSP files
  - `load_and_concatenate_gdp_data()` → Loads GDP density from SSP-specific files
  - `resolve_netcdf_filepath()` → Resolves file paths using configured prefixes
  - `get_grid_metadata()` → Extracts spatial/temporal coordinates from NetCDF files

**`calculate_weather_vars()`** → **[coin_ssp_utils.py]**
- Computes weather variability components (LOESS-filtered climate signals)
- Separates short-term variability from long-term climate trends
- Sub-functions:
  - `apply_time_series_filter()` → **[coin_ssp_math_utils.py]** LOESS filtering with 30-year window
  - `calculate_area_weights()` → **[coin_ssp_math_utils.py]** Cosine-latitude weighting

### Step 1: Target GDP Changes

**`step1_calculate_target_gdp_changes()`** → **[main.py]**
- Calculates spatial patterns of target GDP reductions that optimization will try to achieve
- Supports multiple target types: constant, linear, quadratic temperature relationships
- Called once per pipeline run using reference SSP

Key functions:
- `calculate_all_target_reductions()` → **[coin_ssp_target_calculations.py]**
  - Computes target reduction patterns for all configured GDP targets
  - Sub-functions:
    - `calculate_constant_target_reduction()` → Uniform spatial targets (e.g., -5% everywhere)
    - `calculate_linear_target_reduction()` → Temperature-dependent linear targets
    - `calculate_quadratic_target_reduction()` → Temperature-dependent quadratic targets
- `save_step1_results_netcdf()` → **[coin_ssp_netcdf.py]**
  - Writes target patterns to NetCDF with full metadata
- `create_target_gdp_visualization()` → **[coin_ssp_reporting.py]**
  - Generates PDF maps showing spatial distribution of targets

**Outputs:**
- `step1_{json_id}_{model}_{ssp}_target_gdp.nc` (~220 KB)
- `step1_{json_id}_{model}_{ssp}_target_gdp_visualization.pdf` (~125 KB)

### Step 2: Baseline TFP

**`step2_calculate_baseline_tfp()`** → **[main.py]**
- Calculates baseline economic variables (TFP, capital) without any climate effects
- Provides counterfactual "what growth would be without climate change"
- Called once per SSP scenario (typically 1-2 SSPs)

Key functions:
- `calculate_tfp_coin_ssp()` → **[coin_ssp_core.py]**
  - Runs Solow-Swan growth model with zero climate response parameters
  - Sub-functions:
    - `calculate_coin_ssp_forward_model()` → Core economic model integration
      - Solves differential equations for capital accumulation
      - Computes TFP from GDP, capital, and population
      - Returns time series of economic variables
- `save_step2_results_netcdf()` → **[coin_ssp_netcdf.py]**
  - Writes baseline TFP and capital to NetCDF
- `create_baseline_tfp_visualization()` → **[coin_ssp_reporting.py]**
  - Generates PDF time series plots and percentile analysis

**Outputs:**
- `step2_{json_id}_{model}_{ssp}_baseline_tfp.nc` (~30 MB per SSP)
- `step2_{json_id}_{model}_baseline_tfp_visualization.pdf` (~470 KB)

### Step 3: Scaling Factor Optimization

**`step3_calculate_scaling_factors_per_cell()`** → **[main.py]**
- Most computationally expensive step (~5-10 minutes for full grid)
- Optimizes scaling factors for each grid cell to match target GDP patterns
- Processes all combinations of response functions × GDP targets
- Uses reference SSP only for calibration

Key functions:

**For damage-type targets:**
- `process_response_target_optimization()` → **[coin_ssp_core.py]**
  - Loops over all grid cells running optimization for each
  - Sub-functions:
    - `optimize_climate_response_scaling()` → Per-grid-cell constrained optimization
      - Uses scipy.optimize.minimize with constraint satisfaction
      - Objective: minimize squared error between simulated and target GDP
      - Calls `calculate_coin_ssp_forward_model()` repeatedly during optimization
      - Returns scaling factor that best matches target for this grid cell

**For variability-type targets:**
- `calculate_variability_climate_response_parameters()` → **[coin_ssp_core.py]** (NEW December 2025)
  - 4-step calibration process for variability targets:
    1. **Phase 1**: Optimize for uniform 10% GDP loss (establishes baseline strength)
    2. **Phase 2**: Run forward model with weather components to isolate variability effects
    3. **Phase 3**: Compute regression slopes (GDP_weather ~ TAS_weather) over historical period
    4. **Phase 4**: Normalize parameters by regression slope to match observed sensitivity
  - Returns calibrated parameters for all response functions
  - Outputs per-response-function regression slope statistics to CSV

- `calculate_variability_scaling_parameters()` → **[coin_ssp_core.py]**
  - Applies variability parameters with target-specific scaling

**Analysis and reporting:**
- `calculate_weather_gdp_regression_slopes()` → **[coin_ssp_core.py]**
  - Analyzes historical weather-GDP relationships for all response functions
  - Computes regression slopes for each grid cell
  - Returns GDP-weighted statistics
- `save_step3_results_netcdf()` → **[coin_ssp_netcdf.py]**
  - Writes scaling factors, parameters, convergence flags to NetCDF
- `create_scaling_factors_visualization()` → **[coin_ssp_reporting.py]**
  - Generates PDF maps of optimized scaling factors
- `create_objective_function_visualization()` → **[coin_ssp_reporting.py]**
  - Generates PDF maps of optimization errors
- `create_regression_slopes_visualization()` → **[coin_ssp_reporting.py]**
  - Generates PDF maps of weather-GDP regression slopes
- `print_gdp_weighted_scaling_summary()` → **[coin_ssp_reporting.py]**
  - Computes and writes GDP-weighted statistics to CSV
- `write_variability_calibration_summary()` → **[coin_ssp_reporting.py]**
  - Writes per-response-function variability calibration results to CSV

**Outputs:**
- `step3_{json_id}_{model}_{ssp}_scaling_factors.nc` (~3-6 MB)
- `step3_{json_id}_{model}_{ssp}_scaling_factors_summary.csv` (~1-2 KB)
- `step3_{json_id}_{model}_{ssp}_variability_calibration_summary.csv` (~1 KB, if variability targets exist)
- `step3_{json_id}_{model}_{ssp}_scaling_factors_visualization.pdf` (~700 KB)
- `step3_{json_id}_{model}_{ssp}_objective_function_visualization.pdf` (~700 KB)
- `step3_{json_id}_{model}_{ssp}_regression_slopes_visualization.pdf` (~460 KB)

### Step 4: Forward Integration

**`step4_forward_integration_all_ssps()`** → **[main.py]**
- Runs forward model for all configured SSP scenarios using calibrated parameters
- Generates climate-integrated economic projections (GDP, capital, TFP)
- Only executed if `forward_simulation_ssps` specified in configuration

Key functions:
- `calculate_coin_ssp_forward_model()` → **[coin_ssp_core.py]**
  - Core Solow-Swan model with climate response functions applied
  - Called for every grid cell × response function × target × SSP combination
  - Uses calibrated scaling factors from Step 3
- `save_step4_results_netcdf_split()` → **[coin_ssp_netcdf.py]**
  - Writes separate NetCDF files for each SSP and variable type
  - Files: `step4_{json_id}_{model}_{ssp}_forward_{gdp|tfp|capital}.nc`
- `create_forward_model_visualization()` → **[coin_ssp_reporting.py]**
  - Generates time series line plots comparing scenarios
- `create_forward_model_maps_visualization()` → **[coin_ssp_reporting.py]**
  - Generates spatial impact maps (both linear and log10 scales)
- `create_forward_model_ratio_visualization()` → **[coin_ssp_reporting.py]**
  - Generates ratio maps (climate/weather effects)

**Outputs:**
- `step4_{json_id}_{model}_{ssp}_forward_{variable}.nc` (~65-70 MB per SSP×variable)
- `step4_{json_id}_{model}_forward_model_lineplots.pdf` (~55-125 KB)
- `step4_{json_id}_{model}_forward_model_maps.pdf` (~900 KB - 2.6 MB)
- `step4_{json_id}_{model}_forward_model_maps_log10.pdf` (~900 KB - 2.7 MB)
- `step4_{json_id}_{model}_forward_model_ratios.pdf` (~70-180 KB)

### Step 5: Processing Summary

**`step5_processing_summary()`** → **[main.py]**
- Prints final statistics and timing information
- Currently minimal - placeholder for future summary analysis

### Utility Functions

**Mathematical Operations** → **[coin_ssp_math_utils.py]**
- `calculate_global_mean()` → Area-weighted spatial averages with masking
- `calculate_area_weights()` → Cosine-latitude area weighting
- `calculate_zero_biased_range()` → Visualization range calculation (extends to include zero)
- `calculate_time_means()` → Temporal averaging over specified periods
- `apply_loess_subtract()` → Degree-2 LOESS smoothing with trend subtraction and reference period mean addition
- `apply_loess_divide()` → Degree-2 LOESS smoothing applied to log-transformed data (difference of logs for quasi-exponential series like GDP)

**Data I/O Operations** → **[coin_ssp_netcdf.py]**
- `create_serializable_config()` → Converts config dict to JSON-safe format for NetCDF attributes
- `extract_year_coordinate()` → Extracts time coordinates from NetCDF files
- `interpolate_to_annual_grid()` → Temporal interpolation to annual resolution
- `resolve_netcdf_filepath()` → Constructs file paths using configured prefixes and naming conventions

**Visualization Utilities** → **[coin_ssp_reporting.py]**
- `get_adaptive_subplot_layout()` → Calculates optimal subplot arrangement based on number of targets
- `add_extremes_info_box()` → Adds min/max value boxes to map visualizations
- All visualization functions use consistent styling and adaptive layouts

---

## Installation

```bash
git clone https://github.com/KCaldeira/coin_ssp.git
cd coin_ssp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration

Configuration files use JSON format with several key sections:

### Key Configuration Sections

- **`run_metadata`**: Identifies the configuration run with `json_id`, `run_name`, and `description`
- **`climate_model`**: NetCDF file patterns and variable names for climate data
- **`ssp_scenarios`**: Reference SSP for calibration and forward simulation SSPs
- **`time_periods`**: Reference, historical, target, and prediction periods
- **`gdp_targets`**: Economic impact targets with `target_type` ("damage" or "variability")
- **`model_params`**: Solow-Swan model parameters (s, alpha, delta, climate response parameters)
- **`response_function_scalings`**: Climate response function configurations specifying active parameters

See example configuration files in the `configs/` directory for detailed examples.

---

## Output Structure

Results are organized in timestamped directories:

```
data/output/output_{model}_{timestamp}/{json_id}_{timestamp}/
├── coin_ssp_config_{json_id}.json (configuration copy)
├── all_loaded_data_{json_id}_{model}.nc (all input data)
├── step1_{json_id}_{model}_{ssp}_target_gdp.* (target patterns)
├── step2_{json_id}_{model}_{ssp}_baseline_tfp.* (baseline economics)
├── step3_{json_id}_{model}_{ssp}_scaling_factors.* (optimization results)
├── step3_{json_id}_{model}_{ssp}_*_summary.csv (GDP-weighted statistics)
└── step4_{json_id}_{model}_{ssp}_forward_*.* (climate projections, if configured)
```

---

## Key Features

- **Production Ready**: Complete 5-step processing pipeline
- **Adaptive Optimization**: Automatic bounds expansion when hitting limits
- **Variability Calibration**: 4-step algorithm for variability-type targets
- **Weather Analysis**: Pre-computed LOESS-filtered climate variability
- **Fail-Fast Design**: Clean error handling without defensive programming
- **Comprehensive Visualization**: Multi-page PDFs with adaptive layouts

---

## Documentation

- **`README.md`**: This file - complete methods documentation and usage guide
- **`CLAUDE.md`**: Code style guide and architecture decisions

---

## Next Steps

### Immediate Priorities

1. **Complete xarray DataArray Migration**
   - Finish removing all legacy code that used to work with numpy arrays
   - Convert remaining numpy array initializations to xarray DataArrays (e.g., regression slopes in coin_ssp_netcdf.py)
   - All arrays with time, lat, or lon dimensions MUST be xarray DataArrays with properly labeled coordinates
   - Remove any remaining numpy fallback branches

2. **Step 1 Mean Reduction Verification**
   - Investigate why mean reductions printed on Step 1 reports are not consistent with values requested in config JSON file
   - Verify target calculation functions are correctly implementing requested reduction percentages
   - Check GDP-weighted mean calculations for accuracy

---

## Contributing

This project follows elegant, fail-fast coding principles:
- No input validation on function parameters
- Let exceptions bubble up naturally
- Prefer mathematical clarity over defensive checks
- Use numpy vectorization instead of loops

See `CLAUDE.md` for detailed code style requirements.

---

## License

MIT License - See LICENSE file for details.

---

## Citation

If you use COIN_SSP in your research, please cite:
```
COIN_SSP: A spatially-explicit climate-economic impact model implementing
the Solow-Swan growth model with gridded climate response functions.
```
