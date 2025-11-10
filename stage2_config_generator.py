#!/usr/bin/env python3
"""
Stage 2: Analyze Stage 1 results and generate Stage 3 configuration

This script takes Stage 1 parameter sensitivity results and a template configuration
to generate a new configuration file for Stage 3 multi-variable simulations.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List


def load_stage1_results(stage1_output_dir: str) -> Dict[str, float]:
    """
    Load GDP-weighted parameter medians from Stage 1 CSV results.

    Parameters
    ----------
    stage1_output_dir : str
        Directory containing Stage 1 output files

    Returns
    -------
    Dict[str, float]
        GDP-weighted medians for each of the 12 scaling parameters
    """
    # Find the scaling factors summary CSV file
    csv_files = list(Path(stage1_output_dir).glob("step3_*_scaling_factors_summary.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No scaling factors CSV found in {stage1_output_dir}")

    csv_path = csv_files[0]  # Use the first/only CSV file
    print(f"Loading Stage 1 results from: {csv_path}")

    # Load the CSV file
    df = pd.read_csv(csv_path)

    param_columns = ['y_tas1', 'y_tas2', 'k_tas1', 'k_tas2', 'tfp_tas1', 'tfp_tas2',
                     'y_pr1', 'y_pr2', 'k_pr1', 'k_pr2', 'tfp_pr1', 'tfp_pr2']

    # Use the GDP-weighted median values that are already computed in the CSV
    # The CSV contains one row per response function, and we want the parameter value
    # from the row where that parameter is non-zero (the individual parameter assessment)

    gdp_weighted_medians = {}
    for param in param_columns:
        if param in df.columns:
            # Find the row where this parameter is non-zero (individual assessment)
            param_row = df[df[param] != 0.0]
            if len(param_row) > 0:
                # Use the GDP-weighted median from the individual parameter assessment
                gdp_weighted_median = param_row.iloc[0]['gdp_weighted_median']
                gdp_weighted_medians[param] = gdp_weighted_median
                print(f"  {param}: {gdp_weighted_median:.6f}")
            else:
                gdp_weighted_medians[param] = 0.0
        else:
            gdp_weighted_medians[param] = 0.0

    return gdp_weighted_medians


def load_template_config(template_path: str) -> Dict[str, Any]:
    """
    Load template configuration with ratio specifications.

    Parameters
    ----------
    template_path : str
        Path to template JSON configuration file

    Returns
    -------
    Dict[str, Any]
        Template configuration dictionary
    """
    with open(template_path, 'r') as f:
        template = json.load(f)

    print(f"Loaded template configuration from: {template_path}")
    print(f"  Found {len(template.get('response_function_scalings', []))} response function templates")

    return template


def generate_stage3_config(stage1_medians: Dict[str, float],
                          template_config: Dict[str, Any],
                          base_config_path: str) -> Dict[str, Any]:
    """
    Generate Stage 3 configuration by applying template ratios to Stage 1 baseline values.

    Parameters
    ----------
    stage1_medians : Dict[str, float]
        GDP-weighted medians from Stage 1 for each parameter
    template_config : Dict[str, Any]
        Template configuration with ratio specifications
    base_config_path : str
        Path to base configuration to copy other settings from

    Returns
    -------
    Dict[str, Any]
        Complete Stage 3 configuration
    """
    # Load base configuration for non-scaling settings
    with open(base_config_path, 'r') as f:
        stage3_config = json.load(f)

    # Update metadata
    stage3_config['run_metadata']['json_id'] = 'stage3-multi-variable'
    stage3_config['run_metadata']['run_name'] = 'Stage3_Multi_Variable_Simulations'
    stage3_config['run_metadata']['description'] = 'Multi-variable response functions based on Stage 1 analysis'

    # Generate new response function scalings based on template ratios
    new_scalings = []

    for template_scaling in template_config['response_function_scalings']:
        new_scaling = {
            'scaling_name': template_scaling['scaling_name'],
            'description': template_scaling['description']
        }

        # Apply template ratios to Stage 1 baseline values and normalize
        param_columns = ['y_tas1', 'y_tas2', 'k_tas1', 'k_tas2', 'tfp_tas1', 'tfp_tas2',
                         'y_pr1', 'y_pr2', 'k_pr1', 'k_pr2', 'tfp_pr1', 'tfp_pr2']

        # Step 1: Calculate raw values = template_ratio × stage1_baseline
        # Note: Multiply by -1 to convert from negative scaling factors to positive parameters
        # representing the magnitude of climate response (positive = GDP increases with temperature)
        raw_values = {}
        for param in param_columns:
            template_ratio = template_scaling.get(param, 0.0)
            baseline_value = stage1_medians.get(param, 0.0)
            raw_values[param] = template_ratio * baseline_value * (-1)

        # Step 2: Normalize by sum of non-zero raw values
        non_zero_sum = sum(abs(v) for v in raw_values.values() if v != 0.0)

        if non_zero_sum > 0:
            for param in param_columns:
                if raw_values[param] != 0.0:
                    final_value = raw_values[param] / non_zero_sum
                    new_scaling[param] = final_value
                    print(f"  {template_scaling['scaling_name']}.{param}: "
                          f"{template_scaling.get(param, 0.0)} × {stage1_medians.get(param, 0.0):.6f} × (-1) / {non_zero_sum:.6f} = {final_value:.6f}")
        else:
            print(f"  Warning: {template_scaling['scaling_name']} has no non-zero parameters")

        new_scalings.append(new_scaling)

    # Replace response function scalings with new multi-variable versions
    stage3_config['response_function_scalings'] = new_scalings

    # Copy GDP targets from template
    stage3_config['gdp_targets'] = template_config['gdp_targets']

    # Copy forward simulation SSPs from template
    stage3_config['ssp_scenarios']['forward_simulation_ssps'] = template_config['ssp_scenarios']['forward_simulation_ssps']

    return stage3_config


def main():
    """
    Example usage of Stage 2 configuration generation.
    """
    # Example paths (would be command line arguments in real usage)
    stage1_output_dir = "data/output/output_parameter-sensitivity_CanESM5_20250927_231354"
    template_path = "coin_ssp_config_response_functions_template.json"
    base_config_path = "data/output/output_parameter-sensitivity_CanESM5_20250927_231354/coin_ssp_config_var-test.json"
    output_path = "coin_ssp_config_stage3_generated.json"

    print("="*80)
    print("STAGE 2: CONFIGURATION GENERATION")
    print("="*80)

    # Step 1: Load Stage 1 results
    print("\nStep 1: Loading Stage 1 GDP-weighted parameter medians...")
    stage1_medians = load_stage1_results(stage1_output_dir)

    # Step 2: Load template configuration
    print("\nStep 2: Loading template configuration...")
    template_config = load_template_config(template_path)

    # Step 3: Generate Stage 3 configuration
    print("\nStep 3: Generating Stage 3 configuration...")
    stage3_config = generate_stage3_config(stage1_medians, template_config, base_config_path)

    # Step 4: Save the generated configuration
    print(f"\nStep 4: Saving Stage 3 configuration to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(stage3_config, f, indent=2)

    print(f"\n✅ Stage 3 configuration generated successfully!")
    print(f"   Generated {len(stage3_config['response_function_scalings'])} multi-variable response functions")
    print(f"   Ready for Stage 3 execution: python main.py {output_path}")


if __name__ == "__main__":
    main()