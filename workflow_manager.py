#!/usr/bin/env python3
"""
Workflow Manager for COIN-SSP Multi-Stage Pipeline

This script orchestrates a three-stage workflow:
1. Stage 1: Run individual response function assessments using initial config
2. Stage 2: Analyze Stage 1 results and generate new config based on GDP-weighted parameter means
3. Stage 3: Run final simulations using the generated config from Stage 2

Usage examples:
    # Full workflow from stage 1
    python workflow_manager.py coin_ssp_config_parameter_sensitivity.json coin_ssp_config_response_functions_template.json

    # Start from stage 2 (if stage 1 already completed)
    python workflow_manager.py coin_ssp_config_parameter_sensitivity.json coin_ssp_config_response_functions_template.json --start-stage 2 --stage1-output ./results/stage1_20231201/

    # Only run stage 3
    python workflow_manager.py coin_ssp_config_parameter_sensitivity.json coin_ssp_config_response_functions_template.json --start-stage 3 --stage2-config ./configs/generated_config.json
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Import stage2_config_generator functions
from stage2_config_generator import load_stage1_results, load_template_config, generate_stage3_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkflowManager:
    """Manages the three-stage COIN-SSP workflow pipeline."""

    def __init__(self, config_dir: Path, output_base_dir: Path = None, verbose: bool = False):
        self.config_dir = Path(config_dir)
        self.output_base_dir = output_base_dir or Path('./workflow_outputs')
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.verbose = verbose

    def run_stage1(self, stage1_config: Path, template_config: Path = None, model_name: str = None, json_id: str = None) -> Path:
        """
        Run Stage 1: Individual response function assessments.

        Args:
            stage1_config: Path to initial configuration file
            template_config: Path to template configuration file (will be copied to output)
            model_name: Climate model name (extracted from config if not provided)
            json_id: JSON ID (extracted from config if not provided)

        Returns:
            Path to stage 1 output directory
        """
        # Load config to extract model_name and json_id if not provided
        if model_name is None or json_id is None:
            with open(stage1_config, 'r') as f:
                config = json.load(f)
            if model_name is None:
                model_name = config['climate_model']['model_name']
            if json_id is None:
                json_id = config['run_metadata']['json_id']

        # Create hierarchical output directory structure
        # data/output/output_{model_name}_{timestamp}/{json_id}
        base_output_dir = Path("./data/output")
        model_dir = base_output_dir / f"output_{model_name}_{self.timestamp}"
        output_dir = model_dir / f"{json_id}"

        logger.info(f"Starting Stage 1: Running main.py with config {stage1_config}")
        logger.info(f"Stage 1 output directory: {output_dir}")

        try:
            # Run main.py with the stage 1 config and explicit output directory
            cmd = ['python', 'main.py', str(stage1_config), '--output-dir', str(output_dir)]

            if self.verbose:
                # Pass through output directly to terminal
                result = subprocess.run(cmd, cwd=Path.cwd(), check=True)
            else:
                # Capture output for logging
                result = subprocess.run(
                    cmd,
                    cwd=Path.cwd(),
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.debug(f"Stage 1 stdout: {result.stdout}")

            logger.info("Stage 1 completed successfully")

            # Copy configuration files to output directory for documentation
            logger.info("Copying configuration files to output directory")
            shutil.copy2(stage1_config, output_dir / stage1_config.name)
            logger.info(f"  Copied {stage1_config.name}")

            if template_config and template_config.exists():
                shutil.copy2(template_config, output_dir / template_config.name)
                logger.info(f"  Copied {template_config.name}")

            return output_dir

        except subprocess.CalledProcessError as e:
            logger.error(f"Stage 1 failed with exit code {e.returncode}")
            if not self.verbose and hasattr(e, 'stderr'):
                logger.error(f"Stage 1 stderr: {e.stderr}")
            raise


    def analyze_stage1_results(self, stage1_output_dir: Path) -> Dict[str, Any]:
        """
        Analyze Stage 1 results to extract GDP-weighted parameter means.

        Args:
            stage1_output_dir: Path to Stage 1 output directory

        Returns:
            Dictionary containing analysis results for Stage 2 config generation
        """
        logger.info(f"Starting Stage 2 analysis of: {stage1_output_dir}")

        # This is where you would implement the analysis of Stage 1 results
        # For now, return a placeholder structure
        analysis_results = {
            'gdp_weighted_means': {
                'y_tas1_mean': 0.0,
                'y_tas2_mean': 0.0,
                'k_tas1_mean': 0.0,
                'k_tas2_mean': 0.0,
                'tfp_tas1_mean': 0.0,
                'tfp_tas2_mean': 0.0,
            },
            'stage1_output_dir': str(stage1_output_dir),
            'analysis_timestamp': self.timestamp
        }

        logger.info("Stage 2 analysis completed")
        logger.info(f"GDP-weighted parameter means: {analysis_results['gdp_weighted_means']}")

        return analysis_results

    def generate_stage2_config(self, stage1_output_dir: Path,
                              template_config_path: Path,
                              output_config_path: Path = None) -> Path:
        """
        Generate Stage 2 configuration file based on Stage 1 results.

        Args:
            stage1_output_dir: Directory containing Stage 1 results
            template_config_path: Required template config to base Stage 2 config on
            output_config_path: Optional override for output config path

        Returns:
            Path to generated Stage 2 config file
        """
        if output_config_path is None:
            output_config_path = self.config_dir / f"coin_ssp_config_stage2_generated_{self.timestamp}.json"

        output_config_path = Path(output_config_path)
        output_config_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating Stage 2 config: {output_config_path}")

        if not template_config_path or not Path(template_config_path).exists():
            raise ValueError(f"Template config file is required and must exist: {template_config_path}")

        # Load Stage 1 results using the proper algorithm
        stage1_medians = load_stage1_results(str(stage1_output_dir))

        # Load template configuration
        template_config = load_template_config(str(template_config_path))

        # Find a base config file (use the original Stage 1 config)
        base_config_candidates = list(Path(stage1_output_dir).glob("coin_ssp_config_*.json"))
        if not base_config_candidates:
            raise ValueError(f"No base config file found in {stage1_output_dir}")
        base_config_path = base_config_candidates[0]

        # Generate Stage 3 configuration using the proper algorithm
        stage3_config = generate_stage3_config(stage1_medians, template_config, str(base_config_path))

        # Save the generated config
        with open(output_config_path, 'w') as f:
            json.dump(stage3_config, f, indent=2)

        logger.info(f"Stage 2 config generated successfully: {output_config_path}")
        return output_config_path

    def run_stage3(self, stage2_config: Path) -> Path:
        """
        Run Stage 3: Execute final simulations with generated config.

        Args:
            stage2_config: Path to Stage 2 generated config file

        Returns:
            Path to stage 3 output directory
        """
        # Load config to extract model_name and json_id
        with open(stage2_config, 'r') as f:
            config = json.load(f)
        model_name = config['climate_model']['model_name']
        json_id = config['run_metadata']['json_id']

        # Create hierarchical output directory structure for Stage 3
        # data/output/output_{model_name}_{timestamp}/{json_id}
        base_output_dir = Path("./data/output")
        model_dir = base_output_dir / f"output_{model_name}_{self.timestamp}"
        output_dir = model_dir / f"{json_id}"

        logger.info(f"Starting Stage 3: Running main.py with generated config {stage2_config}")
        logger.info(f"Stage 3 output directory: {output_dir}")

        try:
            # Run main.py with the stage 2 config and explicit output directory
            cmd = ['python', 'main.py', str(stage2_config), '--output-dir', str(output_dir)]

            if self.verbose:
                # Pass through output directly to terminal
                result = subprocess.run(cmd, cwd=Path.cwd(), check=True)
            else:
                # Capture output for logging
                result = subprocess.run(
                    cmd,
                    cwd=Path.cwd(),
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.debug(f"Stage 3 stdout: {result.stdout}")

            logger.info("Stage 3 completed successfully")

            # Copy configuration file to output directory for documentation
            logger.info("Copying configuration file to output directory")
            shutil.copy2(stage2_config, output_dir / stage2_config.name)
            logger.info(f"  Copied {stage2_config.name}")

            return output_dir

        except subprocess.CalledProcessError as e:
            logger.error(f"Stage 3 failed with exit code {e.returncode}")
            if not self.verbose and hasattr(e, 'stderr'):
                logger.error(f"Stage 3 stderr: {e.stderr}")
            raise

    def run_full_workflow(self, stage1_config: Path,
                         template_config: Path = None) -> Dict[str, Path]:
        """
        Run the complete three-stage workflow.

        Args:
            stage1_config: Path to initial configuration file
            template_config: Optional template for Stage 2 config generation

        Returns:
            Dictionary with paths to all stage outputs
        """
        logger.info("Starting full three-stage workflow")

        results = {}

        # Stage 1: Individual assessments
        stage1_output = self.run_stage1(stage1_config, template_config)
        results['stage1_output'] = stage1_output

        # Stage 2: Config generation from Stage 1 results
        stage2_config = self.generate_stage2_config(stage1_output, template_config)
        results['stage2_config'] = stage2_config

        # Stage 3: Final simulations
        stage3_output = self.run_stage3(stage2_config)
        results['stage3_output'] = stage3_output

        logger.info("Full workflow completed successfully")
        logger.info(f"Results: {results}")

        return results


def main():
    """Main entry point for the workflow manager."""
    parser = argparse.ArgumentParser(
        description="COIN-SSP Multi-Stage Workflow Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required positional arguments
    parser.add_argument('stage1_config', type=Path,
                       help='Configuration file for Stage 1 (e.g., coin_ssp_config_parameter_sensitivity.json)')
    parser.add_argument('template_config', type=Path,
                       help='Template configuration file for Stage 2 (e.g., coin_ssp_config_response_functions_template.json)')

    # Optional arguments
    parser.add_argument('--config-dir', type=Path, default='./configs',
                       help='Directory for configuration files (default: ./configs)')
    parser.add_argument('--output-dir', type=Path, default='./workflow_outputs',
                       help='Base directory for workflow outputs (default: ./workflow_outputs)')
    parser.add_argument('--start-stage', type=int, choices=[1, 2, 3], default=1,
                       help='Stage to start workflow from (default: 1)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    # Stage-specific arguments
    parser.add_argument('--stage1-output', type=Path,
                       help='Stage 1 output directory (required if starting from stage 2+)')
    parser.add_argument('--stage2-config', type=Path,
                       help='Stage 2 generated config file (required if starting from stage 3)')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validation
    if not args.stage1_config.exists():
        parser.error(f"Stage 1 config file not found: {args.stage1_config}")
    if not args.template_config.exists():
        parser.error(f"Template config file not found: {args.template_config}")
    if args.start_stage >= 2 and not args.stage1_output:
        parser.error("--stage1-output is required when starting from stage 2 or later")
    if args.start_stage == 3 and not args.stage2_config:
        parser.error("--stage2-config is required when starting from stage 3")

    try:
        # Initialize workflow manager
        workflow = WorkflowManager(args.config_dir, args.output_dir, verbose=args.verbose)

        if args.start_stage == 1:
            # Full workflow
            results = workflow.run_full_workflow(args.stage1_config, args.template_config)

        elif args.start_stage == 2:
            # Start from stage 2
            stage2_config = workflow.generate_stage2_config(args.stage1_output, args.template_config)
            stage3_output = workflow.run_stage3(stage2_config)
            results = {
                'stage1_output': args.stage1_output,
                'stage2_config': stage2_config,
                'stage3_output': stage3_output
            }

        elif args.start_stage == 3:
            # Only stage 3
            stage3_output = workflow.run_stage3(args.stage2_config)
            results = {'stage3_output': stage3_output}

        logger.info("Workflow completed successfully!")
        print("\n" + "="*60)
        print("WORKFLOW RESULTS:")
        for stage, path in results.items():
            print(f"  {stage}: {path}")
        print("="*60)

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()