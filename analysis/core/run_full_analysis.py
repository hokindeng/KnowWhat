#!/usr/bin/env python3
"""
Run complete analysis pipeline.
"""

import subprocess
import sys
from pathlib import Path
import logging

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, 
                   MACHINE_RESULTS_DIR, STATISTICAL_RESULTS_FILE, setup_logging)

# Setup logger
logger = setup_logging(__name__)

def run_script(script_name, check_error=True):
    """Run a Python script and check for errors."""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=check_error)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Info:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Run full analysis pipeline."""
    logger.info("Starting full analysis pipeline")
    
    analysis_dir = Path(__file__).parent.parent # This is .../analysis
    
    # Ensure output directories exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("MAZE UNDERSTANDING ANALYSIS PIPELINE")
    print("="*60)
    print("This will run the complete analysis in order:")
    print("0. Calculate edit distances (Human and LLM)")
    print("1. Data aggregation and preprocessing")
    print("2. Prepare data for GLMM analysis")
    print("3. Statistical analysis (GLMMs)")
    print("4. Figure generation")
    print("5. Generate analysis report")
    print("="*60)
    
    # Step 0: Calculate Edit Distances
    print("\nSTEP 0: Calculating Edit Distances")
    if not run_script(analysis_dir / "utils" / "calculate_human_edit_distances.py"):
        print("Warning: Human edit distance calculation failed. Continuing...")
    if not run_script(analysis_dir / "utils" / "calculate_edit_distances.py"):
        print("Warning: LLM edit distance calculation failed. Continuing...")

    # Step 1: Data Aggregation
    print("\nSTEP 1: Data Aggregation and Preprocessing")
    if not run_script(analysis_dir / "utils" / "data_aggregation.py"):
        print("Failed at Step 1. Stopping pipeline.")
        return

    # Step 2: Prepare data for GLMM
    print("\nSTEP 2: Preparing Data for Statistical Analysis")
    if not run_script(analysis_dir / "statistical" / "prepare_glmm_data.py"):
        print("Failed at Step 2. Stopping pipeline.")
        return

    # Step 3: Statistical Analysis
    print("\nSTEP 3: Statistical Analysis (GLMM)")
    if not run_script(analysis_dir / "statistical" / "statistical_analysis.py"):
        print("Failed at Step 3. Stopping pipeline.")
        return

    # Step 4: Figure Generation
    print("\nSTEP 4: Figure Generation")
    if not run_script(analysis_dir / "visualization" / "generate_figures.py"):
        print("Failed at Step 4. Stopping pipeline.")
        return

    # Step 5: Report Generation
    print("\nSTEP 5: Generating Analysis Report")
    if not run_script(analysis_dir / "utils" / "report_generation.py"):
        print("Failed at Step 5. Report generation incomplete.")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nResults saved to:")
    print(f"- Processed data: {PROCESSED_DATA_DIR}")
    print(f"- Statistical results: {STATISTICAL_RESULTS_FILE}")
    print(f"- Figures: {FIGURES_DIR}")
    print(f"- Analysis report: {PROCESSED_DATA_DIR.parent / 'analysis_report.md'}")
    
    # List generated files
    if PROCESSED_DATA_DIR.exists():
        print(f"\nProcessed data files:")
        for file in sorted(PROCESSED_DATA_DIR.glob("*")):
            if file.is_file():
                print(f"  - {file.name}")
    
    if FIGURES_DIR.exists():
        print(f"\nGenerated figures:")
        for file in sorted(FIGURES_DIR.glob("*")):
            if file.is_file():
                print(f"  - {file.name}")

if __name__ == "__main__":
    main() 