#!/usr/bin/env python3
"""
Calculate edit distances between LLM-generated mazes and their closest correct solutions.
"""

import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import hamming
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MACHINE_RESULTS_DIR, PROCESSED_DATA_DIR, setup_logging

# Import edit distance utilities
from scripts.edit_distance import analyze_results

# Set up logging
logger = setup_logging(__name__)

def main():
    """Process all machine results and calculate edit distances."""
    logger.info("Starting edit distance calculation for machine results")
    
    # Use path from config
    machine_results_dir = MACHINE_RESULTS_DIR.parent  # Go up one level to get the data directory
    logger.info(f"Processing machine results from: {machine_results_dir}")
    
    try:
        print("="*60)
        print("Calculating Edit Distances for LLM Generation Results")
        print("="*60)
        
        # Change to the machine results directory
        import os
        original_dir = os.getcwd()
        os.chdir(machine_results_dir)
        
        # Call the existing analyze_results function
        analyze_results(".")
        
        # Move the output files to the processed data directory
        os.chdir(original_dir)
        
        # Check if files were created and move them
        json_file = machine_results_dir / "maze_edit_distances.json"
        csv_file = machine_results_dir / "maze_edit_distances.csv"
        
        if json_file.exists():
            # Ensure processed data directory exists
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            
            # Move files to processed data directory
            import shutil
            shutil.move(str(json_file), str(PROCESSED_DATA_DIR / "maze_edit_distances.json"))
            print(f"\nMoved results to: {PROCESSED_DATA_DIR / 'maze_edit_distances.json'}")
            
            if csv_file.exists():
                shutil.move(str(csv_file), str(PROCESSED_DATA_DIR / "maze_edit_distances.csv"))
                print(f"Moved results to: {PROCESSED_DATA_DIR / 'maze_edit_distances.csv'}")
        else:
            print("\nWarning: No edit distance files were created.")
            print("This might be due to missing LLM generation results.")
            
    except Exception as e:
        print(f"\nError during edit distance calculation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 