#!/usr/bin/env python3
"""
Prepare comprehensive data for GLMM analysis by reshaping the data
to have one row per trial with all necessary information.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import PROCESSED_DATA_DIR, TRIAL_DATA_FILE, AGGREGATED_DATA_FILE, setup_logging

logger = setup_logging(__name__)

def prepare_comprehensive_data():
    """Prepare data in long format for GLMM analysis.
    
    Note: Duplicate trials (where participants repeated the same maze) have already
    been filtered in the data aggregation step - only the best attempt is included.
    """
    logger.info("Preparing comprehensive data for GLMM analysis...")
    
    # Load the raw trial data
    try:
        all_trials = pd.read_pickle(TRIAL_DATA_FILE)
    except FileNotFoundError:
        logger.error(f"Trial data file not found at {TRIAL_DATA_FILE}. Run comprehensive_analysis.py first.")
        return None
        
    # Reshape to long format (one row per task per trial)
    long_data = []
    
    for _, row in all_trials.iterrows():
        base_info = {
            'participant_type': row['participant_type'],
            'participant_id': row['participant_id'],
            'model_name': row['model_name'],
            'encoding_type': row['encoding_type'],
            'maze_size': row['maze_size'],
            'maze_shape': row['maze_shape'],
            'maze_file': row['maze_file'],
            'trial_num': row.get('trial_num', 0)
        }
        
        # Add solve task
        if 'solve_success' in row and pd.notna(row['solve_success']):
            solve_row = base_info.copy()
            solve_row['task'] = 'solve'
            solve_row['success'] = row['solve_success']
            solve_row['edit_distance'] = np.nan
            long_data.append(solve_row)
        
        # Add recognize task
        if 'recognize_success' in row and pd.notna(row['recognize_success']):
            recognize_row = base_info.copy()
            recognize_row['task'] = 'recognize'
            recognize_row['success'] = row['recognize_success']
            recognize_row['edit_distance'] = np.nan
            long_data.append(recognize_row)
        
        # Add generate task
        if 'generate_success' in row and pd.notna(row['generate_success']):
            generate_row = base_info.copy()
            generate_row['task'] = 'generate'
            generate_row['success'] = row['generate_success']
            generate_row['edit_distance'] = row.get('edit_distance', np.nan)
            long_data.append(generate_row)
    
    # Create DataFrame
    df = pd.DataFrame(long_data)
    
    # Clean up data types
    df['success'] = df['success'].astype(int)
    
    logger.info(f"Created {len(df)} trial records for GLMM.")
    logger.info(f"Tasks: {df['task'].value_counts().to_dict()}")
    
    # Save the comprehensive data
    output_pkl = PROCESSED_DATA_DIR / "comprehensive_glmm_data.pkl"
    output_csv = PROCESSED_DATA_DIR / "comprehensive_glmm_data.csv"
    df.to_pickle(output_pkl)
    df.to_csv(output_csv, index=False)
    
    logger.info(f"Comprehensive GLMM data saved to {output_pkl}")
    
    return df

if __name__ == "__main__":
    df = prepare_comprehensive_data() 