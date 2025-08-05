#!/usr/bin/env python3
"""
Comprehensive Analysis of Maze Understanding: Humans vs. LLMs
This script implements the full analysis plan for comparing procedural 
and conceptual knowledge in maze tasks.
"""

import os
import json
import ast
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    HUMAN_RESULTS_DIR, MACHINE_RESULTS_DIR, EXPERIMENT_MAZES_DIR,
    PROCESSED_DATA_DIR, TRIAL_DATA_FILE, AGGREGATED_DATA_FILE, 
    OVERALL_RATES_FILE, LLM_MODELS, ENCODING_TYPES, MAZE_SHAPES,
    HUMAN_EDIT_DISTANCES_FILE, EDIT_DISTANCES_FILE,
    MODEL_FAMILY, MAZE_SIZES, get_results_path,
    setup_logging
)

# Import maze-specific utilities
from core.solution_verifier import is_correct_recognize, is_correct_generate
from core.maze_generator import all_square_path_mazes, all_cross_path_mazes, all_spiral_path_mazes
from core.maze_generator import all_triangle_path_mazes, all_C_path_mazes, all_Z_path_mazes
from core.maze_generator import init_all_start_end

# Setup logging
logger = setup_logging(__name__)

# ==================== Helper Functions ====================

def flatten_dict(d, parent_key='', sep='.'):
    """Flatten nested dictionary."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        else:
            items[k] = v
    return items

def simulate_moves_and_check_near_goal(maze, moves):
    """Simulate moves and return True if final position is at goal or one step away."""
    try:
        # Constants
        WALL = 0
        PATH = 1
        POS = 2
        END = 3
        # Find start and goal
        start = tuple(map(int, np.argwhere(maze == POS)[0]))
        goal = tuple(map(int, np.argwhere(maze == END)[0]))
        r, c = start
        move_delta = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
            "up-left": (-1, -1),
            "up-right": (-1, 1),
            "down-left": (1, -1),
            "down-right": (1, 1),
        }
        for mv in moves:
            if mv not in move_delta:
                return False  # invalid move string
            dr, dc = move_delta[mv]
            r, c = r + dr, c + dc
            # Bounds check
            if r < 0 or r >= maze.shape[0] or c < 0 or c >= maze.shape[1]:
                return False
            # Check wall
            if maze[r, c] == WALL:
                return False
        # Compute distance to goal
        dist = max(abs(goal[0] - r), abs(goal[1] - c))  # Chebyshev distance supports diagonal
        return dist <= 1
    except Exception:
        return False

def load_human_data():
    """Load and process human participant data."""
    logger.info("Loading human data...")
    
    all_trials = []
    
    # Load human edit distances if available
    human_edit_dist_dict = {}
    if HUMAN_EDIT_DISTANCES_FILE.exists():
        try:
            with open(HUMAN_EDIT_DISTANCES_FILE, 'r') as f:
                human_edit_distances = json.load(f)
            human_edit_dist_dict = {
                (d['participant_id'], d['maze_size'], d['maze_shape'], d['maze_file']): d['edit_distance'] 
                for d in human_edit_distances
            }
            logger.info(f"Loaded human edit distances for {len(human_edit_dist_dict)} trials")
        except Exception as e:
            logger.warning(f"Could not load human edit distances: {e}")
    else:
        logger.warning(f"Human edit distances file not found at {HUMAN_EDIT_DISTANCES_FILE}")
    
    # Process each participant directory
    participant_count = 0
    duplicate_count = 0
    
    for participant_dir in HUMAN_RESULTS_DIR.iterdir():
        if not participant_dir.is_dir():
            continue
            
        participant_id = participant_dir.name
        participant_count += 1
        
        # Dictionary to track best trials for this participant
        participant_best_trials = {}
        
        # Process each JSON file in the participant's directory
        json_files = list(participant_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Create unique key for this maze
                maze_key = (data['maze_type']['size'], data['maze_type']['shape'], data['maze_file'])
                
                # Extract key information
                trial = {
                    'participant_type': 'human',
                    'participant_id': participant_id,
                    'model_name': 'human',
                    'encoding_type': 'visual',  # Humans use visual interface
                    'maze_size': data['maze_type']['size'],
                    'maze_shape': data['maze_type']['shape'],
                    'maze_file': data['maze_file'],
                    'timestamp': data.get('timestamp', None),
                    'json_file': json_file.name
                }
                
                # Process moves
                moves = data.get('moves', [])
                if isinstance(moves, str):
                    try:
                        moves = json.loads(moves)
                    except:
                        # Fallback: split by comma or space
                        moves = [mv.strip() for mv in moves.split(',') if mv.strip()]
                
                # Solve task results
                solve_success = data.get('maze_complete', False) and not data.get('failed', False)
                
                # If not marked as complete, check if they were close to goal
                if not solve_success and moves:
                    maze_path = EXPERIMENT_MAZES_DIR / data['maze_type']['size'] / data['maze_type']['shape'] / data['maze_file']
                    if maze_path.exists():
                        try:
                            maze_arr = np.load(maze_path)
                            # Check if they were one step from goal
                            solve_success = simulate_moves_and_check_near_goal(maze_arr, moves)
                        except Exception as e:
                            logger.debug(f"Could not check near-goal for {json_file}: {e}")
                
                trial['solve_success'] = solve_success
                trial['solve_moves'] = len(moves)
                trial['solve_time'] = data.get('completion_time', None)
                
                # Recognition task results
                if 'recognized_shape' in data:
                    trial['recognize_success'] = data.get('recognition_correct', False)
                    trial['recognized_shape'] = data.get('recognized_shape', None)
                else:
                    trial['recognize_success'] = None
                    trial['recognized_shape'] = None
                
                # Generation task results
                if 'generation_result' in data:
                    gen_result = data['generation_result']
                    trial['generate_success'] = gen_result.get('valid', False)
                    trial['generate_shape'] = gen_result.get('shape', None)
                    
                    # Get edit distance if available
                    if gen_result.get('valid', False):
                        edit_key = (participant_id, data['maze_type']['size'], 
                                   data['maze_type']['shape'], data['maze_file'])
                        trial['edit_distance'] = human_edit_dist_dict.get(edit_key, None)
                    else:
                        trial['edit_distance'] = None
                else:
                    trial['generate_success'] = None
                    trial['generate_shape'] = None
                    trial['edit_distance'] = None
                
                # Calculate a quality score for this trial
                quality_score = 0
                if trial['solve_success']:
                    quality_score += 1
                if trial['recognize_success']:
                    quality_score += 1
                if trial['generate_success']:
                    quality_score += 1
                    # For generation, prefer lower edit distance
                    if trial['edit_distance'] is not None:
                        quality_score -= trial['edit_distance']
                
                trial['_quality_score'] = quality_score
                
                # Check if we already have a trial for this maze
                if maze_key in participant_best_trials:
                    duplicate_count += 1
                    # Keep the better trial based on quality score
                    if quality_score > participant_best_trials[maze_key]['_quality_score']:
                        logger.debug(f"Replacing trial for {participant_id} - {maze_key} (better quality: {quality_score:.3f})")
                        participant_best_trials[maze_key] = trial
                else:
                    participant_best_trials[maze_key] = trial
                    
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                continue
        
        # Add best trials for this participant
        for trial in participant_best_trials.values():
            # Remove the internal quality score before adding
            trial.pop('_quality_score', None)
            all_trials.append(trial)
    
    df = pd.DataFrame(all_trials)
    logger.info(f"Loaded {len(df)} human trials from {participant_count} participants")
    logger.info(f"Filtered out {duplicate_count} duplicate trials (kept best attempts)")
    
    # Validate expected number of trials per participant
    if len(df) > 0:
        trials_per_participant = df.groupby('participant_id').size()
        expected_trials = 12  # 2 sizes × 6 shapes = 12
        
        participants_with_full_data = (trials_per_participant == expected_trials).sum()
        participants_with_partial_data = (trials_per_participant < expected_trials).sum()
        participants_with_extra_data = (trials_per_participant > expected_trials).sum()
        
        logger.info(f"Data completeness check:")
        logger.info(f"  - {participants_with_full_data} participants with complete data (12 trials)")
        logger.info(f"  - {participants_with_partial_data} participants with partial data (<12 trials)")
        logger.info(f"  - {participants_with_extra_data} participants with extra data (>12 trials, should be 0 after deduplication)")
        
        if participants_with_extra_data > 0:
            logger.warning("Some participants still have >12 trials after deduplication. Investigating...")
            for pid in trials_per_participant[trials_per_participant > expected_trials].index:
                participant_trials = df[df['participant_id'] == pid]
                logger.warning(f"  Participant {pid}: {len(participant_trials)} trials")
                maze_combos = participant_trials[['maze_size', 'maze_shape', 'maze_file']].value_counts()
                if any(maze_combos > 1):
                    logger.warning(f"    Still has duplicates: {maze_combos[maze_combos > 1].to_dict()}")
    
    return df

def load_llm_data():
    """Load and process all LLM data."""
    logger.info("Loading LLM data...")
    all_trials = []
    
    # Load edit distances if available
    edit_dist_dict = {}
    if EDIT_DISTANCES_FILE.exists():
        try:
            with open(EDIT_DISTANCES_FILE, 'r') as f:
                edit_distances = json.load(f)
            edit_dist_dict = {
                (d['encoding_type'], d['model_name'], d['trial_file']): d['edit_distance'] 
                for d in edit_distances
            }
            logger.info(f"Loaded edit distances for {len(edit_dist_dict)} trials")
        except Exception as e:
            logger.warning(f"Could not load edit distances: {e}")
    else:
        logger.warning(f"Edit distances file not found at {EDIT_DISTANCES_FILE}")
    
    # Process each encoding type and model
    for encoding_type in ENCODING_TYPES:  # Process all encoding types including vision
        if encoding_type == "vision":
            # Handle vision models specially
            vision_dirs = [
                ("opus_vision_results", "claude-4-opus"),
                ("sonnet_vision_results", "claude-3.5-sonnet")
            ]
            for dir_name, model_name in vision_dirs:
                results_dir = MACHINE_RESULTS_DIR / dir_name
                if results_dir.exists():
                    process_model_results(results_dir, encoding_type, model_name, 
                                        all_trials, edit_dist_dict)
        else:
            # Process text-based models
            for model_key, model_provider in LLM_MODELS.items():
                # Get results directory
                results_dir = get_results_path(model_provider, encoding_type)
                
                if not results_dir.exists():
                    logger.debug(f"Results directory not found: {results_dir}")
                    continue
                
                process_model_results(results_dir, encoding_type, model_key, 
                                    all_trials, edit_dist_dict)
    
    df = pd.DataFrame(all_trials)
    logger.info(f"Loaded {len(df)} LLM records")
    
    return df

def process_model_results(results_dir, encoding_type, model_name, all_trials, edit_dist_dict):
    """Process results for a single model."""
    # Process each size
    for size_dir in results_dir.iterdir():
        if not size_dir.is_dir():
            continue
            
        # Validate size
        if size_dir.name not in [f"{s[0]}x{s[1]}" for s in MAZE_SIZES]:
            continue
            
        maze_size = size_dir.name

        # Process each shape
        for shape_dir in size_dir.iterdir():
            if not shape_dir.is_dir() or shape_dir.name not in MAZE_SHAPES:
                continue
                
            maze_shape = shape_dir.name

            # Process each maze file
            for maze_dir in shape_dir.iterdir():
                if not maze_dir.is_dir() or not maze_dir.name.endswith('.npy'):
                    continue
                    
                maze_file = maze_dir.name

                # Process each trial
                for trial_file in maze_dir.glob("*.txt"):
                    try:
                        trial = load_llm_trial(
                            trial_file, encoding_type, model_name, 
                            maze_size, maze_shape, maze_file,
                            edit_dist_dict
                        )
                        if trial:
                            all_trials.append(trial)
                    except Exception as e:
                        logger.error(f"Error processing {trial_file}: {e}")
                        continue

def load_llm_trial(trial_file, encoding_type, model_name, maze_size, maze_shape, 
                   maze_file, edit_dist_dict):
    """Load a single LLM trial from a text file."""
    try:
        with open(trial_file, 'r') as f:
            content = f.read()
        
        record = {
            'participant_type': 'llm',
            'participant_id': f"{model_name}_{encoding_type}",
            'model_name': model_name,
            'encoding_type': encoding_type,
            'maze_size': maze_size,
            'maze_shape': maze_shape,
            'maze_file': maze_file,
            'trial_num': int(trial_file.stem)
        }
        
        # Parse results based on SUCCESS/FAIL markers
        
        # Solve task
        if 'SOLVE: SUCCESS' in content:
            record['solve_success'] = True
        elif 'SOLVE: FAIL' in content:
            record['solve_success'] = False
        else:
            # If no explicit marker, try to infer from content
            solve_success = False
            if "successfully reached the goal" in content.lower() or "maze solved" in content.lower():
                solve_success = True
            elif "Here is the path" in content and "G:" in content:
                solve_success = True
            record['solve_success'] = solve_success
        
        # Recognition task
        if 'RECOGNIZE: SUCCESS' in content:
            record['recognize_success'] = True
        elif 'RECOGNIZE: FAIL' in content:
            record['recognize_success'] = False
        else:
            # If no explicit marker, use the verifier function
            if "shape" in content.lower() and maze_shape:
                recognize_success = is_correct_recognize(content, maze_shape)
                record['recognize_success'] = recognize_success
        
        # Generation task
        if 'GENERATE: SUCCESS' in content:
            record['generate_success'] = True
        elif 'GENERATE: FAIL' in content:
            record['generate_success'] = False
        else:
            # If no explicit marker, use the verifier function
            if "generate" in content.lower() or "new maze" in content.lower():
                # For now, we'll use the marker if available
                record['generate_success'] = False  # Default if no marker
            
        # Get edit distance for generation task
        if 'generate_success' in record:
            trial_path = f"{maze_size}/{maze_shape}/{maze_file}/{trial_file.name}"
            # The edit distance file uses the full trial_file path from the results directory
            # We need to reconstruct the full path as it appears in the JSON file
            if encoding_type == "vision":
                # Handle vision results specially
                if "opus" in model_name:
                    results_prefix = "opus_vision_results"
                elif "sonnet" in model_name:
                    results_prefix = "sonnet_vision_results"
                else:
                    results_prefix = f"vision_{model_name}_results"
            else:
                # Map model to provider for directory name
                provider = LLM_MODELS.get(model_name)
                provider_map = {
                    "anthropic": "claude",
                    "openai": "openai", 
                    "google": "gemini",
                    "llama": "llama"
                }
                provider_dir = provider_map.get(provider, provider)
                results_prefix = f"{encoding_type}_{provider_dir}_results"
            
            # Construct the full trial file path as it appears in edit distance JSON
            full_trial_path = f"{results_prefix}/{trial_path}"
            
            # Now lookup using the provider name as the model_name in the key
            provider = LLM_MODELS.get(model_name)
            if provider:
                edit_key = (encoding_type, provider, full_trial_path)
                record['edit_distance'] = edit_dist_dict.get(edit_key, None)
                
                # Debug: print first few lookups to verify
                if not hasattr(load_llm_trial, '_debug_count'):
                    load_llm_trial._debug_count = 0
                if load_llm_trial._debug_count < 3 and record.get('generate_success'):
                    logger.debug(f"Edit distance lookup: key={edit_key}, found={record['edit_distance']}")
                    load_llm_trial._debug_count += 1
            else:
                record['edit_distance'] = None
        
        return record
        
    except Exception as e:
        logger.error(f"Error loading {trial_file}: {e}")
        return None

def aggregate_data(df):
    """Aggregate data by participant and condition."""
    print("Aggregating data...")
    
    # Group by participant and experimental conditions
    group_cols = ['participant_type', 'participant_id', 'model_name', 
                  'encoding_type', 'maze_size', 'maze_shape']
    
    # Calculate success rates for each task
    agg_funcs = {
        'solve_success': ['mean', 'count'],
        'recognize_success': ['mean', 'count'],
        'generate_success': ['mean', 'count'],
        'edit_distance': ['mean', 'median', 'std']
    }
    
    # Remove columns that don't exist
    existing_cols = df.columns
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in existing_cols}
    
    aggregated = df.groupby(group_cols).agg(agg_funcs).reset_index()
    
    # Flatten column names
    aggregated.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                         for col in aggregated.columns.values]
    
    # Rename columns for clarity
    rename_dict = {
        'solve_success_mean': 'solve_rate',
        'solve_success_count': 'solve_trials',
        'recognize_success_mean': 'recognize_rate',
        'recognize_success_count': 'recognize_trials',
        'generate_success_mean': 'generate_rate',
        'generate_success_count': 'generate_trials',
        'edit_distance_mean': 'edit_distance_mean',
        'edit_distance_median': 'edit_distance_median',
        'edit_distance_std': 'edit_distance_std'
    }
    
    aggregated.rename(columns={k: v for k, v in rename_dict.items() if k in aggregated.columns}, 
                     inplace=True)
    
    return aggregated

def calculate_overall_rates(df):
    """Calculate overall success rates by participant type and task."""
    print("Calculating overall success rates...")
    
    # Pivot to get rates for each task
    overall_rates = []
    
    for participant_type in df['participant_type'].unique():
        for model in df[df['participant_type'] == participant_type]['model_name'].unique():
            for encoding in df[(df['participant_type'] == participant_type) & 
                             (df['model_name'] == model)]['encoding_type'].unique():
                
                subset = df[(df['participant_type'] == participant_type) & 
                          (df['model_name'] == model) & 
                          (df['encoding_type'] == encoding)]
                
                if len(subset) == 0:
                    continue
                
                label = "Human"
                if participant_type == 'llm':
                    family = MODEL_FAMILY.get(model, model)
                    if encoding == 'vision':
                        if 'sonnet' in model:
                            label = f"{family}-Sonnet (vision)"
                        elif 'opus' in model:
                            label = f"{family}-Opus (vision)"
                        else:
                            label = f"{family} (vision)"
                    else:
                        label = f"{family} ({encoding})"

                row = {
                    'participant_type': participant_type,
                    'model_name': model,
                    'encoding_type': encoding,
                    'label': label
                }
                
                for task in ['solve', 'recognize', 'generate']:
                    rate_col = f'{task}_rate'
                    trials_col = f'{task}_trials'
                    
                    if rate_col in subset.columns and trials_col in subset.columns:
                        # Weighted average across all conditions
                        total_trials = subset[trials_col].sum()
                        if total_trials > 0:
                            weighted_rate = (subset[rate_col] * subset[trials_col]).sum() / total_trials
                            row[f'{task}_rate'] = weighted_rate
                            row[f'{task}_trials'] = total_trials
                        else:
                            row[f'{task}_rate'] = 0
                            row[f'{task}_trials'] = 0
                    else:
                        row[f'{task}_rate'] = 0
                        row[f'{task}_trials'] = 0
                
                overall_rates.append(row)
    
    return pd.DataFrame(overall_rates)

def save_results(all_trials_df, aggregated_df, overall_rates_df):
    """Save all results to disk."""
    logger.info("Saving results...")
    
    # Create output directory if it doesn't exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save all dataframes
    all_trials_df.to_csv(TRIAL_DATA_FILE.with_suffix('.csv'), index=False)
    all_trials_df.to_pickle(TRIAL_DATA_FILE)
    
    aggregated_df.to_csv(AGGREGATED_DATA_FILE.with_suffix('.csv'), index=False)
    aggregated_df.to_pickle(AGGREGATED_DATA_FILE)
    
    overall_rates_df.to_csv(OVERALL_RATES_FILE.with_suffix('.csv'), index=False)
    overall_rates_df.to_pickle(OVERALL_RATES_FILE)
    
    logger.info(f"Results saved to {PROCESSED_DATA_DIR}")
    logger.info(f"  - {TRIAL_DATA_FILE.name}")
    logger.info(f"  - {AGGREGATED_DATA_FILE.name}")
    logger.info(f"  - {OVERALL_RATES_FILE.name}")

def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("Maze Understanding Analysis: Humans vs. LLMs")
    print("=" * 60)
    
    # Load data
    human_df = load_human_data()
    llm_df = load_llm_data()
    
    # Combine dataframes
    all_data = pd.concat([human_df, llm_df], ignore_index=True)
    
    # Aggregate by condition
    aggregated_data = aggregate_data(all_data)
    
    # Calculate overall rates
    overall_rates = calculate_overall_rates(aggregated_data)
    
    # Save processed data
    save_results(all_data, aggregated_data, overall_rates)
    
    print("\nData processing complete!")
    print(f"Total trials: {len(all_data)}")
    print(f"Aggregated conditions: {len(aggregated_data)}")
    print(f"\nOverall success rates:")
    print(overall_rates[['label', 'solve_rate', 'recognize_rate', 'generate_rate']])
    
    return all_data, aggregated_data, overall_rates

if __name__ == "__main__":
    all_data, aggregated_data, overall_rates = main() 