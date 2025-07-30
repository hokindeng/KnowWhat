#!/usr/bin/env python3
"""
Calculate edit distances for human-generated mazes.
"""

import json
import numpy as np
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import HUMAN_RESULTS_DIR, setup_logging, PROCESSED_DATA_DIR

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'scripts'))
from edit_distance import get_closest_correct_maze, path_edit_distance

# Set up logging
logger = setup_logging(__name__)

def process_human_generation_data():
    """Calculate edit distances for human-generated mazes."""
    human_dir = HUMAN_RESULTS_DIR
    results = []
    
    logger.info(f"Processing human generation data from: {human_dir}")
    
    # Process each participant directory
    participant_dirs = sorted([d for d in human_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(participant_dirs)} participant directories.")

    for i, participant_dir in enumerate(participant_dirs):
        participant_id = participant_dir.name
        logger.info(f"[{i+1}/{len(participant_dirs)}] Processing participant: {participant_id}")
        
        # Dictionary to store trials by maze key (size, shape, maze_file)
        participant_trials = {}
        
        # Process each JSON file
        json_files = sorted(list(participant_dir.glob("*.json")))
        total_files = len(json_files)
        logger.info(f"    {total_files} JSON files found.")
        
        for j, json_file in enumerate(json_files):
            if j % 10 == 0:
                print(f"    Progress: {j}/{total_files} files processed for {participant_id}")
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Check if this file has generation data
                if 'generation_result' not in data:
                    continue
                
                gen_result = data['generation_result']
                if not gen_result.get('valid', False):
                    continue
                
                # Extract maze info
                maze_size = data['maze_type']['size']
                maze_shape = data['maze_type']['shape']
                maze_file = data['maze_file']
                size = tuple(map(int, maze_size.split('x')))
                
                # Create key for this maze
                maze_key = (maze_size, maze_shape, maze_file)
                
                # Get generated coordinates
                coords = gen_result['generated_shape']
                
                # Create maze array from coordinates
                generated_maze = np.zeros(size, dtype=int)
                for coord in coords:
                    row, col = coord
                    if 0 <= row < size[0] and 0 <= col < size[1]:
                        generated_maze[row, col] = 1  # Mark as path
                
                # Mark start and end positions
                start = tuple(gen_result.get('start_position', coords[0]))
                end = tuple(gen_result.get('end_position', coords[-1]))
                generated_maze[start] = 2  # Start
                generated_maze[end] = 3    # End
                
                # Find closest correct maze and compute distance
                closest_maze, distance = get_closest_correct_maze(
                    generated_maze, maze_shape, size
                )
                
                if closest_maze is not None:
                    trial_data = {
                        'participant_type': 'human',
                        'participant_id': participant_id,
                        'maze_size': maze_size,
                        'maze_shape': maze_shape,
                        'maze_file': maze_file,
                        'edit_distance': distance,
                        'path_ratio': gen_result.get('path_ratio', None),
                        'timestamp': data.get('timestamp', None),
                        'json_file': json_file.name
                    }
                    
                    # Store trial, keeping the best one if duplicate
                    if maze_key not in participant_trials or distance < participant_trials[maze_key]['edit_distance']:
                        participant_trials[maze_key] = trial_data
                    
            except Exception as e:
                print(f"  - Error processing {json_file.name}: {e}")
                continue
        
        # Add the best trials for this participant to results
        results.extend(participant_trials.values())
        print(f"    Completed participant {participant_id}. Selected {len(participant_trials)} best trials from {total_files} files.")
    
    return results

def main():
    """Main function."""
    print("Calculating edit distances for human-generated mazes...")
    
    results = process_human_generation_data()
    
    if results:
        # Calculate summary statistics
        import pandas as pd
        df = pd.DataFrame(results)
        
        print(f"\nProcessed {len(results)} human-generated mazes")
        print("\nEdit Distance Summary by Shape:")
        summary = df.groupby('maze_shape')['edit_distance'].agg(['mean', 'std', 'min', 'max'])
        print(summary)
        
        print("\nOverall Statistics:")
        print(f"Mean edit distance: {df['edit_distance'].mean():.3f}")
        print(f"Std deviation: {df['edit_distance'].std():.3f}")
        print(f"Min: {df['edit_distance'].min():.3f}")
        print(f"Max: {df['edit_distance'].max():.3f}")
        
        # Save results
        output_file = PROCESSED_DATA_DIR / "human_edit_distances.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
        # Also save as CSV
        df.to_csv(PROCESSED_DATA_DIR / "human_edit_distances.csv", index=False)
    else:
        print("No human generation data found!")

if __name__ == "__main__":
    main() 