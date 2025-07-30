"""
Visualization of participant data showing moves, recognized shapes, and generated shapes.
Creates both text output and PNG visualizations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
from collections import defaultdict

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import HUMAN_RESULTS_DIR, EXPERIMENT_MAZES_DIR, PROJECT_ROOT, setup_logging

# Setup logger
logger = setup_logging(__name__)

# Constants
WALL = 0
PATH = 1
POS = 2
END = 3

# Output directory - now in analysis_results instead of data
OUTPUT_DIR = PROJECT_ROOT / 'analysis_results' / 'participant_visualizations'

def load_maze_file(maze_path):
    """Load a maze from .npy file."""
    return np.load(maze_path)

def apply_moves_to_maze(original_maze, moves):
    """Apply a sequence of moves to show the solving process."""
    maze = original_maze.copy()
    
    # Find starting position
    start_pos = np.argwhere(maze == POS)
    if len(start_pos) == 0:
        return None, []
    
    current_pos = start_pos[0].copy()
    path_taken = [tuple(current_pos)]
    
    # Direction mapping
    move_map = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1),
        'up-left': (-1, -1),
        'up-right': (-1, 1),
        'down-left': (1, -1),
        'down-right': (1, 1)
    }
    
    for move in moves:
        if move.lower() in move_map:
            dy, dx = move_map[move.lower()]
            new_pos = current_pos + np.array([dy, dx])
            
            # Check if move is valid
            if (0 <= new_pos[0] < maze.shape[0] and 
                0 <= new_pos[1] < maze.shape[1] and
                maze[new_pos[0], new_pos[1]] != WALL):
                
                current_pos = new_pos
                path_taken.append(tuple(current_pos))
    
    return current_pos, path_taken

def visualize_solving_process(ax, maze, moves, title="", recognized_shape=None, show_completion_status=True):
    """Visualize the maze with the path taken by the participant."""
    # Create base maze visualization
    height, width = maze.shape
    
    # Color mapping
    color_map = {
        WALL: [0.3, 0.3, 0.3],      # dark gray - wall
        PATH: [1.0, 1.0, 1.0],      # white - path
        POS: [0.0, 0.8, 0.0],       # green - start
        END: [0.8, 0.0, 0.0],       # red - end
    }
    
    # Create RGB image
    img = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            img[i, j] = color_map.get(maze[i, j], [1, 1, 1])
    
    ax.imshow(img)
    
    # Apply moves and draw path
    final_pos, path_taken = apply_moves_to_maze(maze, moves)
    
    # Draw the path taken
    if len(path_taken) > 1:
        for i in range(len(path_taken) - 1):
            y1, x1 = path_taken[i]
            y2, x2 = path_taken[i + 1]
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=3, alpha=0.7)
        
        # Mark positions along the path
        for i, (y, x) in enumerate(path_taken):
            if i == 0:  # Start
                ax.plot(x, y, 'go', markersize=12, label='Start')
            elif i == len(path_taken) - 1:  # Current/final position
                ax.plot(x, y, 'bo', markersize=12, label='Final')
            else:  # Intermediate positions
                ax.plot(x, y, 'b.', markersize=8, alpha=0.5)
    
    # Add recognized shape if provided
    if recognized_shape:
        ax.text(0.5, -0.05, f"Recognized: {recognized_shape}", transform=ax.transAxes, 
               ha='center', color='blue', fontsize=10, weight='bold')
    
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    
    # Add grid
    for i in range(height + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
    for j in range(width + 1):
        ax.axvline(j - 0.5, color='black', linewidth=0.5, alpha=0.3)

def visualize_generated_maze(ax, coords, size, start_pos, end_pos, title=""):
    """Visualize the participant's generated maze."""
    height, width = [int(x) for x in size.split('x')]
    maze = np.zeros((height, width), dtype=np.int8)
    
    # Mark all path coordinates
    for coord in coords:
        if isinstance(coord, list) and len(coord) == 2:
            r, c = coord
            if 0 <= r < height and 0 <= c < width:
                maze[r, c] = PATH
    
    # Mark start and end
    if start_pos and isinstance(start_pos, list) and len(start_pos) == 2:
        maze[start_pos[0], start_pos[1]] = POS
    if end_pos and isinstance(end_pos, list) and len(end_pos) == 2:
        maze[end_pos[0], end_pos[1]] = END
    
    # Visualization with participant colors
    color_map = {
        WALL: [1.0, 1.0, 1.0],      # white background
        PATH: [1.0, 1.0, 0.0],      # yellow - path
        POS: [1.0, 0.5, 0.0],       # orange - start
        END: [0.0, 0.8, 0.0],       # green - end
    }
    
    img = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            img[i, j] = color_map.get(maze[i, j], [1, 1, 1])
    
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    
    # Add grid
    for i in range(height + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
    for j in range(width + 1):
        ax.axvline(j - 0.5, color='black', linewidth=0.5, alpha=0.3)

def create_participant_visualization(participant_id, attempts, output_dir):
    """Create PNG visualization for a participant."""
    # Create figure with subplots for each maze attempt
    n_attempts = len(attempts)
    n_cols = 3  # Original, Solving Process, Generated
    n_rows = n_attempts
    
    fig = plt.figure(figsize=(12, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    for idx, attempt in enumerate(attempts):
        # Load original maze
        maze_path = Path(EXPERIMENT_MAZES_DIR) / attempt['size'] / attempt['shape'] / attempt['maze_file']
        if not maze_path.exists():
            logger.warning(f"Maze file not found: {maze_path}")
            continue
        
        original_maze = load_maze_file(maze_path)
        
        # 1. Original maze (no completion status)
        ax1 = fig.add_subplot(gs[idx, 0])
        visualize_solving_process(ax1, original_maze, [], 
                                f"Original: {attempt['shape']} {attempt['size']}",
                                show_completion_status=False)
        
        # 2. Solving process with moves and recognized shape
        ax2 = fig.add_subplot(gs[idx, 1])
        visualize_solving_process(ax2, original_maze, attempt['moves'], 
                                "",  # Empty title
                                recognized_shape=attempt['recognized_shape'],
                                show_completion_status=True)
        
        # 3. Generated maze (if available)
        ax3 = fig.add_subplot(gs[idx, 2])
        if attempt.get('generated_shape') and isinstance(attempt['generated_shape'], list):
            visualize_generated_maze(ax3, 
                                   attempt['generated_shape'],
                                   attempt['size'],
                                   attempt.get('start_position'),
                                   attempt.get('end_position'),
                                   "")  # Empty title
        else:
            ax3.text(0.5, 0.5, "No generation\ndata available", 
                    transform=ax3.transAxes, ha='center', va='center',
                    fontsize=12, color='gray')
            ax3.axis('off')
    
    plt.suptitle(f'Participant {participant_id} - Maze Solving Data', fontsize=16)
    
    # Save figure
    participant_dir = output_dir / f'participant_{participant_id}'
    participant_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(participant_dir / 'visualization.png', dpi=150, bbox_inches='tight')
    plt.close()

def extract_key_data(json_file):
    """Extract the key pieces of data from a JSON file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract the key pieces
        result = {
            'file': json_file.name,
            'maze_file': data.get('maze_file', 'unknown'),
            'size': data.get('maze_type', {}).get('size', 'unknown'),
            'shape': data.get('maze_type', {}).get('shape', 'unknown'),
            'moves': data.get('moves', []),
            'recognized_shape': data.get('recognized_shape', 'Not available'),
            'generated_shape': data.get('generation_result', {}).get('generated_shape', 
                             data.get('generation_coordinates', 'Not available')),
            'start_position': data.get('generation_result', {}).get('start_position'),
            'end_position': data.get('generation_result', {}).get('end_position')
        }
        
        return result
    except Exception as e:
        logger.error(f"Error loading {json_file}: {e}")
        return None

def display_participant_data(participant_id, data_list):
    """Display participant data in a clean format."""
    print(f"\n{'='*80}")
    print(f"PARTICIPANT: {participant_id}")
    print(f"{'='*80}")
    
    for idx, data in enumerate(data_list, 1):
        print(f"\nAttempt {idx}: {data['shape']} {data['size']} ({data['file']})")
        print("-" * 60)
        
        # Display moves
        print("MOVES:")
        if data['moves']:
            print(f"  {' → '.join(data['moves'])}")
            print(f"  (Total: {len(data['moves'])} moves)")
        else:
            print("  No moves recorded")
        
        # Display recognized shape
        print(f"\nRECOGNIZED SHAPE:")
        print(f"  {data['recognized_shape']}")
        
        # Display generated shape
        print(f"\nGENERATED SHAPE:")
        if isinstance(data['generated_shape'], list):
            print("  Coordinates:")
            for coord in data['generated_shape']:
                print(f"    {coord}")
            
            # Display start and end positions
            if data.get('start_position'):
                print(f"\n  Start Position: {data['start_position']}")
            if data.get('end_position'):
                print(f"  End Position: {data['end_position']}")
        else:
            print(f"  {data['generated_shape']}")
        
        print("-" * 60)

def save_participant_data(participant_id, data_list, output_dir):
    """Save participant data to files."""
    participant_dir = output_dir / f'participant_{participant_id}'
    participant_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    output_file = participant_dir / 'key_data.json'
    with open(output_file, 'w') as f:
        json.dump({
            'participant_id': participant_id,
            'attempts': data_list
        }, f, indent=2)
    
    # Save text
    text_file = participant_dir / 'key_data.txt'
    with open(text_file, 'w') as f:
        f.write(f"PARTICIPANT: {participant_id}\n")
        f.write("="*80 + "\n")
        
        for idx, data in enumerate(data_list, 1):
            f.write(f"\nAttempt {idx}: {data['shape']} {data['size']} ({data['file']})\n")
            f.write("-" * 60 + "\n")
            
            f.write("MOVES:\n")
            if data['moves']:
                f.write(f"  {' → '.join(data['moves'])}\n")
                f.write(f"  (Total: {len(data['moves'])} moves)\n")
            else:
                f.write("  No moves recorded\n")
            
            f.write(f"\nRECOGNIZED SHAPE:\n")
            f.write(f"  {data['recognized_shape']}\n")
            
            f.write(f"\nGENERATED SHAPE:\n")
            if isinstance(data['generated_shape'], list):
                f.write("  Coordinates:\n")
                for coord in data['generated_shape']:
                    f.write(f"    {coord}\n")
                
                # Write start and end positions
                if data.get('start_position'):
                    f.write(f"\n  Start Position: {data['start_position']}\n")
                if data.get('end_position'):
                    f.write(f"  End Position: {data['end_position']}\n")
            else:
                f.write(f"  {data['generated_shape']}\n")
            
            f.write("-" * 60 + "\n")

def main():
    """Main function to extract and display participant data."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading participant data...")
    
    # Process all participants
    all_participants = defaultdict(list)
    
    participant_dirs = sorted(list(HUMAN_RESULTS_DIR.glob("*")))
    participant_dirs = [d for d in participant_dirs if d.is_dir()]
    
    for participant_dir in participant_dirs:
        participant_id = participant_dir.name
        json_files = sorted(list(participant_dir.glob("*.json")))
        
        for json_file in json_files:
            data = extract_key_data(json_file)
            if data:
                all_participants[participant_id].append(data)
    
    logger.info(f"Loaded data for {len(all_participants)} participants")
    
    # Process each participant
    for participant_id, attempts in all_participants.items():
        # Sort by filename to maintain order
        attempts.sort(key=lambda x: x['file'])
        
        # Display in console
        display_participant_data(participant_id, attempts)
        
        # Save to files
        save_participant_data(participant_id, attempts, OUTPUT_DIR)
        
        # Create PNG visualization
        logger.info(f"Creating visualization for participant {participant_id}...")
        create_participant_visualization(participant_id, attempts, OUTPUT_DIR)
    
    # Create summary
    summary = {
        'total_participants': len(all_participants),
        'participants': list(all_participants.keys()),
        'total_attempts': sum(len(attempts) for attempts in all_participants.values())
    }
    
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total participants: {summary['total_participants']}")
    print(f"Total attempts: {summary['total_attempts']}")
    print(f"\nData saved to: {OUTPUT_DIR}")
    print(f"Each participant has their own folder with:")
    print(f"  - key_data.json (raw data)")
    print(f"  - key_data.txt (readable format)")
    print(f"  - visualization.png (visual representation)")

if __name__ == "__main__":
    main() 