import os
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Set
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from core.maze_generator import *
import json
from tqdm import tqdm
from config import LLM_MODELS, VISION_MODELS, MAZE_SHAPES as SHAPES

# Cache for generated canonical mazes to avoid re-computation
_canonical_maze_cache = {}

# --- Fast mask cache ---------------------------------------------------------
_mask_cache = {}

def _maze_to_mask(maze: np.ndarray) -> np.ndarray:
    """Convert a maze to a boolean mask of path cells (including P/G)."""
    key = maze.tobytes()
    if key in _mask_cache:
        return _mask_cache[key]
    mask = ((maze == PATH) | (maze == POS) | (maze == END))
    _mask_cache[key] = mask
    return mask

# -----------------------------------------------------------------------------

def parse_matrix_maze_old(text: str) -> np.ndarray:
    """Parse a maze in matrix format into a numpy array."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # Find the maze part - look for a block of lines containing only maze characters
    maze_lines = []
    in_maze = False
    for line in lines:
        if all(c in '01PG ' for c in line):
            in_maze = True
            maze_lines.append(line)
        elif in_maze:  # Stop when we exit the maze block
            break
    
    if not maze_lines:
        return None
        
    # Convert to numpy array
    maze = np.array([list(line.replace(' ', '')) for line in maze_lines])
    # Convert string elements to integers using constants from maze_generator
    mapping = {'0': PATH, '1': WALL, 'P': POS, 'G': END}
    maze_int = np.zeros_like(maze, dtype=int)
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            maze_int[i,j] = mapping[maze[i,j]]
    return maze_int

def parse_matrix_maze_1(temp_text: str) -> np.ndarray:
    maze = []
    print(temp_text)
    try:
        # This is for OpenAI
        # First attempt: Extract using a fenced code block pattern
        match = re.search(r"```(.*?)```", temp_text, re.DOTALL)
        if not match:
            raise ValueError("No fenced code block found.")
        match = match.group(1)
    except ValueError as e:
        # Fallback: Try extracting a matrix-like pattern
        try:
            # This is for Claude
            matrix_pattern = r"(\d+(?: \d+)*)(?:\n|\Z)+"
            match = re.search(matrix_pattern, temp_text, re.DOTALL)
            if not match:
                print(temp_text)
                raise ValueError("No maze matrix found in the text.")
            match = match.group(0).strip()
        except ValueError as final_error:
            raise ValueError("Failed to extract data: {}".format(final_error))
    matrix_text = match.group(1)
    for line in matrix_text.split('\n'):
        row = []
        for char in line.split():
            if char == 'P':
                row.append(-1)  # Represent 'P' as -1
            elif char == 'G':
                row.append(-2)  # Represent 'G' as -2
            else:
                row.append(int(char))  # Convert '1' and '0' to integers
        maze.append(row)
    maze_array = np.array(maze)
    return maze_array

def parse_matrix_maze_v2(temp_text: str) -> np.ndarray:
    print(temp_text)
    # Try to extract a fenced code block first
    match = re.search(r"```(.*?)```", temp_text, re.DOTALL)
    if match:
        # If we found a code block, use the captured group as the matrix text
        matrix_text = match.group(1).strip()
    else:
        # No fenced code block found, try the fallback pattern
        matrix_pattern = r"(\d+(?: \d+)*(?:\n\d+(?: \d+)*)*)"
        match = re.search(matrix_pattern, temp_text, re.DOTALL)
        if not match:
            print(temp_text)
            raise ValueError("No maze matrix found in the text.")
        # Here, match.group(1) will contain all the lines matched by the pattern
        matrix_text = match.group(1).strip()

    # Now parse the extracted matrix_text
    maze = []
    print(matrix_text)
    for line in matrix_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        row = []
        for char in parts:
            if char == 'P':
                row.append(-1)  # Represent 'P' as -1
            elif char == 'G':
                row.append(-2)  # Represent 'G' as -2
            else:
                row.append(int(char))  # Convert '1' and '0' to integers
        maze.append(row)

    maze_array = np.array(maze)
    return maze_array

def parse_matrix_maze_v3(temp_text: str) -> np.ndarray:
    print(temp_text)

    # Try to extract a fenced code block first
    code_block_match = re.search(r"```(.*?)```", temp_text, re.DOTALL)
    if code_block_match:
        # If we found a code block, use that text
        matrix_text = code_block_match.group(1).strip()
    else:
        # No fenced code block found, try a fallback pattern.
        # This regex matches multiple lines of maze characters (0,1,P,G), each line possibly containing multiple space-separated values.
        matrix_pattern = r"([0-1PG](?: [0-1PG])*(?:\n[0-1PG](?: [0-1PG])*)*)"
        fallback_match = re.search(matrix_pattern, temp_text, re.DOTALL)
        if not fallback_match:
            # If still not found, raise an error
            print(temp_text)
            raise ValueError("No maze matrix found in the text.")
        # Extract the matched text from the fallback pattern
        matrix_text = fallback_match.group(1).strip()

    # Now parse the extracted matrix_text
    maze = []
    print("Matrix text extracted:")
    print(matrix_text)
    for line in matrix_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        row = []
        for char in parts:
            if char == 'P':
                row.append(-1)  # Represent 'P' as -1
            elif char == 'G':
                row.append(-2)  # Represent 'G' as -2
            else:
                # '0' or '1' expected here
                row.append(int(char))
        maze.append(row)

    maze_array = np.array(maze)
    return maze_array

def parse_coordinate_maze(text: str, size: Tuple[int, int]) -> np.ndarray:
    """Parse a maze in coordinate format into a numpy array."""
    try:
        # Extract coordinates using regex with more flexible patterns
        walls_match = re.search(r"Walls:\s*(\([\d\s,]+\)(?:\s*,\s*\([\d\s,]+\))*)", text)
        empty_match = re.search(r"Empty:\s*(\([\d\s,]+\)(?:\s*,\s*\([\d\s,]+\))*)", text)
        player_match = re.search(r"Player position:\s*\((\d+)\s*,\s*(\d+)\)", text)
        goal_match = re.search(r"Goal:\s*\((\d+)\s*,\s*(\d+)\)", text)
        
        if not all([walls_match, empty_match, player_match, goal_match]):
            return None
            
        # Initialize maze with walls
        maze = np.full(size, WALL)  # Using WALL constant from maze_generator
        
        # Parse coordinates
        def parse_coords(coord_str: str) -> List[Tuple[int, int]]:
            coords = []
            for match in re.finditer(r"\((\d+)\s*,\s*(\d+)\)", coord_str):
                coords.append((int(match.group(1)), int(match.group(2))))
            return coords
        
        # Fill in maze using constants from maze_generator
        walls_coords = parse_coords(walls_match.group(1))
        empty_coords = parse_coords(empty_match.group(1))
        
        for x, y in walls_coords:
            maze[x, y] = WALL
        for x, y in empty_coords:
            maze[x, y] = PATH
        
        # Set player and goal positions
        player_x, player_y = int(player_match.group(1)), int(player_match.group(2))
        goal_x, goal_y = int(goal_match.group(1)), int(goal_match.group(2))
        
        maze[player_x, player_y] = POS
        maze[goal_x, goal_y] = END
        
        return maze
    except Exception as e:
        print(f"Error parsing coordinate maze: {e}")
        return None

def get_path_coordinates(maze: np.ndarray) -> Set[Tuple[int, int]]:
    """Get set of coordinates that form the path (empty spaces) in the maze."""
    return {tuple(coord) for coord in np.argwhere((maze == PATH) | (maze == POS) | (maze == END))}

def path_edit_distance(maze1: np.ndarray, maze2: np.ndarray) -> float:
    """Compute edit distance between two mazes using XOR on boolean masks.
    Returns a normalised score between 0 (identical) and 1 (completely different)."""
    mask1 = _maze_to_mask(maze1)
    mask2 = _maze_to_mask(maze2)
    diff = np.logical_xor(mask1, mask2)
    return diff.sum() / diff.size

def get_closest_correct_maze(generated_maze: np.ndarray, shape: str, size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    """
    Find the correct maze that's most similar to the generated one.
    Returns the closest maze and its edit distance.
    """
    cache_key = (shape, size)

    if cache_key in _canonical_maze_cache:
        all_mazes_with_paths = _canonical_maze_cache[cache_key]
    else:
        try:
            if shape not in SHAPES:
                raise ValueError(f"Invalid shape: {shape}. Must be one of {SHAPES}")
                
            # Get all possible correct mazes for this shape
            all_mazes = globals()[f"all_{shape}_path_mazes"](size)
            print(f"    [DEBUG] Generated {len(all_mazes)} base maze(s) for {shape} {size}")
            
            # Limit the number of variations to prevent combinatorial explosion
            MAX_VARIATIONS = 1000
            all_mazes_with_paths = []
            print(f"  [Cache PREP] Generating and caching variations for {len(all_mazes)} base mazes...")
            for maze in tqdm(all_mazes[:10], desc=f"Generating {shape} variations"):  # Limit to first 10 base mazes
                variations = init_all_start_end(maze)
                all_mazes_with_paths.extend(variations[:MAX_VARIATIONS // len(all_mazes)])
                if len(all_mazes_with_paths) >= MAX_VARIATIONS:
                    break
            
            all_mazes_with_paths = all_mazes_with_paths[:MAX_VARIATIONS]
            _canonical_maze_cache[cache_key] = all_mazes_with_paths
            print(f"  [Cache PREP] Cached {len(all_mazes_with_paths)} canonical maze variations for {shape} {size} (limited from potentially more)")

        except Exception as e:
            print(f"Error generating canonical mazes for {shape} {size}: {e}")
            return None, float('inf')

    # Find the maze with minimum edit distance
    min_distance = float('inf')
    closest_maze = None

    if not all_mazes_with_paths:
        return None, float('inf')

    for maze in all_mazes_with_paths:
        distance = path_edit_distance(generated_maze, maze)
        if distance < min_distance:
            min_distance = distance
            closest_maze = maze
            
    return closest_maze, min_distance

def extract_generated_maze(text: str) -> str:
    """Extract the generated maze text from the LLM's response."""
    # Look for the maze text that appears before "GENERATE: FAIL"
    parts = text.split("GENERATE: FAIL")
    if len(parts) < 2:
        return None
    
    # Look for the last maze definition before "GENERATE: FAIL"
    generated_text = parts[0]
    
    # For coordinate format, look for the last occurrence of coordinate lists
    coord_matches = list(re.finditer(r"(Walls:.*?Goal:.*?)(?:\n\n|$)", generated_text, re.DOTALL))
    if coord_matches:
        return coord_matches[-1].group(1)  # Get the last match
    keywords = [
        "SOLVE: SUCCESS",
        "SOLVE: FAIL",
        "RECOGNIZE: SUCCESS",
        "RECOGNIZE: FAIL",
        "GENERATE: SUCCESS",
        "GENERATE: FAIL",
    ]
    after_keywords = []
    found_keyword = False
    lines = generated_text.splitlines()
    for line in lines:
        if not found_keyword and any(keyword in line for keyword in keywords):
            # We've found a keyword line, skip this line and start collecting after it
            found_keyword = True
            continue
        if found_keyword:
            after_keywords.append(line)
            after_keywords.append('\n')
            after_keywords.append('\n')
    temp_text = ''.join(after_keywords)
    return temp_text

def analyze_results(base_dir: str):
    """Analyze results across all conditions and save metrics."""
    results = []
    
    # Mapping from directory name parts to provider names
    dir_to_provider = {
        'claude': 'anthropic',
        'openai': 'openai',
        'llama': 'llama',
        'gemini': 'google'
    }

    machine_results_dir = Path(base_dir) / "machine_results"
    if not machine_results_dir.exists():
        print(f"Machine results directory not found: {machine_results_dir}")
        return
    
    result_dirs = sorted([d for d in machine_results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(result_dirs)} result directories to process.")

    for i, result_dir in enumerate(result_dirs):
        dir_name = result_dir.name
        encoding = None
        model_provider = None
        
        # Determine encoding and provider from directory name
        if dir_name.startswith('matrix_') or dir_name.startswith('coord_list_'):
            parts = dir_name.replace('_results', '').split('_', 1)
            encoding = parts[0]
            model_part = parts[1]
            model_provider = dir_to_provider.get(model_part)
        elif 'vision_results' in dir_name:
            encoding = 'vision'
            if 'opus' in dir_name:
                model_name = VISION_MODELS.get('opus')
                model_provider = LLM_MODELS.get(model_name)
            elif 'sonnet' in dir_name:
                model_name = VISION_MODELS.get('sonnet')
                model_provider = LLM_MODELS.get(model_name)

        if not encoding or not model_provider:
            print(f"Skipping directory with unknown format: {dir_name}")
            continue

        print(f"\n[{i+1}/{len(result_dirs)}] Processing {encoding.upper()} {model_provider.upper()} results from {dir_name}...")
        
        for size_dir in sorted(result_dir.glob("[0-9]x[0-9]")):
            size = tuple(map(int, size_dir.name.split('x')))
            
            for shape_dir in sorted(size_dir.glob("*")):
                if not shape_dir.is_dir() or shape_dir.name not in SHAPES:
                    continue
                    
                shape = shape_dir.name
                print(f"  - Processing {size_dir.name} / {shape}...")
                
                # Process each maze directory
                for maze_dir in sorted(shape_dir.iterdir()):
                    if not maze_dir.is_dir() or not maze_dir.name.endswith('.npy'):
                        continue
                        
                    # Process each trial file in the maze directory
                    trial_files = sorted(list(maze_dir.glob("*.txt")))
                    for j, trial_file in enumerate(trial_files):
                        print(f"    ({j+1}/{len(trial_files)}) Processing trial: {trial_file.relative_to(result_dir)}", end='\r')
                        try:
                            text = trial_file.read_text()
                            
                            # Check if generation was attempted
                            if "GENERATE:" not in text:
                                continue
                                
                            generated_text = extract_generated_maze(text)
                            if generated_text:
                                # Parse the generated maze based on encoding
                                if encoding == 'matrix':
                                    generated_maze = parse_matrix_maze_v3(generated_text)
                                else:
                                    generated_maze = parse_coordinate_maze(generated_text, size)
                                
                                if generated_maze is not None:
                                    # Find closest correct maze and compute distance
                                    closest_maze, distance = get_closest_correct_maze(
                                        generated_maze, shape, size
                                    )
                                    
                                    if closest_maze is not None:
                                        result_entry = {
                                            'encoding_type': encoding,
                                            'model_name': model_provider,  # Use the provider name as the key
                                            'maze_size': f"{size[0]}x{size[1]}",
                                            'maze_shape': shape,
                                            'trial_file': str(trial_file.relative_to(machine_results_dir)),
                                            'edit_distance': distance
                                        }
                                        results.append(result_entry)
                        except Exception as e:
                            print(f"\n    Error processing {trial_file}: {e}")
                            continue
                    print(" " * 100, end='\r') # Clear the line
                print(f"      -> Completed {len(results)} results so far for {shape} {size_dir.name}")
        print(f"[DONE] {result_dir.name} processed. Accumulated results: {len(results)}")
        print() # Final newline

    if not results:
        print("No results were found to analyze.")
        return

    # Save results
    output_path = Path(base_dir) / 'maze_edit_distances.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {output_path}")
    
    # Also save as CSV with statistics
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        csv_path = Path(base_dir) / 'maze_edit_distances.csv'
        df.to_csv(csv_path, index=False)
        
        # Print overall statistics
        print("\nOverall Statistics:")
        print(f"Total trials analyzed: {len(df)}")
        print(f"Average edit distance: {df['edit_distance'].mean():.3f}")
        print(f"Standard deviation: {df['edit_distance'].std():.3f}")
        print(f"Min edit distance: {df['edit_distance'].min():.3f}")
        print(f"Max edit distance: {df['edit_distance'].max():.3f}")
        
        # Print summary by model and encoding
        summary = df.groupby(['encoding_type', 'model_name', 'maze_shape'])['edit_distance'].agg(['mean', 'std', 'count'])
        print("\nEdit Distance Summary Statistics:")
        print(summary)
    except Exception as e:
        print(f"Error creating summary statistics: {e}")

if __name__ == "__main__":
    analyze_results("../../spiral")
