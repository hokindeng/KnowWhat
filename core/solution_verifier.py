import re
from pathlib import Path

import numpy as np

from .maze_generator import *
from .prompt_builder import *

def encode_maze(maze, encoding_type):
    if encoding_type == "matrix":
        return encode_standard_matrix_maze(maze)
    elif encoding_type == "coord_list":
        return encode_coordinate_list_maze(maze)
    elif encoding_type == "vision":
        return encode_standard_matrix_maze(maze)
    else:
        raise NotImplementedError

# RECOGNIZE 
def contains_word_case_insensitive(s, word_list):
    word_list = [re.escape(word) for word in word_list]
    pattern = r'\b(?:' + '|'.join(word_list) + r')\b'
    return re.search(pattern, s, re.IGNORECASE) is not None

def is_correct_recognize(response, shape):
    if shape == 'square':
        word_list = ['square', 'box', 'cube', 'symmetrical rectangle', 'right-angled quadrilateral', 'symmetric quadrilateral', "four-sided polygon"]
    elif shape == 'triangle':
        word_list = ['triangle', 'pyramid', 'trilateral', 'three-sided polygon', 'isosceles']
    elif shape == 'spiral':
        word_list = ['spiral', 'helix', 'coil', 'whorl', 'swirl', 'vortex']
    elif shape == 'cross':
        word_list = ['cross', 'times', 'multiplication', 'X', 'X-shape', 'crossed lines', 'crossing']
    elif shape == 'C':
        word_list = ['C', 'crescent', 'half-circle', 'curved line', 'semi-circle', 'semi circle', 'open curve']
    elif shape == 'Z':
        word_list = ['Z', 'zigzag', 'lightning bolt']

    return contains_word_case_insensitive(response, word_list)

# GENERATE

def is_correct_generate(response, encoding_type, size, shape, orig_maze):
    all_mazes = globals()[f"all_{shape}_path_mazes"](size)
    all_mazes_and_paths = []
    for maze in all_mazes:
        all_mazes_and_paths.extend(init_all_start_end(maze))
    all_maze_encodings = set()
    for maze in all_mazes_and_paths:
        all_maze_encodings.add(encode_maze(maze, encoding_type))
    
    all_maze_encodings.remove(encode_maze(orig_maze, encoding_type))

    for maze_encoding in all_maze_encodings:
        if maze_encoding in response:
            return True
    return False

def is_correct_solution(response, maze):
    # Find all coordinates that appear in the response.
    pattern = r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)'
    coordinates = re.findall(pattern, response)
    coordinates = [(int(x), int(y)) for x, y in coordinates]

    # As responses will usually talk about the solution before presenting
    # the final form, we will start from the last provided end point 
    # of the maze and work backwards until we arrive at the solution.
    start = tuple(np.argwhere(maze == POS)[0])
    end = tuple(np.argwhere(maze == END)[0])

    try:
        i = find_last_index(coordinates, end)
    except ValueError:
        return False

    while i > 0:
        curr_pos = coordinates[i]
        if curr_pos == start:
            return True
        next_pos = coordinates[i - 1]
        if next_pos in get_valid_moves(maze, curr_pos):
            i = i - 1
        else:
            break
    return False

def is_correct_solution_human(moves, maze):
    """
    Verifies if a sequence of directional moves correctly navigates from start to end in the maze.
    
    Args:
        moves: List of string directions ('up', 'down', 'left', 'right')
        maze: The maze as a numpy array where:
              POS = start position
              END = end position
              WALL = wall
              PATH = path
    
    Returns:
        bool: True if the moves successfully navigate from start to end, False otherwise
    """
    if not moves:
        return False
        
    # Find start and end positions
    start = tuple(np.argwhere(maze == POS)[0])
    end = tuple(np.argwhere(maze == END)[0])
    
    # Direction to coordinate change mapping
    direction_map = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }
    
    # Start at the beginning
    current_pos = start
    
    # Follow the moves
    for move in moves:
        move = move.lower()  # Normalize to lowercase
        
        # Skip invalid directions
        if move not in direction_map:
            continue
            
        dx, dy = direction_map[move]
        new_pos = (current_pos[0] + dx, current_pos[1] + dy)
        
        # Check if the move is valid (within bounds and not a wall)
        rows, cols = maze.shape
        if (0 <= new_pos[0] < rows and 0 <= new_pos[1] < cols and 
            maze[new_pos] != WALL):
            current_pos = new_pos
        # If invalid move, stay in place
    
    # Check if we reached the end
    return current_pos == end

def find_last_index(lst, element):
    return len(lst) - 1 - lst[::-1].index(element)

def get_valid_moves(maze, pos):
    x, y = pos
    rows = len(maze)
    cols = len(maze[0])
    
    # All 8 possible directions (up, down, left, right and diagonals)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),         (0, 1),
        (1, -1), (1, 0), (1, 1) 
    ]

    valid_moves = []
    for dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        
        # Check if the new coordinates are within bounds
        if 0 <= new_x < rows and 0 <= new_y < cols:
            if maze[new_x, new_y] != WALL:
                valid_moves.append((new_x, new_y))
    
    return valid_moves

def parse_coordinates(text):
    walls = re.findall(r"Walls: ([\d, ()]+)", text)
    empty = re.findall(r"Empty: ([\d, ()]+)", text)
    start = re.findall(r"Start: \((\d+, \d+)\)", text)
    end = re.findall(r"End: \((\d+, \d+)\)", text)

    walls = [tuple(map(int, coord.split(', '))) for coord in re.findall(r"\((\d+, \d+)\)", walls[0])]
    empty = [tuple(map(int, coord.split(', '))) for coord in re.findall(r"\((\d+, \d+)\)", empty[0])]
    start = tuple(map(int, start[0].split(', ')))
    end = tuple(map(int, end[0].split(', ')))

    return walls, empty, start, end

def parse_coordinate_maze(text, size):
    walls, empty, start, end = parse_coordinates(text)
    maze = np.full(size, -1)
    for x, y in walls:
        maze[x, y] = WALL
    for x, y in empty:
        maze[x, y] = PATH
    maze[start[0], start[1]] = POS
    maze[end[0], end[1]] = END
    return maze

def grade_prompts(version):
    parent_dir = Path(f"results/{version}/")
    for model in parent_dir.iterdir():
        grade_info = ""
        total = 0
        total_correct = 0
        total_size = {size_t.name: 0 for size_t in (parent_dir / model.name).iterdir() if not size_t.is_file()}
        total_size_correct = {size_t.name: 0 for size_t in (parent_dir / model.name).iterdir() if not size_t.is_file()}
        total_encoding = {"standard": 0, "unusual": 0, "coordinate": 0}
        total_encoding_correct = {"standard": 0, "unusual": 0, "coordinate": 0}
        total_shape = {shape: 0 for shape in SHAPES}
        total_shape_correct = {shape: 0 for shape in SHAPES}
        for size_t in (parent_dir / model.name).iterdir():
            if size_t.is_file():
                continue
            size = (int(size_t.name[0]), int(size_t.name[2]))
            for encoding in (parent_dir / model.name / size_t.name).iterdir():
                for shape in (parent_dir / model.name / size_t.name / encoding.name).iterdir():
                    for response_file in (parent_dir / model.name / size_t.name / encoding.name / shape.name).iterdir():
                        response_text = response_file.read_text()
                        # Get coordinate version of maze given in prompt so we can decode.
                        coord_prompt_text = Path(f"prompts/v1/diag/{size_t.name}/coordinate/{shape.name}/{response_file.name}").read_text()
                        maze = parse_coordinate_maze(coord_prompt_text, size)
                        is_correct = is_correct_solution(response_text, maze)

                        grade_info += f"{size_t.name}-{encoding.name}: {response_file.name}: {is_correct}\n"

                        total += 1
                        total_correct += is_correct
                        total_size[size_t.name] += 1
                        total_size_correct[size_t.name] += is_correct
                        total_encoding[encoding.name] += 1
                        total_encoding_correct[encoding.name] += is_correct
                        total_shape[shape.name] += 1
                        total_shape_correct[shape.name] += is_correct
        
        grade_info += "\n\n### SUMMARY ###\n"
        grade_info += f"Total: {total_correct}/{total}\n\n"
        grade_info += "Following show total correct for one variable across all others:\n"
        grade_info += "# SIZES #\n"
        for size, correct in total_size_correct.items():
            grade_info += f"{size}: {correct}/{total_size[size]}\n"
        grade_info += "# ENCODING #\n"
        for encoding, correct in total_encoding_correct.items():
            grade_info += f"{encoding}: {correct}/{total_encoding[encoding]}\n"
        grade_info += "# SHAPE #\n"
        for shape, correct in total_shape_correct.items():
            grade_info += f"{shape}: {correct}/{total_shape[shape]}\n"

        (parent_dir / model.name / "grade_info.txt").write_text(grade_info)

        print(f"{model.name} - total: {total_correct}/{total}")
        import csv

        # Define the path to the CSV file
        csv_path = parent_dir / model.name / "grade_info.csv"

        # Prepare the data for the CSV
        csv_data = [
            [model.name, "Total Correct", "Total"],
            ["Summary", total_correct, total],
        ]

        # Add size data to csv_data
        csv_data.append(["SIZES", "Correct", "Total"])
        for size, correct in total_size_correct.items():
            csv_data.append([size, correct, total_size[size]])

        # Add encoding data to csv_data
        csv_data.append(["ENCODING", "Correct", "Total"])
        for encoding, correct in total_encoding_correct.items():
            csv_data.append([encoding, correct, total_encoding[encoding]])

        # Add shape data to csv_data
        csv_data.append(["SHAPE", "Correct", "Total"])
        for shape, correct in total_shape_correct.items():
            csv_data.append([shape, correct, total_shape[shape]])

        # Write the csv_data to the CSV file
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)
        print(f"CSV file saved at {csv_path}")

if __name__ == "__main__":
    grade_prompts("only_one_per_folder")