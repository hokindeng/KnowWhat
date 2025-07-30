import random
import os

import numpy as np

from .maze_generator import * 

# Turn 2D maze array into a string representation as follows:
# 011P 
# 0100 
# 011G 
def encode_standard_matrix_maze(maze):

    def symbol(x):
        if x == WALL:
            return "1"
        elif x == PATH:
            return "0"
        elif x == POS:
            return "P"
        elif x == END:
            return "G"

    return "\n".join(" ".join(map(symbol, row)) for row in maze)

# Turn 2D maze array into a string representation as follows:
# @**S 
# @*@@ 
# @**E 
def encode_unusual_matrix_maze(maze):

    def symbol(x):
        if x == WALL:
            return "@"
        elif x == PATH:
            return "*"
        elif x == POS:
            return "P"
        elif x == END:
            return "G"

    return "\n".join(" ".join(map(symbol, row)) for row in maze)

def str_of_coordinates(indices):
    return ", ".join(f"({p[0]}, {p[1]})" for p in indices)

# Turn 2D maze array into a list of coordinates for the walls,
# path, start, and end coordinates.
def encode_coordinate_list_maze(maze):
    walls_s = "Walls: " + str_of_coordinates(np.argwhere(maze == WALL))
    paths_s = "Empty: "  + str_of_coordinates(np.argwhere(maze == PATH))
    player_coord = np.argwhere(maze == POS)[0]
    start_s = "Player position: " + f"({player_coord[0]}, {player_coord[1]})"
    end_coord = np.argwhere(maze == END)[0]
    end_s = "Goal: " + f"({end_coord[0]}, {end_coord[1]})"
    return "\n".join((walls_s, paths_s, start_s, end_s))

# Get stats on number of paths present for each shape in a given maze size.
def print_num_paths(maze_size):
    print("For a maze size of " + str(maze_size) + " there are:")

    print(len(all_square_path_mazes(maze_size)), "squres")
    print(len(all_cross_path_mazes(maze_size)), "crosses")
    print(len(all_spiral_path_mazes(maze_size)), "spirals")
    print(len(all_triangle_path_mazes(maze_size)), "triangles")
    print(len(all_C_path_mazes(maze_size)), "Cs")
    print(len(all_Z_path_mazes(maze_size)), "Zs")

SHAPES = ["square", "cross", "spiral", "triangle", "C", "U", "Z", "N"]

def get_sample_mazes(maze_size, shape, k):
    if shape not in SHAPES:
        raise NotImplemented("Shape not supported.")
    all_mazes = globals()[f"all_{shape}_path_mazes"](maze_size)
    sample_mazes = random.sample(all_mazes, k=k)
    for maze in sample_mazes:
        init_random_start_end(maze)
    return sample_mazes

STANDARD_PROMPT_INSTRUCTION = """You are tasked with solving a maze. The maze is a 2D grid with walls, empty spaces, a start point and an end point. The maze has the following encoding: 
* Walls are represented by a '1'
* Empty spaces are represented by a '0' 
* The start point is represented by 'S'
* The end point is represented by 'E' 

Your task is to provide a step-by-step solution to move from the start point (S) to the end point (E), navigating only through empty spaces while avoiding walls. For reference, the coordinates of the top left corner of the maze are (0,0). You can move to any empty space that is adjacent to the current position: either up, down, left, right, or any diagonal.

Please provide a step-by-step solution from the start point (S) to the end point (E) using coordinates to specify each location visited. Your solution should:
1. Start at the coordinates of S
2. End at the coordinates of E
3. Only move through valid paths (empty spaces)

Present your solution as a series of coordinates, showing each step of the path from start to end. Do not use any code execution. Check each time if there is a wall. If there is a wall, then don't proceed. Plan using language. Explain your reasoning. You are allowed to take up to 50 steps.

Here is the maze you should solve:

"""

UNUSUAL_PROMPT_INSTRUCTION = """You are tasked with solving a maze. The maze is a 2D grid with walls, empty spaces, a start point and an end point. The maze has the following encoding: 
* Walls are represented by a '@'
* Empty spaces are represented by a '*' 
* The start point is represented by 'S'
* The end point is represented by 'E' 

Your task is to provide a step-by-step solution to move from the start point (S) to the end point (E), navigating only through empty spaces while avoiding walls. For reference, the coordinates of the top left corner of the maze are (0,0). You can move to any empty space that is adjacent to the current position: either up, down, left, right, or any diagonal.

Please provide a step-by-step solution from the start point (S) to the end point (E) using coordinates to specify each location visited. Your solution should:
1. Start at the coordinates of S
2. End at the coordinates of E
3. Only move through valid paths (empty spaces)

Present your solution as a series of coordinates, showing each step of the path from start to end. Do not use any code execution. Check each time if there is a wall. If there is a wall, then don't proceed. Plan using language. Explain your reasoning. You are allowed to take up to 50 steps.

Here is the maze you should solve:

"""

COORDINATE_PROMPT_INSTRUCTION = """You are tasked with solving a maze. The maze is a 2D grid with walls, empty spaces, a start point and an end point. The maze will be provided as a list of coordinates denoting the walls, a list of coordinates denoting the empty spaces, and coordinates for the start point and end point. 

Your task is to provide a step-by-step solution to move from the start point (S) to the end point (E), navigating only through empty spaces while avoiding walls. For reference, the coordinates of the top left corner of the maze are (0,0). You can move to any empty space that is adjacent to the current position: either up, down, left, right, or any diagonal.

Please provide a step-by-step solution from the start point (S) to the end point (E) using coordinates to specify each location visited. Your solution should:
1. Start at the coordinates of S
2. End at the coordinates of E
3. Only move through valid paths (empty spaces)

Present your solution as a series of coordinates, showing each step of the path from start to end. Do not use any code execution. Check each time if there is a wall. If there is a wall, then don't proceed. Plan using language. Explain your reasoning. You are allowed to take up to 50 steps.

Here is the maze you should solve:

"""

def generate_prompts(version, maze_size):
    random.seed(10)
    np.random.seed(10)

    parent_dir = f"prompts/{version}/{maze_size[0]}x{maze_size[1]}/"
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)

    for shape in SHAPES:
        all_mazes = globals()[f"all_{shape}_path_mazes"](maze_size)
        for i, maze in enumerate(all_mazes):
            random_mazes = init_random_start_end(maze, k=3)
            for j, random_maze in enumerate(random_mazes):
                maze_name = f"{shape}_{i}_{j}.txt"

                # Standard encoding.
                maze_path = parent_dir + f"standard/{shape}/"
                if not os.path.isdir(maze_path):
                    os.makedirs(maze_path)
                s_encoding = encode_standard_matrix_maze(random_maze)
                s_prompt = STANDARD_PROMPT_INSTRUCTION + s_encoding
                with open(maze_path + maze_name, "w") as file:
                    file.write(s_prompt)

                # Unusual encoding.
                maze_path = parent_dir + f"unusual/{shape}/"
                if not os.path.isdir(maze_path):
                    os.makedirs(maze_path)
                u_encoding = encode_unusual_matrix_maze(random_maze)
                u_prompt = UNUSUAL_PROMPT_INSTRUCTION + u_encoding
                with open(maze_path + maze_name, "w") as file:
                    file.write(u_prompt)

                # Coordinate list encoding. 
                maze_path = parent_dir + f"coordinate/{shape}/"
                if not os.path.isdir(maze_path):
                    os.makedirs(maze_path)
                c_encoding = encode_coordinate_list_maze(random_maze)
                c_prompt = COORDINATE_PROMPT_INSTRUCTION + c_encoding
                with open(maze_path + maze_name, "w") as file:
                    file.write(c_prompt)