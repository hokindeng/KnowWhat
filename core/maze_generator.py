import os
import random
import numpy as np
from scipy.ndimage import zoom

# IMPORTANT: paths are 1, walls are 0
# Get all mazes of some size with paths of a given shape with:
# `all_{shape}_path_mazes(size)`

### SQUARES ###

# Return a list of all minimal bounding box squares up to a certain size.
def all_mbbox_squares(max_size):
    all_squares = []
    # Start with smallest possible square.
    curr_square = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) 
    curr_size = curr_square.shape[0]
    while curr_size <= max_size[0] and curr_size <= max_size[1]:
        all_squares.append(curr_square)
        curr_square = zoom(
            curr_square,
            ((curr_size + 1) / curr_size, (curr_size + 1) / curr_size),
            order=0
        )
        curr_size = curr_square.shape[0]
    return all_squares

# Generate all square paths in a maze 
def all_square_path_mazes(maze_size):
    paths = all_mbbox_squares(maze_size)
    return all_mazes_with_implanted_paths(maze_size, paths)

### CROSSES ###

def create_cross(size):
    diag_1 = np.eye(size, dtype=int)
    diag_2 = np.fliplr(np.eye(size, dtype=int))
    return diag_1 | diag_2

# Only allowing crosses with an uneven size to avoid a 2x2 block in the middle.
def all_mbbox_crosses(max_size):
    all_crosses = []
    # Start with smallest possible cross.
    curr_size = 3
    curr_cross = create_cross(curr_size)
    while curr_size <= max_size[0] and curr_size <= max_size[1]:
        all_crosses.append(curr_cross)
        curr_size += 2
        curr_cross = create_cross(curr_size)
    return all_crosses

def all_cross_path_mazes(maze_size):
    paths = all_mbbox_crosses(maze_size)
    return all_mazes_with_implanted_paths(maze_size, paths)


### SPIRAL ###

# Fill a square grid with clockwise spiral from top left.
def create_base_spiral(size):
    spiral = np.zeros((size, size), dtype=int)

    top, bottom = 0, size - 1
    left, right = 0, size - 1
    
    while top <= bottom and left <= right:
        # Fill the top row (left to right)
        for i in range(max(left - 1, 0), right + 1):
            spiral[top][i] = 1
        top += 2
        
        # Fill the right column (top to bottom)
        for i in range(top - 1, bottom + 1):
            spiral[i][right] = 1
        right -= 2
        
        if top <= bottom:
            # Fill the bottom row (right to left)
            for i in range(right + 1, left - 1, -1):
                spiral[bottom][i] = 1
            bottom -= 2
        
        if left <= right:
            # Fill the left column (bottom to top)
            for i in range(bottom + 1, top - 1, -1):
                spiral[i][left] = 1
            left += 2
    
    return spiral
 

# Creates all square spirals going both clockwise and anticlockwise in all
# four orientations.
def all_mbbox_spirals(max_size):
    all_spirals = []
    curr_size = 5
    while curr_size <= max_size[0] and curr_size <= max_size[1]:
        cw_base_spiral = create_base_spiral(curr_size)
        acw_base_spiral = np.fliplr(cw_base_spiral)
        for k in range(4):
            all_spirals.append(np.rot90(cw_base_spiral, k=k))
            all_spirals.append(np.rot90(acw_base_spiral, k=k))
        curr_size += 1
    return all_spirals

def all_spiral_path_mazes(maze_size):
    paths = all_mbbox_spirals(maze_size)
    return all_mazes_with_implanted_paths(maze_size, paths)

### TRIANGLE ###

# Create all triangles with base at bottom, pointing up.
def all_mbbox_bottom_base_triangles(max_size):
    all_triangles = []
    # Smallest base bottom triangle.
    curr_size = (3, 5)
    while curr_size[0] <= max_size[0] and curr_size[1] <= max_size[1]:
        triangle = np.zeros(curr_size, dtype=int)
        # Top row has single 1 in the middle.
        mid_point = curr_size[1] // 2
        triangle[0, mid_point] = 1
        # Bottom row is all 1's.
        triangle[-1, :] = 1
        # Intermediate rows have two 1's expanding out from top to bottom.
        for i in range(1, curr_size[0] - 1):
            triangle[i, mid_point - i] = 1
            triangle[i, mid_point + i] = 1
        all_triangles.append(triangle)
        curr_size = (curr_size[0] + 1, curr_size[1] + 2)
    return all_triangles

# Create all triangles with diagonal bases for all 4 orientations.
def all_mbbox_diag_base_traingles(max_size):
    all_triangles = []
    curr_size = 4
    while curr_size <= max_size[0] and curr_size <= max_size[1]:
        triangle = np.eye(curr_size, dtype=int)
        triangle[:, 0] = 1
        triangle[-1, :] = 1
        for k in range(4):
            all_triangles.append(np.rot90(triangle, k=k))
        curr_size += 1
    return all_triangles

def all_triangle_path_mazes(maze_size):
    paths = all_mbbox_bottom_base_triangles(maze_size)
    paths.extend(all_mbbox_diag_base_traingles(maze_size))
    return all_mazes_with_implanted_paths(maze_size, paths)


### C ###

def all_mbbox_Cs(max_size):
    all_cs = []
    # C where length = width.
    curr_c = np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]]) 
    curr_size = curr_c.shape[0]
    while curr_size <= max_size[0] and curr_size <= max_size[1]:
        all_cs.append(curr_c)
        curr_c = zoom(
            curr_c,
            ((curr_size + 1) / curr_size, (curr_size + 1) / curr_size),
            order=0
        )
        curr_size = curr_c.shape[0]
    
    # C where length > width.
    curr_c = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1]]) 
    while curr_c.shape[0] <= max_size[0] and curr_c.shape[1] <= max_size[1]:
        all_cs.append(curr_c)
        curr_c = zoom(
            curr_c,
            ((curr_c.shape[0] + 1) / curr_c.shape[0], (curr_c.shape[1] + 1) / curr_c.shape[1]),
            order=0
        )

    return all_cs

def all_C_path_mazes(maze_size):
    paths = all_mbbox_Cs(maze_size)
    return all_mazes_with_implanted_paths(maze_size, paths)

### U ###

def all_mbbox_Us(max_size):
    all_us = []
    # U where length = width.
    curr_u = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]]) 
    curr_size = curr_u.shape[0]
    while curr_size <= max_size[0] and curr_size <= max_size[1]:
        all_us.append(curr_u)
        curr_u = zoom(
            curr_u,
            ((curr_size + 1) / curr_size, (curr_size + 1) / curr_size),
            order=0
        )
        curr_size = curr_u.shape[0]
    
    # U where length > width.
    curr_u = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]]) 
    while curr_u.shape[0] <= max_size[0] and curr_u.shape[1] <= max_size[1]:
        all_us.append(curr_u)
        curr_u = zoom(
            curr_u,
            ((curr_u.shape[0] + 1) / curr_u.shape[0], (curr_u.shape[1] + 1) / curr_u.shape[1]),
            order=0
        )

    return all_us

def all_U_path_mazes(maze_size):
    paths = all_mbbox_Us(maze_size)
    return all_mazes_with_implanted_paths(maze_size, paths)

### Z ###

def create_Z(size):
    z = np.fliplr(np.eye(size, dtype=int))
    z[0, :] = 1
    z[-1, :] = 1
    return z

def all_mbbox_Zs(max_size):
    all_Zs = []
    # Start with smallest possible Z.
    curr_size = 4
    curr_cross = create_Z(curr_size)
    while curr_size <= max_size[0] and curr_size <= max_size[1]:
        all_Zs.append(curr_cross)
        curr_size += 1
        curr_cross = create_Z(curr_size)
    return all_Zs

def all_Z_path_mazes(maze_size):
    paths = all_mbbox_Zs(maze_size)
    return all_mazes_with_implanted_paths(maze_size, paths)

### N ###

def all_mbbox_Ns(max_size):
    return [np.rot90(z) for z in all_mbbox_Zs(max_size)]

def all_N_path_mazes(maze_size):
    paths = all_mbbox_Ns(maze_size)
    return all_mazes_with_implanted_paths(maze_size, paths)

### MAZES ###

WALL = 0
PATH = 1
POS = 2
END = 3

MIN_DISTANCE = 2

# Generate all mazes of a given size with each provided path placed into each
# possible position of the maze.
def all_mazes_with_implanted_paths(maze_size, paths):
    all_mazes = []
    for path in paths:
        assert path.shape[0] <= maze_size[0] and path.shape[1] <= maze_size[1]
        for i in range(0, maze_size[0] - path.shape[0] + 1):
            for j in range(0, maze_size[1] - path.shape[1] + 1):
                maze = np.zeros(maze_size, dtype=int)
                maze[i: i + path.shape[0], j: j + path.shape[1]] = path
                all_mazes.append(maze)
    return all_mazes

def distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def init_random_start_end(maze, k=1):
    path_indices = np.argwhere(maze == PATH)
    valid_pairs = []
    for p1 in path_indices:
        for p2 in path_indices:
            if distance(tuple(p1), tuple(p2)) >= MIN_DISTANCE:
                valid_pairs.append((tuple(p1), tuple(p2)))

    random_start_ends = random.sample(valid_pairs, k=k)
    random_mazes = []
    for start, end in random_start_ends:
        random_maze = np.copy(maze)
        random_maze[start] = POS
        random_maze[end] = END
        random_mazes.append(random_maze)

    if k == 1:
        return random_mazes[0]

    return random_mazes

def init_all_start_end(maze):
    path_indices = np.argwhere(maze == PATH)
    valid_pairs = []
    for p1 in path_indices:
        for p2 in path_indices:
            if tuple(p1) != tuple(p2):
                valid_pairs.append((tuple(p1), tuple(p2)))
    mazes = []
    for start, end in valid_pairs:
        path_maze = np.copy(maze)
        path_maze[start] = POS
        path_maze[end] = END
        mazes.append(path_maze)

    return mazes

SHAPES = ["square", "cross", "spiral", "triangle", "C", "Z"]

def get_sample_mazes(maze_size, shape, k):
    if shape not in SHAPES:
        raise NotImplemented("Shape not supported.")
    all_mazes = globals()[f"all_{shape}_path_mazes"](maze_size)
    sample_mazes = random.sample(all_mazes, k=k)
    mazes_with_path = []
    for maze in sample_mazes:
        mazes_with_path.append(init_random_start_end(maze))
    return mazes_with_path

def generate_maze_experiments(size, shape):
    size_shape_path = f"experiment_mazes/{str(size[0])}x{str(size[1])}/{shape}/"
    if not os.path.isdir(size_shape_path):
        os.makedirs(size_shape_path)
    if size == (5, 5):
        # Generate 5 samples of shape
        sample_mazes = get_sample_mazes(size, shape, k=5)
        for i, maze in enumerate(sample_mazes):
            np.save(size_shape_path + f"{str(size[0])}x{str(size[1])}_{shape}_{str(i)}.npy", maze)
    elif size == (7, 7):
        # Generate 30 samples of shape
        sample_mazes = get_sample_mazes(size, shape, k=30)
        for i, maze in enumerate(sample_mazes):
            np.save(size_shape_path + f"{str(size[0])}x{str(size[1])}_{shape}_{str(i)}.npy", maze)
    elif size == (9, 9):
        # Generate 50 samples of shape
        sample_mazes = get_sample_mazes(size, shape, k=50)
        for i, maze in enumerate(sample_mazes):
            np.save(size_shape_path + f"{str(size[0])}x{str(size[1])}_{shape}_{str(i)}.npy", maze)


'''
if __name__ == "__main__":
    for size in [(5, 5), (7, 7), (9, 9)]:
        for shape in SHAPES:
            generate_maze_experiments(size, shape)
'''