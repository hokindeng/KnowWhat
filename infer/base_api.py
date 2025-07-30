"""
Base API class for maze solving experiments.
Provides common functionality to reduce code duplication across different API implementations.
"""

import os
import sys
import time
from pathlib import Path
import re
import numpy as np
from enum import Enum
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from core.prompt_builder import encode_standard_matrix_maze, encode_coordinate_list_maze
from core.maze_generator import *
from core.solution_verifier import get_valid_moves, is_correct_generate, is_correct_recognize
from core.prompts import *


class Status(Enum):
    SUCCESS = 1
    FAIL = 2
    CONTINUE = 3


class BaseAPISolver(ABC):
    """Base class for all API-based maze solvers."""
    
    def __init__(self, api_key: str = None):
        """Initialize the solver with API key from environment or parameter."""
        self.api_key = api_key or self._get_api_key()
        self.client = self._initialize_client()
        self.alert_count = 0
        
    @abstractmethod
    def _get_api_key(self) -> str:
        """Get API key from environment variables."""
        pass
        
    @abstractmethod
    def _initialize_client(self):
        """Initialize the API client."""
        pass
        
    @abstractmethod
    def _make_api_call(self, messages: list, **kwargs) -> str:
        """Make an API call and return the response text."""
        pass
    
    def encode_maze(self, maze: np.ndarray, encoding_type: str = 'matrix') -> str:
        """Encode the maze based on the specified encoding type."""
        if encoding_type == "matrix":
            return encode_standard_matrix_maze(maze)
        elif encoding_type == "coord_list":
            return encode_coordinate_list_maze(maze)
        else:
            raise NotImplementedError(f"Encoding type '{encoding_type}' not supported")
    
    def parse_and_update(self, response: str, maze: np.ndarray) -> Tuple[Status, np.ndarray]:
        """Parse the model response and update the maze state."""
        pattern = r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)'
        coordinates = re.findall(pattern, response)
        
        if not coordinates:
            return Status.FAIL, None
            
        coordinates = [(int(x), int(y)) for x, y in coordinates]
        # Take the last coordinate as the final move
        move = coordinates[-1]
        
        player_coord = np.argwhere(maze == POS)[0]
        player_coord = (player_coord[0], player_coord[1])
        valid_moves = get_valid_moves(maze, player_coord)
        
        if move in valid_moves:
            if maze[move] == END:
                return Status.SUCCESS, None
            else:
                new_maze = maze.copy()
                new_maze[player_coord[0], player_coord[1]] = PATH
                new_maze[move[0], move[1]] = POS
                return Status.CONTINUE, new_maze
        else:
            return Status.FAIL, None
    
    def get_initial_prompt(self, maze_string: str, encoding_type: str, use_diag: bool) -> str:
        """Get the appropriate initial prompt based on encoding type and diagonal movement."""
        if not use_diag:
            if encoding_type == "matrix":
                return INITIAL_SOLVE_PROMPT_1 + maze_string + INITIAL_SOLVE_PROMPT_2
            elif encoding_type == "coord_list":
                return INITIAL_SOLVE_PROMPT_COORD_1 + maze_string + INITIAL_SOLVE_PROMPT_2
        else:
            if encoding_type == "matrix":
                return INITIAL_SOLVE_PROMPT_DIAG_1 + maze_string + INITIAL_SOLVE_PROMPT_2
            elif encoding_type == "coord_list":
                return INITIAL_SOLVE_PROMPT_DIAG_COORD_1 + maze_string + INITIAL_SOLVE_PROMPT_2
    
    def solve_maze_with_api(self, maze: np.ndarray, encoding_type: str, use_diag: bool, 
                           size: Tuple[int, int], shape: str, result_path: str):
        """Main method to solve maze using the API."""
        try:
            messages = []
            status = Status.CONTINUE
            
            with open(result_path, "w") as file:
                i = 0
                while status == Status.CONTINUE:
                    time.sleep(1)  # Rate limiting
                    
                    # Encode the maze
                    maze_string = self.encode_maze(maze, encoding_type)
                    
                    # Get appropriate prompt
                    if i == 0:
                        prompt = self.get_initial_prompt(maze_string, encoding_type, use_diag)
                    else:
                        prompt = UPDATE_SOLVE_PROMPT_1 + maze_string + UPDATE_SOLVE_PROMPT_2
                    
                    messages.append({"role": "user", "content": prompt})
                    
                    # Make API call
                    assistant_reply = self._make_api_call(messages)
                    messages.append({"role": "assistant", "content": assistant_reply})
                    
                    # Log interaction
                    file.write(prompt)
                    file.write("\n")
                    file.write(assistant_reply)
                    file.write("\n")
                    
                    # Parse and update the maze
                    status, updated_maze = self.parse_and_update(assistant_reply, maze)
                    
                    if status == Status.SUCCESS:
                        file.write("\n\nSOLVE: SUCCESS")
                        break
                    elif status == Status.FAIL:
                        file.write("\n\nSOLVE: FAIL")
                        break
                    elif status == Status.CONTINUE:
                        maze = updated_maze
                        if i >= 15:  # Maximum iterations
                            file.write("\n\nSOLVE: FAIL")
                            break
                    else:
                        print("Unexpected status. Terminating.")
                        break
                    
                    i += 1
                
                # Recognition and generation phase
                messages.append({"role": "user", "content": RECOGNIZE_AND_GENERATE_PROMPT})
                assistant_reply = self._make_api_call(messages)
                messages.append({"role": "assistant", "content": assistant_reply})
                
                file.write(RECOGNIZE_AND_GENERATE_PROMPT)
                file.write("\n")
                file.write(assistant_reply)
                file.write("\n")
                
                # Check correctness
                generate_correct = is_correct_generate(assistant_reply, encoding_type, size, shape, maze)
                recognize_correct = is_correct_recognize(assistant_reply, shape)
                
                file.write("\n\nRECOGNIZE: " + ("SUCCESS" if recognize_correct else "FAIL"))
                file.write("\n\nGENERATE: " + ("SUCCESS" if generate_correct else "FAIL"))
                
        except Exception as e:
            with open(result_path, "w") as file:
                print(f"\n\nAlert, query did not run! Path: {result_path}")
                print(f'Alert Number: {self.alert_count}')
                print(f'Error: {str(e)}')
                self.alert_count += 1
                file.write(f"\n\nQuery did not run: {result_path}")
                file.write(f"\n\nError: {str(e)}")
                file.write("\n\nSOLVE: FAIL")
                file.write("\n\nRECOGNIZE: FAIL")
                file.write("\n\nGENERATE: FAIL")
    
    @staticmethod
    def load_npy_files(folder_path: str) -> list:
        """Load all .npy files from a folder and its subfolders."""
        all_file_data = []
        idx = 0
        
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith(".npy"):
                    file_path = os.path.join(root, filename)
                    try:
                        array_data = np.load(file_path)
                        file_data = {
                            'name': filename,
                            'id': idx,
                            'path': os.path.relpath(file_path, folder_path),
                            'data': array_data
                        }
                        idx += 1
                        all_file_data.append(file_data)
                    except Exception as e:
                        print(f"Could not load {filename}: {e}")
        
        return all_file_data
    
    def run_specific_experiments(self, size: Tuple[int, int], shape: str, encoding_type: str):
        """Run experiments for a specific size, shape, and encoding type."""
        folder_path = os.path.join(os.getcwd(), f'experiment_mazes/{size[0]}x{size[1]}/{shape}')
        npy_data = self.load_npy_files(folder_path)
        
        for file_info in npy_data:
            print(f"File ID: {file_info['id']}")
            print(f"File Name: {file_info['name']}")
            print(f"File Path: {file_info['path']}")
            print(f"Array Data:\n{file_info['data']}")
            
            # Determine if diagonal movement is allowed
            diag = shape not in ['square', 'C', 'spiral']
            
            # Determine number of samples based on size
            if size == (5, 5):
                samples = 30
            elif size == (7, 7):
                samples = 5
            elif size == (9, 9):
                samples = 3
            else:
                samples = 1
            
            # Run experiments
            for i in range(samples):
                output_dir = self._get_output_directory(encoding_type)
                output_path = os.path.join(
                    os.getcwd(),
                    f'{output_dir}/{size[0]}x{size[1]}/{shape}/{file_info["path"]}'
                )
                
                if not os.path.isdir(output_path):
                    os.makedirs(output_path)
                
                output_path = os.path.join(output_path, f'{i}.txt')
                
                # Skip if already completed
                if Path(output_path).is_file():
                    with open(output_path) as f:
                        contents = f.read()
                    if 'SOLVE:' in contents and 'did not run' not in contents:
                        continue
                
                print(f"Output Path: {output_path}")
                self.solve_maze_with_api(
                    file_info['data'], encoding_type, diag, size, shape, output_path
                )
    
    @abstractmethod
    def _get_output_directory(self, encoding_type: str) -> str:
        """Get the output directory name for results."""
        pass
    
    def run_all_experiments(self, encoding_types: list = None, sizes: list = None, shapes: list = None):
        """Run all experiments with specified parameters."""
        if encoding_types is None:
            encoding_types = ['matrix', 'coord_list']
        if sizes is None:
            sizes = [(5, 5), (7, 7)]
        if shapes is None:
            shapes = SHAPES
        
        print("Starting the maze-solving experiments...")
        
        for encoding_type in encoding_types:
            for size in sizes:
                for shape in shapes:
                    self.run_specific_experiments(size, shape, encoding_type)
        
        print("All experiments completed.") 