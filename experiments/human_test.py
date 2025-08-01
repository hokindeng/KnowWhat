"""

This version worked.

Human Maze Solving Experiment
-----------------------------
This application creates a web interface for human participants to solve maze puzzles, 
mirroring the experiments conducted with AI systems. The program:

1. Displays mazes from the same dataset used in AI experiments
2. Allows participants to navigate through mazes of various sizes (5x5, 7x7) and shapes 
   (square, cross, spiral, triangle, C, Z)
3. Records performance data including:
   - Time taken to solve each maze
   - Complete movement history
   - Success/failure rate
4. Tests shape recognition by asking participants to identify maze shapes after solving
5. Saves results in a structured format matching the AI experiment results for direct comparison

Usage: Run with 'python human_test.py' to start the interface, which can be shared 
with participants through the generated public URL.
"""

import os
import time
import numpy as np
import gradio as gr
from pathlib import Path
import random
import json
from PIL import Image, ImageDraw, ImageFont
import re
import sys
from collections import defaultdict
import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import from core module
from core.maze_generator import *
from core.solution_verifier import get_valid_moves

# Worked
# Constants for maze representation
WALL = 0
PATH = 1
POS = 2
END = 3
# Additional constant for generation path
GEN_PATH = 1  # Same as PATH but will use a different color in the generation grid

# Constants for experiment parameters
SIZES = [(5, 5), (7, 7)]
SHAPES = ["square", "cross", "spiral", "triangle", "C", "Z"]
MAZES_PER_COMBINATION = 1  # Number of mazes to complete for each size/shape combination

# Variable for loop detection
last_save_timestamp = None
last_save_file = None
save_repeat_count = 0

def save_numpy_array_as_image(array: np.ndarray, cell_size: int = 0, target_size: int = 500,
                             font_scale: float = 0.4, is_generation: bool = False, show_coordinates: bool = False) -> Image.Image:
    """
    Converts a 2D NumPy array to a PIL Image with:
      - Distinct colors for each cell value with optimal contrast.
      - Thick borders around each cell.
      - Final image sized to approximately target_size (default 500px).
      - Optional: Text showing each cell's coordinates (disabled by default)

    Args:
        array: 2D NumPy array representing the maze
        cell_size: Size of each cell in pixels (default: 0, which means calculate based on target_size)
        target_size: Target size for the full image (default: 500px)
        font_scale: Scale factor for the font size as a proportion of cell size (default: 0.4)
        is_generation: If True, use a different color for PATH to make it more visible in generation mode
        show_coordinates: If True, show coordinates in each cell (default: False)

    Returns:
        PIL Image object
    """
    # 1) Map values to colors (improved for better contrast)
    if is_generation:
        # Colors for generation grid - make PATH light gray to show traveled path
        color_map = {
            0: (80, 80, 80),      # dark gray - wall (WALL=0)
            1: (200, 200, 200),   # light gray - path/trail (PATH=1)
            2: (0, 102, 204),     # blue - current position (POS=2)
            3: (204, 0, 0),       # darker red - end (END=3)
        }
    else:
        # Standard colors for maze solving
        color_map = {
            0: (80, 80, 80),      # dark gray - wall (WALL=0)
            1: (255, 255, 255),   # white - path (PATH=1)
            2: (0, 102, 204),     # blue - current position (POS=2)
            3: (204, 0, 0),       # darker red - end (END=3)
        }
    
    border_color = (0, 0, 0)  # black borders for maximum contrast
    border_width = 2  # border thickness

    height, width = array.shape

    # 2) Calculate the appropriate cell size
    if cell_size <= 0:
        # Calculate cell size based on target_size
        cell_size = min(target_size // width, target_size // height)

    # 3) Create the image with calculated dimensions
    img_width = width * cell_size
    img_height = height * cell_size
    image = Image.new("RGB", (img_width, img_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Initialize font only if we need to show coordinates
    font = None
    if show_coordinates:
        # 4) Calculate a dynamic font size based on font_scale (default 40% of cell size)
        font_size = max(10, int(cell_size * font_scale))
        
        # Try to use a common font or default to load_default
        try:
            # Try common fonts that might be available
            fonts_to_try = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf",
                             "LiberationSans-Regular.ttf", "Verdana.ttf"]
            font = None

            for font_name in fonts_to_try:
                try:
                    font = ImageFont.truetype(font_name, font_size)
                    break
                except IOError:
                    continue

            if font is None:
                # If specific fonts failed, try to use system font directory
                system_font_dirs = [
                    "/usr/share/fonts",  # Linux
                    "/Library/Fonts",    # macOS
                    "C:/Windows/Fonts"   # Windows
                ]

                for font_dir in system_font_dirs:
                    if os.path.exists(font_dir):
                        font_files = [f for f in os.listdir(font_dir)
                                     if f.endswith(('.ttf', '.otf')) and 'bold' not in f.lower()]
                        if font_files:
                            try:
                                font = ImageFont.truetype(os.path.join(font_dir, font_files[0]), font_size)
                                break
                            except:
                                pass
        except Exception:
            pass

        # Last resort: fall back to default font
        if font is None:
            font = ImageFont.load_default()

    # 5) Function to get text dimensions that works with different PIL versions (only used if showing coordinates)
    def get_text_dimensions(text, font):
        try:
            # For newer PIL versions
            return font.getbbox(text)[2:4]
        except AttributeError:
            try:
                # For PIL 8.0.0+
                return font.getsize(text)
            except AttributeError:
                # Fallback method for older PIL versions
                return font.getmask(text).size

    # 6) Draw each cell
    for row in range(height):
        for col in range(width):
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            cell_value = array[row, col]
            fill_color = color_map.get(cell_value, (255, 255, 255))

            # Draw the cell with border
            draw.rectangle(
                [x1, y1, x2, y2],
                fill=fill_color,
                outline=border_color,
                width=border_width
            )

            # Only draw coordinates if requested
            if show_coordinates and font is not None:
                # Write the cell coordinates in the center
                text = f"({row},{col})"
                text_width, text_height = get_text_dimensions(text, font)
                text_x = x1 + (cell_size - text_width) // 2
                text_y = y1 + (cell_size - text_height) // 2

                # Text color based on background for better contrast
                luminance = 0.299 * fill_color[0] + 0.587 * fill_color[1] + 0.114 * fill_color[2]
                text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

                draw.text((text_x, text_y), text, fill=text_color, font=font)

    return image

def load_npy_files(folder_path):
    """Load .npy maze files from the specified folder."""
    all_file_data = []
    idx = 0
    folder_path = Path(folder_path)
    
    # Traverse the folder and its subfolders
    for file_path in folder_path.rglob("*.npy"):
        try:
            array_data = np.load(file_path)  # Keep as numpy array for easier processing later
            file_data = {
                'name': file_path.name,
                'id': idx,
                'path': str(file_path.relative_to(folder_path)),  # Store relative path
                'data': array_data  # Store the array data
            }
            idx += 1
            all_file_data.append(file_data)
        except Exception as e:
            print(f"Could not load {file_path.name}: {e}")
    return all_file_data

class MazeExperiment:
    """Class to manage the maze experiment."""
    
    def __init__(self, results_dir="data/human_results", maze_dir="data/experiment_mazes"):
        """Initialize the maze experiment.
        
        Args:
            results_dir: Directory to save experiment results
            maze_dir: Directory containing maze files
        """
        # Get the project root directory (parent of experiments)
        project_root = Path(__file__).parent.parent
        
        # Make paths relative to project root
        self.results_dir = project_root / results_dir
        self.maze_dir = project_root / maze_dir
        self.current_maze = None
        self.current_file_info = None
        self.current_size = None
        self.current_shape = None
        self.start_time = None
        self.maze_complete = False
        self.experiment_complete = False
        self.moves = []
        self.current_phase = "welcome"  # Start with welcome phase
        self.participant_id = None  # Start with no participant ID
        
        # For coordinate-based generation
        self.generation_coords = []
        self.start_position = None
        self.end_position = None
        
        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking of completed mazes for each size/shape combination
        self.combination_counts = defaultdict(int)
        
        # Initialize the completed combinations list
        self.completed_combinations = []
        
        # Calculate total mazes in the experiment
        self.mazes_per_combination = MAZES_PER_COMBINATION
        self.total_mazes = len(SIZES) * len(SHAPES) * self.mazes_per_combination
    
    def set_participant_id(self, participant_id):
        """Set the participant ID and create their results folder.
        
        Args:
            participant_id: String participant identifier
            
        Returns:
            Tuple of (maze_image, message, phase, progress)
        """
        if not participant_id or participant_id.strip() == "":
            # Return error if ID is empty
            blank_maze = np.ones((5, 5), dtype=np.int8)
            return save_numpy_array_as_image(blank_maze), "Please enter your participant ID.", "welcome", "Not started yet"
        
        # Set the participant ID
        self.participant_id = participant_id.strip()
        
        # Create participant's results directory
        participant_dir = self.results_dir / self.participant_id
        participant_dir.mkdir(parents=True, exist_ok=True)
        
        # Update the results dir to the participant's folder
        self.results_dir = participant_dir
        
        # Reset progress counters for this participant
        self.combination_counts = defaultdict(int)
        self.completed_combinations = []
        self.experiment_complete = False
        
        # Force complete reset of generation data
        self.generation_grid = None
        self.generation_coords = []
        self.start_position = None
        self.end_position = None
        
        # Now load the first maze
        self.current_phase = "solve"
        return self.load_next_combination()
        
    def load_random_maze(self, size, shape):
        """Load a random maze of the specified size and shape.
        
        Args:
            size: Tuple of (height, width)
            shape: String representing the maze shape
            
        Returns:
            Tuple of (maze_image, message, phase, progress)
        """
        # Clear any existing generation grid when loading a new maze
        self.clear_generation_grid()
        
        # Update current size and shape
        self.current_size = size
        self.current_shape = shape
        
        # Reset maze state
        self.maze_complete = False
        self.moves = []
        self.start_time = time.time()
        
        try:
            # Get list of maze files for the specified size and shape
            size_str = f"{size[0]}x{size[1]}"
            
            # Check the nested directory structure
            shape_dir = self.maze_dir / size_str / shape
            if not shape_dir.exists():
                return None, f"Directory not found: {shape_dir}", "error", self.get_progress()
                
            # Get all maze files in the shape directory
            maze_files = []
            for file in shape_dir.iterdir():
                if file.suffix == ".npy":
                    maze_files.append(file.name)
            
            if not maze_files:
                return None, f"No maze files found in {shape_dir}", "error", self.get_progress()
            
            # Select a random maze file
            maze_file = random.choice(maze_files)
            maze_path = shape_dir / maze_file
            
            # Load the maze
            self.current_maze = np.load(maze_path)
            
            # Extract file information for tracking completion
            self.current_file_info = {
                "file": maze_file,
                "size": size_str,
                "shape": shape
            }
            
            # Calculate progress information
            completed_mazes = self.combination_counts[(size_str, shape)]
            total_combinations = len(SIZES) * len(SHAPES)
            completed_combinations = len(self.completed_combinations)
            
            # Create progress message
            progress_msg = "Solve Task Instructions:\n" \
                          "- Navigate through the maze from the start (blue) to the end (red)\n" \
                          "- Use the directional buttons to move (Up, Down, Left, Right, and diagonals)\n" \
                          "- Try to complete the maze with the fewest moves possible\n" \
                          "- The maze follows a specific pattern - try to notice it as you solve"
            
            # Return the maze image, progress message, phase, and progress
            return self.render_maze(), progress_msg, "solve", self.get_progress()
        except Exception as e:
            return None, f"Error loading maze: {str(e)}", "error", self.get_progress()
    
    def load_next_combination(self):
        """Load the next maze combination in the experiment sequence.
        
        If all combinations have been completed for all mazes, mark the experiment as complete.
        
        Returns:
            Tuple of (maze_image, message, phase, progress)
        """
        # Ensure any existing generation grid is cleared 
        self.clear_generation_grid()
        
        # Check if any combinations need more mazes
        available_combinations = []
        
        for size in SIZES:
            size_str = f"{size[0]}x{size[1]}"
            for shape in SHAPES:
                combo_key = (size_str, shape)
                if self.combination_counts[combo_key] < MAZES_PER_COMBINATION:
                    if combo_key not in self.completed_combinations:
                        available_combinations.append((size, shape))
        
        if not available_combinations:
            self.experiment_complete = True
            return None, "Experiment complete! Thank you for participating.", "complete", self.get_progress()
        
        # Choose the next combination
        next_combination = available_combinations[0]  # Take the first available
        
        # Load a random maze for this combination
        return self.load_random_maze(*next_combination)
    
    def process_move(self, direction):
        """Process a move in the maze.
        
        Args:
            direction (str): The direction to move in ('up', 'down', 'left', 'right').
            
        Returns:
            Tuple of (maze_image, message, phase, progress)
        """
        if self.current_maze is None:
            return self.load_random_maze(self.current_size, self.current_shape)
        
        # Get the current position
        i, j = np.where(self.current_maze == 2)
        if len(i) == 0:
            return self.render_maze(), "Invalid maze state. No current position found.", self.current_phase, self.get_progress()
        i, j = i[0], j[0]
        
        # Calculate the new position
        new_i, new_j = i, j
        
        # Handle all possible directions including diagonals
        if direction == 'up':
            new_i -= 1
        elif direction == 'down':
            new_i += 1
        elif direction == 'left':
            new_j -= 1
        elif direction == 'right':
            new_j += 1
        # Handle diagonal movements
        elif direction == 'up-left':
            new_i -= 1
            new_j -= 1
        elif direction == 'up-right':
            new_i -= 1
            new_j += 1
        elif direction == 'down-left':
            new_i += 1
            new_j -= 1
        elif direction == 'down-right':
            new_i += 1
            new_j += 1
        else:
            # Unknown direction - do nothing
            return self.render_maze(), f"Unknown direction: {direction}", self.current_phase, self.get_progress()
        
        # Check if the new position is valid
        if new_i < 0 or new_i >= self.current_maze.shape[0] or new_j < 0 or new_j >= self.current_maze.shape[1]:
            # Moving outside the maze counts as a failure just like hitting a wall
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Save the results with failed status
            result_file = self.save_results(elapsed_time, failed=True)
            
            # Clear any existing generation grid to avoid persistence between mazes
            self.clear_generation_grid()
            
            # Mark as complete and transition to recognition phase
            self.maze_complete = True
            self.current_phase = "recognize"
            
            return self.render_maze(), "Out of bounds! Moving to shape recognition. What shape is formed by the non-black squares of the maze? Please select the shape you see from the dropdown, then click on Submit Recognition button.", "recognize", self.get_progress()
            
        # Check if the new position is a wall (0)
        if self.current_maze[new_i, new_j] == WALL:
            # Wall collision counts as a failure but now jumps to recognize phase
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Mark as complete and transition to recognition phase
            self.maze_complete = True
            self.current_phase = "recognize"
            # Save the results with failed status
            result_file = self.save_results(elapsed_time, failed=True)
            
            # Clear any existing generation grid to avoid persistence between mazes
            self.clear_generation_grid()
            
            return self.render_maze(), "Wall hit! Moving to shape recognition. What shape is formed by the non-black squares of the maze? Please select the shape you see from the dropdown, then click on Submit Recognition button.", "recognize", self.get_progress()
            
        # Check if the new position is the end (3)
        if self.current_maze[new_i, new_j] == END:
            # Move to the end
            self.current_maze[i, j] = PATH  # Set old position to path
            self.current_maze[new_i, new_j] = POS  # Set the new position to player
            
            # Calculate elapsed time
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Save the results
            result_file = self.save_results(elapsed_time)
            
            # Clear any existing generation grid to avoid persistence between mazes 
            self.clear_generation_grid()
            
            # Mark as complete and transition to recognition phase
            self.maze_complete = True
            self.current_phase = "recognize"
            return self.render_maze(), "Maze complete! What shape is formed by the non-black squares of the maze? Please select the shape you see from the dropdown, then click on Submit Recognition button.", "recognize", self.get_progress()
                
        # Move to the new position if it's a path (1)
        self.current_maze[i, j] = PATH  # Clear the current position (set to path)
        self.current_maze[new_i, new_j] = POS  # Set the new position to player
        
        # Record the move
        self.moves.append(direction)
        
        return self.render_maze(), f"Moved {direction}. Keep going!", self.current_phase, self.get_progress()
    
    def process_complete_maze(self):
        """Process a completed maze and prepare for results review.
        
        Returns:
            Tuple of (maze_image, message, phase, progress)
        """
        # This method is now only used for showing results after a maze is failed
        # and has been modified to only provide feedback, not to advance to the next maze
        
        # Display appropriate message based on if the maze was failed or completed
        last_maze_failed = False
        
        # Check if we have result files to determine if the last maze was failed
        result_files = []
        for file_path in self.results_dir.rglob("*.json"):
            if file_path.name.startswith(self.participant_id):
                result_files.append(file_path)
        
        if result_files:
            try:
                file_path = sorted(result_files, key=os.path.getmtime)[-1]
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    last_maze_failed = data.get("failed", False)
            except Exception as e:
                print(f"Error checking failure status: {e}")
        
        # Get the current maze image
        img = self.render_maze()
        msg = "Continue solving the maze."
        if last_maze_failed:
            msg = "Maze failed (wall collision). Try again."
            
        return img, msg, self.current_phase, self.get_progress()
    
    def is_experiment_complete(self):
        """Check if all mazes in the experiment have been completed."""
        for size in SIZES:
            size_str = f"{size[0]}x{size[1]}"
            for shape in SHAPES:
                combo_key = (size_str, shape)
                if self.combination_counts[combo_key] < MAZES_PER_COMBINATION:
                    return False
        return True
        
    def load_next_maze(self):
        """Load the next maze based on current progress."""
        # Find the next combination that needs more completions
        next_combination = None
        for size in SIZES:
            size_str = f"{size[0]}x{size[1]}"
            for shape in SHAPES:
                if self.combination_counts[(size_str, shape)] < MAZES_PER_COMBINATION:
                    next_combination = (size, shape)
                    break
            if next_combination:
                break
                
        if next_combination:
            size, shape = next_combination
            self.load_random_maze(size, shape)
        else:
            # All combinations completed
            self.current_maze = None
    
    def save_results(self, elapsed_time, failed=False):
        """Save the results to a file.
        
        Args:
            elapsed_time: Time elapsed during maze solving
            failed: Boolean indicating if the maze was failed
        """
        if not self.current_file_info or not self.participant_id:
            return
        
        global last_save_timestamp, last_save_file, save_repeat_count
        
        # Create a unique filename
        timestamp = int(time.time())
        filename = self.results_dir / f"{self.participant_id}_{self.current_file_info['size']}_{self.current_file_info['shape']}_{timestamp}.json"
        
        # DESYNC DETECTION: If we're saving to the same file repeatedly, we have a loop
        if last_save_file and last_save_file == str(filename):
            save_repeat_count += 1
            print(f"WARNING: Same file saved repeatedly - count: {save_repeat_count}")
            
            if save_repeat_count >= 3:
                print("CRITICAL ERROR: Stuck in a save loop!")
                print("Forcing phase reset to break cycle")
                # Reset counter
                save_repeat_count = 0
                
                # Force phase reset
                self.force_phase_reset()
                
                # Create a different filename by adding a suffix
                timestamp = int(time.time())
                filename = self.results_dir / f"{self.participant_id}_{self.current_file_info['size']}_{self.current_file_info['shape']}_{timestamp}_recovery.json"
        else:
            # New file, reset counter
            save_repeat_count = 0
            
        # Update tracking variables
        last_save_timestamp = timestamp
        last_save_file = str(filename)
        
        # Prepare data to save
        data = {
            "participant_id": self.participant_id,
            "maze_file": self.current_file_info['file'],
            "maze_type": {
                "size": self.current_file_info['size'],
                "shape": self.current_file_info['shape']
            },
            "moves": self.moves,
            "total_moves": len(self.moves),
            "completion_time": elapsed_time,
            "timestamp": timestamp,
            "maze_complete": self.maze_complete,
            "failed": failed,
            "phase": self.current_phase  # Include current phase for diagnostics
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Results saved to {filename}")
        
        # Return the saved file path to verify it was created
        return filename
    
    def skip_maze(self):
        """Skip the current maze and save as not completed."""
        if self.current_maze is not None:
            self.save_results(0)
            self.maze_complete = True
            return save_numpy_array_as_image(self.current_maze), "Maze skipped. Load a new maze.", "solve", self.get_progress()
        return None, "No maze loaded to skip.", "solve", self.get_progress()
    
    def ask_shape_recognition(self):
        """Ask the participant to recognize the shape of the maze."""
        if self.maze_complete and self.current_maze is not None:
            return "What shape do you think this maze was designed to look like? Please enter any shape that comes to mind."
        return ""
    
    def submit_shape_recognition(self, recognized_shape):
        """Submit the participant's shape recognition answer."""
        # Log the current phase
        print(f"submit_shape_recognition called in phase: {self.current_phase}")
        
        # Verify we're in the correct phase before proceeding
        if self.current_phase != "recognize":
            print(f"WARNING: submit_shape_recognition called in incorrect phase: {self.current_phase}")
            return self.render_maze(), f"Error: Not in recognition phase (current phase: {self.current_phase}). Please refresh the page and try again.", self.current_phase, self.get_progress(), "Current path: []"
            
        if not recognized_shape or not self.maze_complete:
            return save_numpy_array_as_image(self.current_maze), "Please solve the maze first and select a shape before submitting.", "recognize", self.get_progress(), "Current path: []"
            
        # Process the recognized shape
        # With dropdown, we already have the clean value - no need for stripping or lowercase conversion
        # But we'll keep the lowercase check for consistency in data storage
        recognized_shape_lower = recognized_shape.lower()
        
        try:
            # Update the saved results file with the recognized shape
            result_files = []
            for file_path in self.results_dir.rglob("*.json"):
                if file_path.name.startswith(self.participant_id):
                    try:
                        # Verify the file is valid JSON before adding it
                        with open(file_path, 'r') as f:
                            json.load(f)
                        result_files.append(file_path)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON file: {file_path}")
                        continue
            
            # Sort by creation time and get the most recent
            if result_files:
                file_path = sorted(result_files, key=os.path.getmtime)[-1]
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Add the recognized shape
                    data["recognized_shape"] = recognized_shape
                    # Check if the recognized shape matches the actual shape
                    # With dropdown, this will always be a predefined shape
                    data["recognition_correct"] = (recognized_shape_lower == self.current_shape.lower())
                    data["is_predefined_shape"] = True
                    
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    # Update phase FIRST, to ensure all subsequent operations are in the correct phase
                    self.current_phase = "generate"
                    print("Phase changed to: generate")
                    
                    # Clear any existing generation grid to ensure we create a fresh one
                    self.clear_generation_grid()
                    
                    # Create a fresh generation grid - fix the error by handling the return type
                    self.create_generation_grid()
                    
                    # Get the generation grid image with coordinates shown - ensure this is an image, not a tuple
                    generation_img = None
                    if hasattr(self, 'generation_grid') and self.generation_grid is not None:
                        generation_img = save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True)
                    else:
                        print("ERROR: generation_grid is None, creating a blank image")
                        # Create a blank image as fallback
                        blank_grid = np.zeros((5, 5), dtype=np.int8)
                        generation_img = save_numpy_array_as_image(blank_grid, is_generation=True, show_coordinates=True)
                    
                    # Prepare feedback message with clear instructions for the coordinate input
                    feedback = f"Thank you for describing the shape you saw in the maze."
                    
                    # Update message for coordinate-based entry with clear instructions for the new interface
                    message = f"""{feedback}
Now, draw the same shape, but it's not the maze that you have just solved, by following these steps:
1. Enter your own Start Position (blue point) in the format row,col (for example: 2,3). This start position is not the same as the start position in the maze that you have just solved.
2. Enter your own End Position (red point) in the same format. This end position is not the same as the end position in the maze that you have just solved.
3. Add intermediate points to outline the path of your maze. Use as many points as needed to show your design clearly. This shouldn't be the same as the path in the maze that you have just solved.
4. Important: Create a brand‑new path. Do NOT copy the exact maze in the previous problem you have solved—use the grid to invent your own route with similar shape.
5. When your shape is complete, click "Submit Generated Shape."
The coordinates are visible on the grid to help you place your points accurately."""
                    # When transitioning to generation phase, always start with a fresh empty path text
                    return generation_img, message, "generate", self.get_progress(), "Current path: []"
                except Exception as e:
                    print(f"Error processing JSON file {file_path}: {e}")
                    # Create a new result file if the existing one is corrupted
                    saved_file = self.save_results(0)
                    # Try again with the newly created file
                    return self.submit_shape_recognition(recognized_shape)
            else:
                # If no valid result files found, create a new one with the current state
                saved_file = self.save_results(0)
                
                # Now attempt to open and update the file we just created
                try:
                    with open(saved_file, 'r') as f:
                        data = json.load(f)
                    
                    # Add the recognized shape
                    data["recognized_shape"] = recognized_shape
                    # Check if the recognized shape matches the actual shape
                    # With dropdown, this will always be a predefined shape
                    data["recognition_correct"] = (recognized_shape_lower == self.current_shape.lower())
                    data["is_predefined_shape"] = True
                    
                    with open(saved_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    # Update phase FIRST, to ensure all subsequent operations are in the correct phase
                    self.current_phase = "generate"
                    print("Phase changed to: generate")
                    
                    # Clear any existing generation grid to ensure we create a fresh one
                    self.clear_generation_grid()
                    
                    # Create a fresh generation grid
                    self.create_generation_grid()
                    
                    # Get the generation grid image with coordinates shown
                    generation_img = None
                    if hasattr(self, 'generation_grid') and self.generation_grid is not None:
                        generation_img = save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True)
                    else:
                        # Create a blank image as fallback
                        blank_grid = np.zeros((5, 5), dtype=np.int8)
                        generation_img = save_numpy_array_as_image(blank_grid, is_generation=True, show_coordinates=True)
                    
                    # Update message for coordinate-based entry with clear instructions
                    message = """Thank you for describing the shape you saw in the maze.

Draw the same shape by following these steps:
1. First, enter the Start Position (blue point) coordinates in the format 'row,col' (e.g., '2,3')
2. Then, enter the End Position (red point) coordinates
3. Finally, add intermediate points to complete the shape's outline
4. When finished, click on 'Submit Generated Shape'

The coordinates are visible on the grid to help you place your points accurately."""
                    return generation_img, message, "generate", self.get_progress(), "Current path: []"
                except Exception as e:
                    print(f"Error updating newly created result file: {e}")
                    return save_numpy_array_as_image(self.current_maze), "An error occurred. Please try again.", "recognize", self.get_progress(), "Current path: []"
        except Exception as e:
            print(f"Error in shape recognition: {e}")
            return save_numpy_array_as_image(self.current_maze), "An error occurred. Please try again.", "recognize", self.get_progress(), "Current path: []"
    
    def create_generation_grid(self):
        """Create a blank grid for maze generation."""
        if self.current_size is None:
            print("ERROR: Cannot create generation grid - no current size set")
            return None
            
        try:
            # Create a blank grid with the same dimensions as the current maze
            height, width = self.current_maze.shape
            grid = np.zeros((height, width), dtype=np.int8)  # Start with all walls (0)
            
            # Reset the generation coordinates
            self.generation_coords = []
            self.start_position = None
            self.end_position = None
            
            # Set initial grid
            self.generation_grid = grid
            
            print(f"Created generation grid with dimensions {height}x{width}")
            # Note: Instead of returning a tuple, we now only set the grid and return None
            # The image will be created where needed
            return None
        except Exception as e:
            print(f"ERROR creating generation grid: {e}")
            return None
    
    def set_start_position(self, coord_text):
        """Set the start position on the generation grid.
        
        Args:
            coord_text: String in format "row,col" e.g. "2,3"
            
        Returns:
            Tuple of (maze_image, message, path_text)
        """
        if not hasattr(self, 'generation_grid') or self.generation_grid is None:
            # Create the generation grid if it doesn't exist
            img = self.create_generation_grid()
            self.generation_coords = []
            return img, "Generation grid created. Enter start position first.", self._format_path_display()
        
        # Parse the coordinate input
        try:
            parts = coord_text.strip().split(',')
            if len(parts) != 2:
                return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), "Invalid format. Please use 'row,col' format (e.g. '2,3').", self._format_path_display()
            
            row = int(parts[0])
            col = int(parts[1])
            
            # Validate coordinates are within grid bounds
            height, width = self.generation_grid.shape
            if row < 0 or row >= height or col < 0 or col >= width:
                return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), f"Coordinates out of bounds. Valid range: (0-{height-1}, 0-{width-1}).", self._format_path_display()
            
            # If there's already a start position, clear it
            if self.start_position:
                old_row, old_col = self.start_position
                # Clear old start position only if it's not part of the path
                if self.generation_grid[old_row, old_col] == POS:
                    self.generation_grid[old_row, old_col] = 0
            
            # Set the new start position
            self.generation_grid[row, col] = POS  # Set as current position (2) - blue
            self.start_position = (row, col)
            
            # Add to generation coords if not already in it
            if not self.generation_coords or self.generation_coords[0] != (row, col):
                self.generation_coords = [(row, col)] + self.generation_coords[1:] if self.generation_coords else [(row, col)]
            
            return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), f"Start point set at ({row},{col}). Now set the end position or add intermediate points.", self._format_path_display()
            
        except ValueError:
            # Handle non-integer inputs
            return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), "Invalid input. Please enter integers in 'row,col' format (e.g. '2,3').", self._format_path_display()
        except Exception as e:
            print(f"Error processing coordinate: {e}")
            return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), f"Error: {str(e)}", self._format_path_display()
    
    def set_end_position(self, coord_text):
        """Set the end position on the generation grid.
        
        Args:
            coord_text: String in format "row,col" e.g. "2,3"
            
        Returns:
            Tuple of (maze_image, message, path_text)
        """
        if not hasattr(self, 'generation_grid') or self.generation_grid is None:
            # Create the generation grid if it doesn't exist
            img = self.create_generation_grid()
            self.generation_coords = []
            return img, "Generation grid created. Please set start position first.", self._format_path_display()
        
        # Must have a start position before setting end position
        if not self.start_position:
            return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), "Please set start position first.", self._format_path_display()
        
        # Parse the coordinate input
        try:
            parts = coord_text.strip().split(',')
            if len(parts) != 2:
                return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), "Invalid format. Please use 'row,col' format (e.g. '2,3').", self._format_path_display()
            
            row = int(parts[0])
            col = int(parts[1])
            
            # Validate coordinates are within grid bounds
            height, width = self.generation_grid.shape
            if row < 0 or row >= height or col < 0 or col >= width:
                return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), f"Coordinates out of bounds. Valid range: (0-{height-1}, 0-{width-1}).", self._format_path_display()
            
            # If there's already an end position, clear it
            if self.end_position:
                old_row, old_col = self.end_position
                # Clear old end position only if it's not part of the path
                if self.generation_grid[old_row, old_col] == END:
                    self.generation_grid[old_row, old_col] = 0
            
            # Make sure the end position is not the same as the start
            if (row, col) == self.start_position:
                return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), "End position cannot be the same as the start position.", self._format_path_display()
            
            # Set the new end position
            self.generation_grid[row, col] = END  # Set as end position (3) - red
            self.end_position = (row, col)
            
            # Add to generation coords if not already in it
            if len(self.generation_coords) > 1:
                # If we have more than start point, ensure end is the last point 
                if self.generation_coords[-1] != (row, col):
                    # Remove if already in the list
                    self.generation_coords = [p for p in self.generation_coords if p != (row, col)]
                    # Add to the end
                    self.generation_coords.append((row, col))
            else:
                # Only have start point, add end point
                self.generation_coords.append((row, col))
            
            return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), f"End point set at ({row},{col}). Now add intermediate points to complete your shape.", self._format_path_display()
            
        except ValueError:
            # Handle non-integer inputs
            return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), "Invalid input. Please enter integers in 'row,col' format (e.g. '2,3').", self._format_path_display()
        except Exception as e:
            print(f"Error processing coordinate: {e}")
            return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), f"Error: {str(e)}", self._format_path_display()
    
    def add_coordinate(self, coord_text):
        """Add a coordinate to the generation grid.
        
        Args:
            coord_text: String in format "row,col" e.g. "2,3"
            
        Returns:
            Tuple of (maze_image, message, path_text)
        """
        if not hasattr(self, 'generation_grid') or self.generation_grid is None:
            # Create the generation grid if it doesn't exist
            img = self.create_generation_grid()
            self.generation_coords = []
            return img, "Generation grid created. Please set start and end positions first.", self._format_path_display()
        
        # Must have a start position before adding points
        if not self.start_position:
            return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), "Please set start position first.", self._format_path_display()
            
        # Parse the coordinate input
        try:
            parts = coord_text.strip().split(',')
            if len(parts) != 2:
                return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), "Invalid format. Please use 'row,col' format (e.g. '2,3').", self._format_path_display()
            
            row = int(parts[0])
            col = int(parts[1])
            
            # Validate coordinates are within grid bounds
            height, width = self.generation_grid.shape
            if row < 0 or row >= height or col < 0 or col >= width:
                return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), f"Coordinates out of bounds. Valid range: (0-{height-1}, 0-{width-1}).", self._format_path_display()
            
            # Don't allow adding to start or end positions
            if (row, col) == self.start_position:
                return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), "This coordinate is already the start position.", self._format_path_display()
                
            if self.end_position and (row, col) == self.end_position:
                return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), "This coordinate is already the end position.", self._format_path_display()
            
            # For intermediate points, check if it's already marked
            if self.generation_grid[row, col] != 0:  # If not a wall
                return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), f"Position ({row},{col}) already used. Try a different coordinate.", self._format_path_display()
            
            # Mark as a path point
            self.generation_grid[row, col] = PATH  # Set to PATH (1)
            
            # If we have an end position, insert before the end
            if self.end_position and len(self.generation_coords) > 1 and self.generation_coords[-1] == self.end_position:
                # Insert before the end
                self.generation_coords.insert(-1, (row, col))
            else:
                # If no end position yet, just append
                self.generation_coords.append((row, col))
            
            return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), f"Added point at ({row},{col}). Continue adding points to create your shape.", self._format_path_display()
            
        except ValueError:
            # Handle non-integer inputs
            return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), "Invalid input. Please enter integers in 'row,col' format (e.g. '2,3').", self._format_path_display()
        except Exception as e:
            print(f"Error processing coordinate: {e}")
            return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True), f"Error: {str(e)}", self._format_path_display()
    
    def _format_path_display(self):
        """Format the current path for display in the UI.
        
        Returns:
            String with formatted path information
        """
        if not self.generation_coords:
            return "Current path: []"
            
        # Create the path display with special marking for start and end
        path_text = "Current path:\n"
        
        for i, (r, c) in enumerate(self.generation_coords):
            position_type = ""
            if (r, c) == self.start_position:
                position_type = " (START)"
            elif self.end_position and (r, c) == self.end_position:
                position_type = " (END)"
                
            path_text += f"{i+1}: ({r},{c}){position_type}\n"
        
        return path_text
        
    def reset_generation(self):
        """Reset the generation grid and all positions."""
        print("Resetting generation grid")
        
        if self.current_size is None:
            return None, "No maze loaded to reset.", "generate", self.get_progress(), "Current path: []"
            
        # Create a new blank grid
        height, width = self.current_maze.shape
        grid = np.zeros((height, width), dtype=np.int8)  # Start with all walls (0)
        
        # Reset all tracking variables
        self.generation_coords = []
        self.start_position = None
        self.end_position = None
        self.generation_grid = grid
        
        return save_numpy_array_as_image(grid, is_generation=True, show_coordinates=True), "Generation grid reset. Please set start position first.", "generate", self.get_progress(), "Current path: []"
    
    def validate_generation(self):
        """Validate the user's generated maze shape against the current shape.
        
        Returns:
            Dict with validation results
        """
        if not hasattr(self, 'generation_grid') or self.generation_grid is None:
            return {
                'valid': False,
                'error': 'No maze has been generated'
            }
        
        # Require both start and end positions
        if not self.start_position:
            return {
                'valid': False,
                'error': 'No start position has been set',
                'feedback': 'Please set a start position before submitting.'
            }
            
        if not self.end_position:
            return {
                'valid': False,
                'error': 'No end position has been set',
                'feedback': 'Please set an end position before submitting.'
            }
            
        # Extract the path from the generation coordinates
        path_cells = self.generation_coords
                    
        # Compute a simple shape score - can be expanded with more sophisticated metric
        total_cells = self.generation_grid.shape[0] * self.generation_grid.shape[1]
        path_ratio = len(path_cells) / total_cells
        
        # Validate based on current shape 
        # (simple validation - could be enhanced with shape recognition)
        valid = True
        feedback = "Shape accepted. Great work!"
        
        # Basic validation to avoid trivial mazes
        if len(path_cells) < 5:
            valid = False
            feedback = "Your drawing is too small. Please create a more complex shape with at least 5 points."
            
        return {
            'valid': valid,
            'shape': self.current_shape,
            'generated_shape': path_cells,
            'feedback': feedback,
            'path_ratio': path_ratio,
            'start_position': self.start_position,
            'end_position': self.end_position
        }
    
    def submit_generation_drawing(self):
        """Process the generated maze and move to the next."""
        result = self.validate_generation()
        
        # Save the results
        self.save_generation_results(result)
        
        # Get feedback message
        feedback = result['feedback']
        
        if not result['valid']:
            # If not valid, let the user try again
            return save_numpy_array_as_image(self.generation_grid, is_generation=True), feedback, "generate", self.get_progress(), self._format_path_display()
        
        # Force complete clearing of generation data
        self.generation_grid = None
        self.generation_coords = []
        self.start_position = None
        self.end_position = None
        
        # Mark this maze as completed with all tasks (solve, recognize, generate)
        maze_img, msg, phase, progress = self.complete_current_maze()
        return maze_img, msg, phase, progress, "Current path: []"
    
    def get_progress(self):
        """Get the current progress information for the experiment.
        
        Returns:
            A string with progress information.
        """
        if self.current_phase == "welcome":
            return "Please enter your participant ID to start"
            
        total_completed = sum(self.combination_counts.values())
        total_mazes = self.total_mazes
        
        completed_combinations = len(self.completed_combinations)
        total_combinations = len(SIZES) * len(SHAPES)
        
        current_size = "None" if self.current_size is None else f"{self.current_size[0]}x{self.current_size[1]}"
        current_shape = "None" if self.current_shape is None else self.current_shape
        
        if self.current_file_info:
            combo_key = (self.current_file_info["size"], self.current_file_info["shape"])
            completed_in_combo = self.combination_counts[combo_key]
        else:
            completed_in_combo = 0
        
        return f"Progress: {total_completed}/{total_mazes} mazes completed ({completed_combinations}/{total_combinations} combinations)"

    def select_next_combination(self):
        """Select the next size/shape combination that needs mazes."""
        # If current combination is complete, find a new one
        if self.current_size and self.current_shape:
            key = f"{self.current_size[0]}x{self.current_size[1]}_{self.current_shape}"
            if self.combination_counts.get(key, 0) >= self.mazes_per_combination:
                # Current combination is complete, find a new one
                for size in SIZES:
                    for shape in SHAPES:
                        check_key = f"{size[0]}x{size[1]}_{shape}"
                        if self.combination_counts.get(check_key, 0) < self.mazes_per_combination:
                            self.current_size = size
                            self.current_shape = shape
                            return True  # Found a new combination
                
                # All combinations are complete
                return False
        
        # Current combination still has mazes to complete
        return True

    def complete_current_maze(self):
        """Mark the current maze as completed and move to the next one."""
        if self.current_size and self.current_shape:
            key = (f"{self.current_size[0]}x{self.current_size[1]}", self.current_shape)
            self.combination_counts[key] += 1
            
            # Check if all mazes for this combination are completed
            if self.combination_counts[key] >= MAZES_PER_COMBINATION:
                self.completed_combinations.append(key)
            
            # Ensure complete reset of generation data before moving to next maze
            self.generation_grid = None
            self.generation_coords = []
            self.start_position = None
            self.end_position = None
            
            # Move to the next combination if all combinations are not complete
            if not self.is_experiment_complete():
                # Explicitly ensure we're in solve phase when loading next maze
                self.current_phase = "solve"
                return self.load_next_combination()
            else:
                self.experiment_complete = True
                return save_numpy_array_as_image(np.zeros((5, 5))), "Experiment complete! Thank you for participating.", "complete", self.get_progress()

    # Add a new method to force phase reset
    def force_phase_reset(self):
        """Force a reset of the experiment's phase state to solve phase and load a new maze.
        This is an emergency recovery function to break out of phase desync loops.
        """
        print("EMERGENCY: Forcing phase reset to solve phase")
        # Clear any existing generation data
        self.generation_grid = None
        self.generation_coords = []
        self.start_position = None
        self.end_position = None
        
        # Force phase back to solve
        self.current_phase = "solve"
        # Mark current maze as complete to move on
        if self.current_size and self.current_shape:
            key = (f"{self.current_size[0]}x{self.current_size[1]}", self.current_shape)
            self.combination_counts[key] += 1
        
        # Load the next maze combination
        return self.load_next_combination()

    def initial_load(self, size, shape):
        """Initial load of a maze with the specified size and shape.
        
        Args:
            size: Tuple of (height, width)
            shape: String representing the maze shape
            
        Returns:
            Tuple of (maze_image, message, phase, progress)
        """
        # Reset experiment state for a new start
        self.maze_complete = False
        self.moves = []
        
        # Load a random maze for the specified size and shape
        return self.load_random_maze(size, shape)

    def render_maze(self):
        """Render the current maze as an image.
        
        Returns:
            PIL Image of the current maze
        """
        if self.current_maze is None:
            # Create a blank maze image
            blank_maze = np.ones((5, 5), dtype=np.int8)
            return save_numpy_array_as_image(blank_maze)
            
        return save_numpy_array_as_image(self.current_maze)

    def save_generation_results(self, result):
        """Save the generation results to a file.
        
        Args:
            result: Dict with generation validation results
        """
        if not self.current_file_info:
            return
            
        # Find the most recent result file for this maze
        result_files = []
        for file_path in self.results_dir.rglob("*.json"):
            if file_path.name.startswith(self.participant_id):
                result_files.append(file_path)
        
        # Sort by creation time and get the most recent
        if result_files:
            file_path = sorted(result_files, key=lambda p: p.stat().st_mtime)[-1]
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Add the generation results
                data["generation_result"] = result
                # Add the coordinate entry method info
                data["generation_method"] = "coordinate_entry"
                data["generation_coordinates"] = self.generation_coords
                
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                print(f"Generation results saved to {file_path}")
                
            except Exception as e:
                print(f"Error saving generation results: {e}")
        else:
            timestamp = int(time.time())
            filename = self.results_dir / f"{self.participant_id}_{self.current_file_info['size']}_{self.current_shape}_generation_{timestamp}.json"
            
            # Prepare data to save
            data = {
                "participant_id": self.participant_id,
                "maze_file": self.current_file_info['file'],
                "maze_type": {
                    "size": self.current_file_info['size'],
                    "shape": self.current_file_info['shape']
                },
                "generation_result": result,
                "generation_method": "coordinate_entry",
                "generation_coordinates": self.generation_coords,
                "timestamp": timestamp
            }
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"Generation results saved to {filename}")

    def get_generation_grid_image(self):
        """Get an image of the current generation grid.
        
        Returns:
            Image: PIL Image of the generation grid
        """
        if not hasattr(self, 'generation_grid') or self.generation_grid is None:
            # Create the generation grid if it doesn't exist
            return self.create_generation_grid()
        
        return save_numpy_array_as_image(self.generation_grid, is_generation=True, show_coordinates=True)

    def clear_generation_grid(self):
        """Clear any existing generation grid to avoid persistence between mazes."""
        print(f"Clearing generation grid in phase: {self.current_phase}")
        # Ensure the generation grid is completely cleared
        if hasattr(self, 'generation_grid'):
            self.generation_grid = None
        if hasattr(self, 'generation_coords'):
            self.generation_coords = []
        if hasattr(self, 'start_position'):
            self.start_position = None
        if hasattr(self, 'end_position'):
            self.end_position = None

def create_interface(experiment):
    """
    Create the Gradio interface for the maze experiment.
    
    Args:
        experiment: MazeExperiment instance.
        
    Returns:
        gr.Interface: The Gradio interface for the experiment.
    """
    # Create the interface components
    with gr.Blocks() as interface:
        # Welcome screen components
        with gr.Group(visible=True) as welcome_screen:
            gr.Markdown("# Welcome to the Maze Solving Experiment")
            gr.Markdown("""
            In this experiment, you will:
            1. Solve a maze puzzle by navigating through it.
            2. Identify the shape pattern of the maze.
            3. Create a maze that is the same shape as the previous given maze. \n

            There will be a total of 12 mazes, with 1 maze for each combination of two sizes and six shapes. \n

            You will see a progress box that shows you how many mazes you have completed. \n

            Please enter your participant ID below to begin \n 
            (If Prolific did not assign you an ID, please enter your birthday in the format DDMMYYYY).
            You must enter an ID before clicking the start button.
            """)
            participant_id_input = gr.Textbox(label="Participant ID", placeholder="Enter your ID here")
            start_button = gr.Button("Start Experiment")
        
        # Main experiment components (initially hidden)
        with gr.Group(visible=False) as experiment_screen:
            with gr.Row():
                maze_display = gr.Image(label="Maze")
                
            with gr.Row():
                message = gr.Markdown("Loading maze...")
                
            with gr.Row():
                phase_info = gr.Textbox(label="Current Phase", value="welcome", visible=False)
                progress_info = gr.Textbox(label="Progress", value="Progress: 0/12 mazes completed (0/12 combinations)")
             
            # Movement buttons - cardinal directions   
            with gr.Row(visible=True) as movement_controls:
                up_button = gr.Button("Up")
                down_button = gr.Button("Down")
                left_button = gr.Button("Left")
                right_button = gr.Button("Right")
            
            # Movement buttons - diagonal directions
            with gr.Row(visible=True) as diagonal_controls:
                up_left_button = gr.Button("↖ Up-Left")
                up_right_button = gr.Button("↗ Up-Right")
                down_left_button = gr.Button("↙ Down-Left")
                down_right_button = gr.Button("↘ Down-Right")
                
            # Shape recognition controls
            with gr.Row(visible=False) as recognition_controls:
                shape_input = gr.Dropdown(
                    label="Recognize Shape",
                    choices=SHAPES,
                    info="What shape is formed by the non-black squares of the maze? Select the shape you see from the dropdown, then click on Submit Recognition button."
                )
                submit_recognition_button = gr.Button("Submit Recognition")
                
            # Generation controls - replacing buttons with coordinate input
            with gr.Group(visible=False) as generation_controls:
                gr.Markdown("### Draw your shape using coordinates")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # Start position input (blue)
                        gr.Markdown("📌 **Start Position** (Blue)")
                        start_pos_input = gr.Textbox(
                            label="Start Position (row,col)", 
                            placeholder="e.g. 2,3",
                            info="Enter the start position as row,col"
                        )
                        set_start_button = gr.Button("Set Start Position", variant="primary")
                    
                    with gr.Column(scale=3):
                        # End position input (red)
                        gr.Markdown("🏁 **End Position** (Red)")
                        end_pos_input = gr.Textbox(
                            label="End Position (row,col)", 
                            placeholder="e.g. 5,5",
                            info="Enter the end position as row,col"
                        )
                        set_end_button = gr.Button("Set End Position", variant="stop")
                
                gr.Markdown("---")
                gr.Markdown("📍 **Intermediate Points** (White)")
                
                with gr.Row():
                    coord_input = gr.Textbox(
                        label="Add Point (row,col)", 
                        placeholder="e.g. 3,4",
                        info="Enter a coordinate as row,col to add an intermediate point to your shape"
                    )
                    add_coord_button = gr.Button("Add Point")
                
                with gr.Row():
                    reset_gen_button = gr.Button("Reset Grid", variant="secondary")
                    submit_gen_button = gr.Button("Submit Generated Shape", variant="primary")
                    
                with gr.Row():
                    current_path_display = gr.Markdown("Current path: []")
                
            # Emergency controls for recovery from bugs
            with gr.Row(visible=True) as emergency_controls:
                gr.Markdown("*Use the button below ONLY if you encounter a bug and need to move to the next maze*")
            
            # Add some space to separate the emergency button from regular controls
            gr.Markdown("---")
            
            # Put emergency button in its own row at the bottom with less prominence
            with gr.Row(visible=True) as emergency_button_row:
                # Add empty column to push button to the right side
                with gr.Column(scale=2):
                    gr.Markdown("")
                with gr.Column(scale=1):
                    reset_state_button = gr.Button("⚠️ Reset/Next Maze", size="sm")
                # Add empty column for balance
                with gr.Column(scale=2):
                    gr.Markdown("")
        
        # Start button event
        def handle_start(participant_id):
            if not participant_id or participant_id.strip() == "":
                return {
                    welcome_screen: gr.update(visible=True),
                    experiment_screen: gr.update(visible=False),
                    participant_id_input: gr.update(value="", placeholder="Please enter your ID to continue"),
                    maze_display: None,  # Add defaults for the other outputs
                    message: "",
                    phase_info: "welcome",
                    progress_info: "Not started yet",
                    current_path_display: ""
                }
            
            # Force reset of any remaining state
            experiment.generation_grid = None
            experiment.generation_coords = []
            experiment.start_position = None
            experiment.end_position = None
            
            # Set participant ID and get first maze
            maze_img, msg, phase, progress = experiment.set_participant_id(participant_id)
            
            return {
                welcome_screen: gr.update(visible=False),
                experiment_screen: gr.update(visible=True),
                maze_display: maze_img,
                message: msg,
                phase_info: phase,
                progress_info: progress,
                current_path_display: "Current path: []"  # Reset path display
            }
            
        start_button.click(
            fn=handle_start,
            inputs=participant_id_input,
            outputs=[welcome_screen, experiment_screen, maze_display, message, phase_info, progress_info, current_path_display]
        )
            
        # Define helper functions to route actions based on current phase
        def handle_move(direction, phase):
            # Verify phase is valid - fallback to solving if corruption occurs
            if phase not in ["solve", "generate", "recognize", "complete"]:
                print(f"Warning: Invalid phase detected: {phase}. Falling back to solve phase.")
                phase = "solve"
            
            # CRITICAL LOG: Output the current phase and direction
            print(f"MOVE REQUEST: Direction={direction}, UI Phase={phase}, Experiment Phase={experiment.current_phase}")
            
            # Double-check the phase matches the experiment's current phase
            if phase != experiment.current_phase:
                print(f"WARNING: Phase mismatch! UI phase: {phase}, Experiment phase: {experiment.current_phase}")
                
                # Use experiment phase as source of truth
                phase = experiment.current_phase
                
                # Special error handling for specific phase mismatches:
                if phase == "recognize":
                    print("Redirecting to recognition screen")
                    return maze_display.value, "Please select a shape to continue.", "recognize", progress_info.value, "Current path: []"
                    
                if phase == "generate" and not hasattr(experiment, 'generation_grid'):
                    print("Phase is generate but no grid exists - forcing reset")
                    # This is a corrupted state - force a reset
                    img, msg, new_phase, progress = experiment.force_phase_reset()
                    return img, msg, new_phase, progress, "Current path: []"
            
            if phase == "solve":
                # In solve phase, always use process_move
                img, msg, new_phase, progress = experiment.process_move(direction)
                # If the phase changed (e.g., to recognize after solving), log it
                if new_phase != phase:
                    print(f"Phase changed during move from {phase} to {new_phase}")
                return img, msg, new_phase, progress, "Current path: []"
            else:
                # For other phases, do nothing and return current state
                path_text = experiment._format_path_display() if hasattr(experiment, 'generation_coords') and experiment.current_phase == "generate" else "Current path: []"
                return maze_display.value, message.value, phase, progress_info.value, path_text
            
        # Button click events - cardinal directions
        up_button.click(
            fn=handle_move,
            inputs=[gr.Textbox(value="up", visible=False), phase_info],
            outputs=[maze_display, message, phase_info, progress_info, current_path_display]
        )
        down_button.click(
            fn=handle_move,
            inputs=[gr.Textbox(value="down", visible=False), phase_info],
            outputs=[maze_display, message, phase_info, progress_info, current_path_display]
        )
        left_button.click(
            fn=handle_move,
            inputs=[gr.Textbox(value="left", visible=False), phase_info],
            outputs=[maze_display, message, phase_info, progress_info, current_path_display]
        )
        right_button.click(
            fn=handle_move,
            inputs=[gr.Textbox(value="right", visible=False), phase_info],
            outputs=[maze_display, message, phase_info, progress_info, current_path_display]
        )
        
        # Button click events - diagonal directions
        up_left_button.click(
            fn=handle_move,
            inputs=[gr.Textbox(value="up-left", visible=False), phase_info],
            outputs=[maze_display, message, phase_info, progress_info, current_path_display]
        )
        up_right_button.click(
            fn=handle_move,
            inputs=[gr.Textbox(value="up-right", visible=False), phase_info],
            outputs=[maze_display, message, phase_info, progress_info, current_path_display]
        )
        down_left_button.click(
            fn=handle_move,
            inputs=[gr.Textbox(value="down-left", visible=False), phase_info],
            outputs=[maze_display, message, phase_info, progress_info, current_path_display]
        )
        down_right_button.click(
            fn=handle_move,
            inputs=[gr.Textbox(value="down-right", visible=False), phase_info],
            outputs=[maze_display, message, phase_info, progress_info, current_path_display]
        )
        
        # Shape recognition event
        submit_recognition_button.click(
            fn=experiment.submit_shape_recognition,
            inputs=shape_input,
            outputs=[maze_display, message, phase_info, progress_info, current_path_display]
        )
        
        # Generation events with coordinate inputs for different position types
        def handle_set_start(coord, phase):
            if phase != "generate":
                return maze_display.value, "Not in generation phase", phase, progress_info.value, current_path_display.value
            
            # Process the start position
            img, msg, path_text = experiment.set_start_position(coord)
            return img, msg, phase, progress_info.value, path_text
            
        def handle_set_end(coord, phase):
            if phase != "generate":
                return maze_display.value, "Not in generation phase", phase, progress_info.value, current_path_display.value
            
            # Process the end position
            img, msg, path_text = experiment.set_end_position(coord)
            return img, msg, phase, progress_info.value, path_text
            
        def handle_add_coordinate(coord, phase):
            if phase != "generate":
                return maze_display.value, "Not in generation phase", phase, progress_info.value, current_path_display.value
            
            # Process the coordinate input for intermediate points
            img, msg, path_text = experiment.add_coordinate(coord)
            return img, msg, phase, progress_info.value, path_text
        
        def handle_reset_generation(phase):
            if phase != "generate":
                return maze_display.value, "Not in generation phase", phase, progress_info.value, current_path_display.value
                
            # Reset the generation grid
            img, msg, phase, progress, path_text = experiment.reset_generation()
            # Return all values
            return img, msg, phase, progress, path_text
        
        # Set handlers for the generation controls
        set_start_button.click(
            fn=handle_set_start,
            inputs=[start_pos_input, phase_info],
            outputs=[maze_display, message, phase_info, progress_info, current_path_display]
        )
        
        set_end_button.click(
            fn=handle_set_end,
            inputs=[end_pos_input, phase_info],
            outputs=[maze_display, message, phase_info, progress_info, current_path_display]
        )
            
        add_coord_button.click(
            fn=handle_add_coordinate,
            inputs=[coord_input, phase_info],
            outputs=[maze_display, message, phase_info, progress_info, current_path_display]
        )
            
        reset_gen_button.click(
            fn=handle_reset_generation,
            inputs=[phase_info],
            outputs=[maze_display, message, phase_info, progress_info, current_path_display]
        )
            
        submit_gen_button.click(
            fn=experiment.submit_generation_drawing,
            inputs=[],
            outputs=[maze_display, message, phase_info, progress_info, current_path_display]
        )
        
        # Add a handler for the emergency reset button
        def handle_emergency_reset():
            print("Emergency reset triggered - moving to next maze")
            # Reset phase mismatch counter
            global phase_mismatch_count
            if 'phase_mismatch_count' in globals():
                phase_mismatch_count = 0
                
            # Force the experiment to reset its phase completely
            img, msg, phase, progress = experiment.force_phase_reset()
            
            # Also reset the current path display
            path_text = "Current path: []"
            
            return img, msg, phase, progress, path_text
            
        reset_state_button.click(
            fn=handle_emergency_reset,
            inputs=[],
            outputs=[maze_display, message, phase_info, progress_info, current_path_display]
        )
        
        # Phase change event handler
        def handle_phase_change(phase):
            # Log phase change for debugging
            print(f"Phase changing to: {phase}")
            
            # Validate phase is a known value
            if phase not in ["solve", "generate", "recognize", "complete", "welcome", "error"]:
                print(f"WARNING: Unknown phase detected: {phase}. Defaulting to solve phase.")
                phase = "solve"
            
            # Force complete reset of generation data during phase transitions
            if phase == "solve":
                print("Performing complete reset of generation data for solve phase")
                experiment.generation_grid = None
                experiment.generation_coords = []
                experiment.start_position = None
                experiment.end_position = None
                
            # STRICT PHASE TRANSITION VALIDATION:
            # Prevent skipping from solve to generate without going through recognize
            current_experiment_phase = experiment.current_phase
            
            # DESYNC DETECTION: If we see the same mismatch repeatedly, trigger emergency reset
            global phase_mismatch_count
            if not 'phase_mismatch_count' in globals():
                phase_mismatch_count = 0
                
            if current_experiment_phase != phase:
                phase_mismatch_count += 1
                print(f"Phase mismatch detected - count: {phase_mismatch_count}")
                
                if phase_mismatch_count > 3:  # If we've detected more than 3 mismatches in a row
                    print("CRITICAL ERROR: Too many phase mismatches detected.")
                    print("Forcing emergency phase reset to break loop")
                    # Reset the counter
                    phase_mismatch_count = 0
                    # Force experiment back to a known good state
                    img, msg, reset_phase, progress = experiment.force_phase_reset()
                    # Update our phase to match
                    phase = reset_phase
            else:
                # Reset counter when phases match
                phase_mismatch_count = 0
            
            if phase == "generate" and current_experiment_phase == "solve":
                print(f"ERROR: Attempted to transition directly from solve to generate phase!")
                print(f"Forcing transition to recognize phase instead.")
                phase = "recognize"
                experiment.current_phase = "recognize"  # Force experiment state to match
            
            # After all validations, ensure the experiment phase matches the UI phase
            # This is crucial to prevent desync
            experiment.current_phase = phase
            
            if phase == "solve":
                # Clear any generation grid data when entering solve phase
                experiment.clear_generation_grid()
                
                # In solve phase, show movement controls but hide recognition and generation controls
                print("UI updated for solve phase: showing movement controls")
                return {
                    movement_controls: gr.update(visible=True),
                    diagonal_controls: gr.update(visible=True),
                    recognition_controls: gr.update(visible=False),
                    generation_controls: gr.update(visible=False),
                    emergency_controls: gr.update(visible=True),
                    emergency_button_row: gr.update(visible=True)
                }
            elif phase == "recognize":
                # In recognition phase, hide movement and generation controls, show recognition controls
                print("UI updated for recognize phase: showing recognition controls")
                
                # Ensure any old generation grid is cleared when entering recognition phase
                if hasattr(experiment, 'generation_grid') and experiment.generation_grid is not None:
                    print("Clearing generation grid when entering recognition phase")
                    experiment.clear_generation_grid()
                
                return {
                    movement_controls: gr.update(visible=False),
                    diagonal_controls: gr.update(visible=False),
                    recognition_controls: gr.update(visible=True),
                    generation_controls: gr.update(visible=False),
                    emergency_controls: gr.update(visible=True),
                    emergency_button_row: gr.update(visible=True)
                }
            elif phase == "generate":
                # In generation phase, verify we have a generation grid
                if not hasattr(experiment, 'generation_grid') or experiment.generation_grid is None:
                    print("WARNING: In generation phase but no generation grid found! Creating one now.")
                    # Create a new generation grid, but only if we're actually in the generate phase
                    if experiment.current_phase == "generate":
                        experiment.create_generation_grid()
                    else:
                        print(f"ERROR: Current phase ({experiment.current_phase}) doesn't match UI phase (generate)!")
                        # This is an error state - the UI thinks we're in generate but the experiment thinks otherwise
                        # Set the phase back to what the experiment thinks it is
                        phase = experiment.current_phase
                        # If we need to return to recognize phase
                        if phase == "recognize":
                            return {
                                movement_controls: gr.update(visible=False),
                                diagonal_controls: gr.update(visible=False),
                                recognition_controls: gr.update(visible=True),
                                generation_controls: gr.update(visible=False),
                                emergency_controls: gr.update(visible=True),
                                emergency_button_row: gr.update(visible=True)
                            }
                
                print("UI updated for generate phase: showing generation controls, hiding movement controls")
                # Show generation controls, hide movement controls
                return {
                    movement_controls: gr.update(visible=False),
                    diagonal_controls: gr.update(visible=False),
                    recognition_controls: gr.update(visible=False),
                    generation_controls: gr.update(visible=True),
                    emergency_controls: gr.update(visible=False),
                    emergency_button_row: gr.update(visible=False)
                }
            else:
                # For other phases (welcome, complete, error), hide all controls
                print(f"UI updated for {phase} phase: hiding all controls")
                return {
                    movement_controls: gr.update(visible=False),
                    diagonal_controls: gr.update(visible=False),
                    recognition_controls: gr.update(visible=False),
                    generation_controls: gr.update(visible=False),
                    emergency_controls: gr.update(visible=True),
                    emergency_button_row: gr.update(visible=True)
                }
                
        phase_info.change(
            fn=handle_phase_change,
            inputs=phase_info,
            outputs=[movement_controls, diagonal_controls, recognition_controls, generation_controls, emergency_controls, emergency_button_row]
        )
        
    return interface

if __name__ == "__main__":
    experiment = MazeExperiment()
    interface = create_interface(experiment)
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Typically False when you want to host on your own
    )
