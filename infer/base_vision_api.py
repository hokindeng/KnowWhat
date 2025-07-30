"""
Base Vision API class for maze solving experiments with image inputs.
Extends the base API with vision-specific functionality.
"""

import os
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple
from abc import abstractmethod

from base_api import BaseAPISolver
from core.prompts import *


class BaseVisionAPISolver(BaseAPISolver):
    """Base class for vision-based API maze solvers."""
    
    def save_numpy_array_as_image(self, array: np.ndarray, save_path: str, 
                                  cell_size: int = 0, target_size: int = 2000,
                                  font_scale: float = 0.4) -> str:
        """
        Saves a 2D NumPy array as an image with:
          - Distinct colors for each cell value with optimal contrast.
          - Thick borders around each cell.
          - Text showing each cell's coordinates in a dynamically sized font.
          - Final image sized to approximately target_size.

        Args:
            array: 2D NumPy array representing the maze
            save_path: Path to save the image
            cell_size: Size of each cell in pixels (0 = auto-calculate)
            target_size: Target size for the full image
            font_scale: Scale factor for font size as proportion of cell size

        Returns:
            Path to the saved image
        """
        # Color map for maze elements
        color_map = {
            0: (255, 255, 255),  # PATH - white
            1: (80, 80, 80),     # WALL - dark gray
            2: (0, 102, 204),    # POS - blue
            3: (204, 0, 0),      # END - red
        }
        border_color = (0, 0, 0)  # black borders
        border_width = 3

        height, width = array.shape

        # Calculate cell size if not provided
        if cell_size <= 0:
            cell_size = min(target_size // width, target_size // height)

        # Create image
        img_width = width * cell_size
        img_height = height * cell_size
        image = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        # Calculate font size
        font_size = max(10, int(cell_size * font_scale))
        font = self._get_font(font_size)

        # Draw cells
        for row in range(height):
            for col in range(width):
                x1 = col * cell_size
                y1 = row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                cell_value = array[row, col]
                fill_color = color_map.get(cell_value, (255, 255, 255))

                # Draw cell with border
                draw.rectangle(
                    [x1, y1, x2, y2],
                    fill=fill_color,
                    outline=border_color,
                    width=border_width
                )

                # Add coordinate text
                text = f"({row},{col})"
                text_width, text_height = self._get_text_dimensions(text, font)
                text_x = x1 + (cell_size - text_width) // 2
                text_y = y1 + (cell_size - text_height) // 2

                # Choose text color for contrast
                luminance = 0.299 * fill_color[0] + 0.587 * fill_color[1] + 0.114 * fill_color[2]
                text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

                draw.text((text_x, text_y), text, fill=text_color, font=font)

        # Save image
        if os.path.isdir(save_path):
            directory = save_path
        else:
            directory = os.path.dirname(save_path)
        output_path = os.path.join(directory, "maze.png")
        image.save(output_path)

        return output_path

    def _get_font(self, font_size: int):
        """Get a font with the specified size, falling back to default if needed."""
        # Try common font paths
        font_paths = [
            "arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf",
            "LiberationSans-Regular.ttf", "Verdana.ttf"
        ]
        
        for font_name in font_paths:
            try:
                return ImageFont.truetype(font_name, font_size)
            except IOError:
                continue

        # Try system font directories
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
                        return ImageFont.truetype(os.path.join(font_dir, font_files[0]), font_size)
                    except:
                        pass

        # Fallback to default
        return ImageFont.load_default()

    def _get_text_dimensions(self, text: str, font):
        """Get text dimensions, handling different PIL versions."""
        try:
            # For newer PIL versions
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            try:
                # For PIL 8.0.0+
                return font.getsize(text)
            except AttributeError:
                # Fallback for older PIL versions
                return font.getmask(text).size

    def encode_image_base64(self, image_path: str) -> str:
        """Encode an image file to base64 string."""
        with open(image_path, "rb") as image_file:
            binary_data = image_file.read()
            base64_encoded = base64.b64encode(binary_data)
            return base64_encoded.decode('utf-8')

    def get_initial_prompt(self, maze_string: str, encoding_type: str, use_diag: bool) -> str:
        """Get the appropriate initial prompt for vision models."""
        if encoding_type == "vision":
            if use_diag:
                return INITIAL_SOLVE_PROMPT_VISION_DIAG
            else:
                return INITIAL_SOLVE_PROMPT_VISION
        else:
            # Fall back to parent implementation for non-vision encoding
            return super().get_initial_prompt(maze_string, encoding_type, use_diag)

    @abstractmethod
    def _format_vision_message(self, text: str, image_base64: str) -> any:
        """Format a message with text and image for the specific API."""
        pass

    def solve_maze_with_api(self, maze: np.ndarray, encoding_type: str, use_diag: bool,
                           size: Tuple[int, int], shape: str, result_path: str):
        """Override to handle vision-specific logic."""
        if encoding_type == "vision":
            self._solve_maze_with_vision(maze, use_diag, size, shape, result_path)
        else:
            super().solve_maze_with_api(maze, encoding_type, use_diag, size, shape, result_path)

    def _solve_maze_with_vision(self, maze: np.ndarray, use_diag: bool,
                               size: Tuple[int, int], shape: str, result_path: str):
        """Solve maze using vision API."""
        try:
            messages = []
            status = self.Status.CONTINUE
            
            with open(result_path, "w") as file:
                i = 0
                while status == self.Status.CONTINUE:
                    # Save maze as image
                    image_path = self.save_numpy_array_as_image(maze, result_path)
                    image_base64 = self.encode_image_base64(image_path)
                    
                    # Get appropriate prompt
                    if i == 0:
                        text_prompt = self.get_initial_prompt("", "vision", use_diag)
                        text_prompt += "\n" + INITIAL_SOLVE_PROMPT_2
                    else:
                        text_prompt = UPDATE_SOLVE_PROMPT_1 + "\n" + UPDATE_SOLVE_PROMPT_2
                    
                    # Format message for specific API
                    vision_message = self._format_vision_message(text_prompt, image_base64)
                    messages.append({"role": "user", "content": vision_message})
                    
                    # Make API call
                    assistant_reply = self._make_api_call(messages)
                    messages.append({"role": "assistant", "content": assistant_reply})
                    
                    # Log interaction
                    file.write(text_prompt)
                    file.write("\n[Image provided]\n")
                    file.write(assistant_reply)
                    file.write("\n")
                    
                    # Parse and update
                    status, updated_maze = self.parse_and_update(assistant_reply, maze)
                    
                    if status == self.Status.SUCCESS:
                        file.write("\n\nSOLVE: SUCCESS")
                        break
                    elif status == self.Status.FAIL:
                        file.write("\n\nSOLVE: FAIL")
                        break
                    elif status == self.Status.CONTINUE:
                        maze = updated_maze
                        if i >= 15:  # Maximum iterations
                            file.write("\n\nSOLVE: FAIL")
                            break
                    
                    i += 1
                
                # Recognition and generation phase
                messages.append({"role": "user", "content": RECOGNIZE_AND_GENERATE_PROMPT_VISION})
                assistant_reply = self._make_api_call(messages)
                
                file.write(RECOGNIZE_AND_GENERATE_PROMPT_VISION)
                file.write("\n")
                file.write(assistant_reply)
                file.write("\n")
                
                # Check correctness
                generate_correct = is_correct_generate(assistant_reply, "vision", size, shape, maze)
                recognize_correct = is_correct_recognize(assistant_reply, shape)
                
                file.write("\n\nRECOGNIZE: " + ("SUCCESS" if recognize_correct else "FAIL"))
                file.write("\n\nGENERATE: " + ("SUCCESS" if generate_correct else "FAIL"))
                
        except Exception as e:
            with open(result_path, "w") as file:
                print(f"\n\nAlert, query did not run! Path: {result_path}")
                print(f'Error: {str(e)}')
                self.alert_count += 1
                file.write(f"\n\nQuery did not run: {result_path}")
                file.write(f"\n\nError: {str(e)}")
                file.write("\n\nSOLVE: FAIL")
                file.write("\n\nRECOGNIZE: FAIL")
                file.write("\n\nGENERATE: FAIL") 