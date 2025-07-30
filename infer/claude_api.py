"""
Claude (Anthropic) API implementation for maze solving experiments.
Supports both text and vision-based approaches.
"""

import os
from dotenv import load_dotenv
import anthropic
from base_api import BaseAPISolver
from base_vision_api import BaseVisionAPISolver

# Load environment variables
load_dotenv()


class ClaudeAPISolver(BaseVisionAPISolver):
    """Claude API implementation for maze solving with text and vision support."""
    
    def __init__(self, api_key: str = None, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize Claude solver with optional model selection."""
        self.model = model
        super().__init__(api_key)
    
    def _get_api_key(self) -> str:
        """Get API key from environment variables."""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        return api_key
    
    def _initialize_client(self):
        """Initialize the Anthropic client."""
        return anthropic.Anthropic(api_key=self.api_key)
    
    def _make_api_call(self, messages: list, **kwargs) -> str:
        """Make an API call to Claude and return the response text."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get('max_tokens', 1000),
            temperature=kwargs.get('temperature', 0),
            system=kwargs.get('system', "You are a helpful AI assistant."),
            messages=messages,
        )
        return response.content[0].text
    
    def _format_vision_message(self, text: str, image_base64: str) -> list:
        """Format a message with text and image for Claude's API."""
        return [
            {"type": "text", "text": text},
            {
                "type": "image", 
                "source": {
                    "type": "base64", 
                    "media_type": "image/png",
                    "data": image_base64
                }
            }
        ]
    
    def _get_output_directory(self, encoding_type: str) -> str:
        """Get the output directory name for results."""
        if encoding_type == "vision":
            return "vision_claude_results"
        else:
            return f"{encoding_type}_claude_results"


def main():
    """Run experiments with Claude API."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run maze solving experiments with Claude')
    parser.add_argument('--model', type=str, default='claude-3-5-sonnet-20241022',
                       help='Claude model to use')
    parser.add_argument('--encoding', type=str, nargs='+', 
                       default=['matrix', 'coord_list', 'vision'],
                       choices=['matrix', 'coord_list', 'vision'],
                       help='Encoding types to use')
    parser.add_argument('--sizes', type=str, nargs='+', default=['5x5', '7x7'],
                       help='Maze sizes to test')
    
    args = parser.parse_args()
    
    # Parse sizes
    sizes = []
    for size_str in args.sizes:
        parts = size_str.split('x')
        sizes.append((int(parts[0]), int(parts[1])))
    
    # Create solver
    solver = ClaudeAPISolver(model=args.model)
    
    # Run experiments
    solver.run_all_experiments(
        encoding_types=args.encoding,
        sizes=sizes,
        shapes=None  # Use all shapes
    )


if __name__ == "__main__":
    main()