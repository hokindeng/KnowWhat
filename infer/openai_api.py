"""
OpenAI API implementation for maze solving experiments.
Supports both regular OpenAI and Azure OpenAI endpoints, with text and vision capabilities.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
from base_api import BaseAPISolver
from base_vision_api import BaseVisionAPISolver

# Load environment variables
load_dotenv()


class OpenAIAPISolver(BaseVisionAPISolver):
    """OpenAI API implementation for maze solving with text and vision support."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o", use_azure: bool = False):
        """Initialize OpenAI solver with optional Azure support."""
        self.model = model
        self.use_azure = use_azure
        super().__init__(api_key)
    
    def _get_api_key(self) -> str:
        """Get API key from environment variables."""
        if self.use_azure:
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
        else:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
        return api_key
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        if self.use_azure:
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            if not endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT not found in environment variables")
            
            return AzureOpenAI(
                api_key=self.api_key,
                api_version='2023-12-01-preview',
                azure_endpoint=endpoint
            )
        else:
            return OpenAI(api_key=self.api_key)
    
    def _make_api_call(self, messages: list, **kwargs) -> str:
        """Make an API call to OpenAI and return the response text."""
        # Add system message if not present
        if not any(msg.get('role') == 'system' for msg in messages):
            messages = [{"role": "system", "content": "You are a helpful AI assistant."}] + messages
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get('max_tokens', 1000),
            temperature=kwargs.get('temperature', 0),
        )
        return response.choices[0].message.content
    
    def _format_vision_message(self, text: str, image_base64: str) -> list:
        """Format a message with text and image for OpenAI's API."""
        return [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            }
        ]
    
    def _get_output_directory(self, encoding_type: str) -> str:
        """Get the output directory name for results."""
        suffix = "_azure" if self.use_azure else ""
        if encoding_type == "vision":
            return f"vision_openai{suffix}_results"
        else:
            return f"{encoding_type}_openai{suffix}_results"


def main():
    """Run experiments with OpenAI API."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run maze solving experiments with OpenAI models')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       help='OpenAI model to use')
    parser.add_argument('--azure', action='store_true',
                       help='Use Azure OpenAI instead of regular OpenAI')
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
    solver = OpenAIAPISolver(model=args.model, use_azure=args.azure)
    
    # Run experiments
    solver.run_all_experiments(
        encoding_types=args.encoding,
        sizes=sizes,
        shapes=None  # Use all shapes
    )


if __name__ == "__main__":
    main()