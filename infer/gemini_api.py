"""
Google Gemini API implementation for maze solving experiments.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
from base_api import BaseAPISolver

# Load environment variables
load_dotenv()


class GeminiAPISolver(BaseAPISolver):
    """Gemini API implementation for maze solving."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-pro"):
        """Initialize Gemini solver with optional model selection."""
        self.model_name = model
        super().__init__(api_key)
    
    def _get_api_key(self) -> str:
        """Get API key from environment variables."""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        return api_key
    
    def _initialize_client(self):
        """Initialize the Gemini client."""
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(self.model_name)
    
    def _make_api_call(self, messages: list, **kwargs) -> str:
        """Make an API call to Gemini and return the response text."""
        # Convert messages to Gemini format
        # Gemini expects a single prompt string or a list of content
        prompt = self._convert_messages_to_prompt(messages)
        
        response = self.client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=kwargs.get('temperature', 0),
                max_output_tokens=kwargs.get('max_tokens', 1000),
            )
        )
        return response.text
    
    def _convert_messages_to_prompt(self, messages: list) -> str:
        """Convert OpenAI-style messages to Gemini prompt format."""
        # For now, concatenate all messages into a single prompt
        # Skip system messages as Gemini doesn't have a separate system role
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
            # Skip system messages or add them as context
            elif role == 'system':
                prompt_parts.insert(0, f"Context: {content}")
        
        return "\n\n".join(prompt_parts)
    
    def _get_output_directory(self, encoding_type: str) -> str:
        """Get the output directory name for results."""
        return f"{encoding_type}_gemini_results"


def main():
    """Run experiments with Gemini API."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run maze solving experiments with Gemini models')
    parser.add_argument('--model', type=str, default='gemini-1.5-pro',
                       help='Gemini model to use')
    parser.add_argument('--encoding', type=str, nargs='+', default=['matrix', 'coord_list'],
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
    solver = GeminiAPISolver(model=args.model)
    
    # Run experiments
    solver.run_all_experiments(
        encoding_types=args.encoding,
        sizes=sizes,
        shapes=None  # Use all shapes
    )


if __name__ == "__main__":
    main()
