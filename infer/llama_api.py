"""
Llama API implementation for maze solving experiments.
Supports multiple Llama providers: Nebius, NVIDIA NIM, and Novita.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from base_api import BaseAPISolver

# Load environment variables
load_dotenv()


class LlamaProvider:
    """Enum-like class for Llama providers."""
    NEBIUS = "nebius"
    NVIDIA_NIM = "nvidia_nim"
    NOVITA = "novita"


class LlamaAPISolver(BaseAPISolver):
    """Llama API implementation for maze solving."""
    
    # Provider configurations
    PROVIDER_CONFIGS = {
        LlamaProvider.NEBIUS: {
            "base_url": "https://api.studio.nebius.ai/v1/",
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "env_key": "NEBIUS_API_KEY"
        },
        LlamaProvider.NVIDIA_NIM: {
            "base_url": "https://integrate.api.nvidia.com/v1",
            "model": "meta/llama-3.1-405b-instruct",
            "env_key": "NVIDIA_API_KEY"
        },
        LlamaProvider.NOVITA: {
            "base_url": "https://api.novita.ai/v3/openai",
            "model": "meta-llama/llama-3.1-405b-instruct",
            "env_key": "NOVITA_API_KEY"
        }
    }
    
    def __init__(self, provider: str = LlamaProvider.NEBIUS, api_key: str = None):
        """Initialize Llama solver with specified provider."""
        if provider not in self.PROVIDER_CONFIGS:
            raise ValueError(f"Unknown provider: {provider}. Choose from: {list(self.PROVIDER_CONFIGS.keys())}")
        
        self.provider = provider
        self.config = self.PROVIDER_CONFIGS[provider]
        super().__init__(api_key)
    
    def _get_api_key(self) -> str:
        """Get API key from environment variables."""
        env_key = self.config["env_key"]
        api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError(f"{env_key} not found in environment variables")
        return api_key
    
    def _initialize_client(self):
        """Initialize the OpenAI-compatible client for Llama."""
        return OpenAI(
            base_url=self.config["base_url"],
            api_key=self.api_key
        )
    
    def _make_api_call(self, messages: list, **kwargs) -> str:
        """Make an API call to Llama and return the response text."""
        response = self.client.chat.completions.create(
            model=self.config["model"],
            messages=messages,
            temperature=kwargs.get('temperature', 0.2),
            top_p=kwargs.get('top_p', 0.7),
            max_tokens=kwargs.get('max_tokens', 1024),
            stream=False
        )
        return response.choices[0].message.content
    
    def _get_output_directory(self, encoding_type: str) -> str:
        """Get the output directory name for results."""
        return f"{encoding_type}_llama_{self.provider}_results"


def main():
    """Run experiments with Llama API."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run maze solving experiments with Llama models')
    parser.add_argument('--provider', type=str, default=LlamaProvider.NEBIUS,
                       choices=[LlamaProvider.NEBIUS, LlamaProvider.NVIDIA_NIM, LlamaProvider.NOVITA],
                       help='Llama provider to use')
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
    solver = LlamaAPISolver(provider=args.provider)
    
    # Run experiments
    solver.run_all_experiments(
        encoding_types=args.encoding,
        sizes=sizes,
        shapes=None  # Use all shapes
    )


if __name__ == "__main__":
    main() 