"""
Central configuration for the KnowWhat project.
All project-wide settings and paths should be defined here.
"""

import os
from pathlib import Path

# Project root directory (automatically detected)
PROJECT_ROOT = Path(__file__).parent.absolute()

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
CORE_DIR = PROJECT_ROOT / "core"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
INFER_DIR = PROJECT_ROOT / "infer"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCS_DIR = PROJECT_ROOT / "docs"

# Data subdirectories
EXPERIMENT_MAZES_DIR = DATA_DIR / "experiment_mazes"
HUMAN_RESULTS_DIR = DATA_DIR / "human_results"
MACHINE_RESULTS_DIR = DATA_DIR / "machine_results"
HUMAN_DATA_ANALYSIS_DIR = DATA_DIR / "human_data_analysis"

# Analysis subdirectories
ANALYSIS_CORE_DIR = ANALYSIS_DIR / "core"
ANALYSIS_VIS_DIR = ANALYSIS_DIR / "visualization"
ANALYSIS_STATS_DIR = ANALYSIS_DIR / "statistical"
ANALYSIS_UTILS_DIR = ANALYSIS_DIR / "utils"
ANALYSIS_REPORTS_DIR = ANALYSIS_DIR / "reports"

# Analysis results directory
ANALYSIS_RESULTS_DIR = PROJECT_ROOT / "analysis_results"

# Processed data directory
PROCESSED_DATA_DIR = ANALYSIS_RESULTS_DIR / "processed_data"

# Figures directory for visualization outputs
FIGURES_DIR = ANALYSIS_RESULTS_DIR / "figures"

# Create directories if they don't exist
for dir_path in [PROCESSED_DATA_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Experiment parameters
MAZE_SIZES = [(5, 5), (7, 7), (9, 9)]
MAZE_SHAPES = ["square", "cross", "spiral", "triangle", "C", "Z"]
ENCODING_TYPES = ["matrix", "coord_list", "vision"]
TASKS = ["solve", "recognize", "generate"]

# Model configurations
LLM_MODELS = {
    "gpt-4o": "openai",
    "claude-3.5-sonnet": "anthropic",
    "claude-4-opus": "anthropic",
    "gemini-1.5-pro": "google",
    "llama-3.1": "llama"
}

MODEL_FAMILY = {
    "claude-3.5-sonnet": "Claude",
    "claude-4-opus": "Claude",
    "gpt-4o": "GPT-4",
    "gemini-1.5-pro": "Gemini",
    "llama-3.1": "Llama"
}

VISION_MODELS = {
    "sonnet": "claude-3.5-sonnet",
    "opus": "claude-4-opus"
}

# Analysis parameters
SIGNIFICANCE_LEVEL = 0.05
RANDOM_SEED = 42

# File paths for commonly used files
EDIT_DISTANCES_FILE = PROCESSED_DATA_DIR / "maze_edit_distances.json"
HUMAN_EDIT_DISTANCES_FILE = PROCESSED_DATA_DIR / "human_edit_distances.json"

# Output file patterns
TRIAL_DATA_FILE = PROCESSED_DATA_DIR / "all_trials.pkl"
AGGREGATED_DATA_FILE = PROCESSED_DATA_DIR / "aggregated_data.pkl"
OVERALL_RATES_FILE = PROCESSED_DATA_DIR / "overall_rates.pkl"
STATISTICAL_RESULTS_FILE = PROCESSED_DATA_DIR / "statistical_results_summary.txt"

def get_project_root():
    """Get the project root directory."""
    return PROJECT_ROOT

def get_data_path(*args):
    """Get a path relative to the data directory."""
    return DATA_DIR.joinpath(*args)

def get_analysis_path(*args):
    """Get a path relative to the analysis directory."""
    return ANALYSIS_DIR.joinpath(*args)

def get_results_path(model_name, encoding_type, *args):
    """Get a path to model results."""
    if encoding_type == "vision":
        # Handle special vision result directory names
        if model_name == "anthropic":
            # Check for both opus and sonnet vision results
            opus_dir = MACHINE_RESULTS_DIR / "opus_vision_results" / Path(*args)
            sonnet_dir = MACHINE_RESULTS_DIR / "sonnet_vision_results" / Path(*args)
            # Return the one that exists, preferring sonnet
            if sonnet_dir.exists():
                return sonnet_dir
            else:
                return opus_dir
        else:
            results_dir = f"vision_{model_name}_results"
    else:
        # Map provider name to directory name
        provider_map = {
            "anthropic": "claude",
            "openai": "openai", 
            "google": "gemini",
            "llama": "llama"
        }
        provider = provider_map.get(model_name, model_name)
        results_dir = f"{encoding_type}_{provider}_results"
    return MACHINE_RESULTS_DIR / results_dir / Path(*args)

# Logging configuration
import logging

def setup_logging(name=None, level=logging.INFO):
    """Setup logging configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger 