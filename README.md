# Know What, Know How: Disentangling Conceptual and Procedural Knowledge in Mazes

[![Repository Status](https://img.shields.io/badge/Status-Production%20Ready-green)](#)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](#)
[![Analysis](https://img.shields.io/badge/Analysis-Complete-success)](#)

**Do AI models truly understand what they're doing, or are they just following procedures?** This research investigates a fundamental question about AI intelligence through an elegant maze experiment that reveals a dramatic gap between procedural and conceptual knowledge.

While humans can both solve mazes AND understand their shapes, our experiments show that even state-of-the-art language models fail catastrophically at recognizing patterns they just navigated through. This repository contains the complete experimental framework, data from 26 human participants and multiple AI models, and analysis pipeline that demonstrates this striking cognitive dissociation.

**🎯 Ready to Run**: This repository includes complete experimental data from 26 human participants and multiple LLM results, with all analysis scripts tested and working. Simply run `make analysis` to reproduce all results and figures.

## 🧠 The Research Question

**Can AI models truly understand concepts, or are they just stochastic parrots?**

This experiment reveals a striking dissociation: while AI models can navigate mazes step-by-step with reasonable success, they fundamentally fail to recognize or reproduce the shapes those mazes form - tasks that humans perform effortlessly.

### Key Findings at a Glance

| Task | Human Performance | Best AI Performance | Model |
|------|------------------|---------------------|-------|
| **Maze Solving** (Procedural) | 99.4% | 91.7% | Claude Opus (vision) |
| **Shape Recognition** (Conceptual) | 99.7% | 19.0% | GPT-4o (coord_list) |
| **Shape Generation** (Conceptual) | 99.4% | 13.0% | Claude Sonnet (matrix) |

**The dissociation is extreme**: Even Claude Opus with vision capabilities achieves 91.7% on maze solving but only 1.5% on shape recognition.

### The Three-Task Paradigm

1. **Maze Solving** - Navigate from start to goal through corridors
2. **Shape Recognition** - Identify what shape the maze corridors form (square, cross, spiral, triangle, C, or Z)
3. **Shape Generation** - Create a new maze with the same shape pattern

### Why This Matters

This research demonstrates that current LLMs:
- Excel at following procedural rules but lack understanding
- Process information locally and sequentially rather than forming global representations  
- Cannot abstract patterns from the details they process

The implications extend beyond mazes: AI systems that can follow recipes but not understand cuisine, apply rules but not grasp principles, navigate details but miss the big picture.

## Features

- **Data Aggregation**: Ingests raw experimental data from human participants and multiple LLM providers (Claude, GPT-4, Llama, Gemini).
- **Multi-Modal Support**: Analyzes performance across different input formats (matrix, coordinate list, and vision).
- **Duplicate Handling**: Intelligently filters duplicate trials from human data, selecting the "best" attempt based on a composite quality score.
- **Edit Distance Calculation**: Computes the `path_edit_distance` between participant-generated mazes and a set of canonical correct mazes, with caching for performance.
- **Statistical Analysis**: Implements robust Generalized Linear Mixed Models (GLMMs) to analyze success rates and a Beta Regression model for edit distances.
- **Visualization**: Generates a suite of publication-quality figures to visualize overall performance, the effects of maze shape and encoding format, and edit distance distributions.
- **Automated Pipeline**: A simple `Makefile` interface to run the entire analysis from start to finish with a single command.
- **Configuration Driven**: Centralized configuration for all paths, model names, and experimental parameters in `config.py`.
- **Secure API Key Management**: Uses `.env` files for secure handling of API keys, with an example file provided.
- **Repository Management**: Comprehensive `.gitignore` file and clean repository structure excluding cache files, secrets, and development artifacts.

## Directory Structure

```
.
├── analysis/             # All data analysis and visualization scripts
│   ├── core/             # Main pipeline runner
│   ├── statistical/      # Statistical modeling scripts (GLMMs)
│   ├── utils/            # Helper scripts (data aggregation, report generation)
│   └── visualization/    # Figure generation scripts
├── core/                 # Core utilities (maze generation, solution verification)
├── data/                 # Raw experimental data
│   ├── experiment_mazes/ # The .npy maze files used in experiments
│   ├── human_results/    # Raw JSON data from human participants
│   └── machine_results/  # Raw .txt outputs from LLM providers
├── experiments/          # Scripts for running experiments
├── infer/                # LLM inference scripts
├── scripts/              # Standalone utility scripts (e.g., edit distance logic)
├── tests/                # Unit and integration tests
├── analysis_results/     # OUTPUT: All generated figures, reports, and processed data
│   ├── figures/
│   └── processed_data/
├── config.py             # Central configuration for all paths and parameters
├── Makefile              # Automation for setup, analysis, and cleaning
├── requirements.txt      # Main Python dependencies
└── README.md             # This file
```

## Experimental Design

### Participants
- **26 Human participants** using visual interface
- **4 AI models tested**:
  - **Claude 3.5 Sonnet** - with text and vision capabilities
  - **Claude Opus** - with text and vision capabilities  
  - **GPT-4o** - text-only
  - **Llama 3.1** - text-only

### Maze Specifications
- **Sizes**: 5×5 and 7×7 grids
- **Shapes**: Square, Cross, Spiral, Triangle, C, and Z
- **Representations**: 
  - Matrix format (0s and 1s)
  - Coordinate lists
  - Visual images (for Claude models only)

### Statistical Rigor
- ~1,800 trials per model
- Generalized Linear Mixed Models (GLMMs) accounting for maze difficulty
- Edit distance metrics for generation quality
- Comprehensive duplicate handling for human data

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd KnowWhat
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n spiral python=3.10 numpy pandas matplotlib seaborn scipy -y
    conda activate spiral
    ```

3.  **Install additional dependencies:**
    ```bash
    make install
    ```
    This will install packages from the root `requirements.txt` as well as the one in the `infer/` directory.

4.  **Set up API Keys (optional, only needed for running new inferences):**
    ```bash
    cp infer/.env.example infer/.env
    ```
    Edit `infer/.env` and add your keys for OpenAI, Anthropic, etc.

## Usage

The primary interaction with the project is through the `Makefile`.

### Run the Full Analysis Pipeline

This is the main command you will use. It runs all steps in the correct order: calculates edit distances, aggregates data, runs statistical models, generates all figures, and creates a final markdown report.

```bash
make analysis
```

The output will be saved to the `analysis_results/` directory:
- `figures/`: Publication-quality figures (PNG format)
  - `figure1_overall_performance.png`: Bar chart comparing human and LLM performance across tasks
  - `figure2_shape_format_effects.png`: Faceted plot showing effects of maze shape and encoding format
  - `figure3_edit_distance.png`: Violin plot of edit distances for the generation task
  - `supp_figure_size_effects.png`: Supplementary figure showing performance by maze size
- `processed_data/`: Aggregated data files in pickle and CSV formats
- `reports/`: Statistical analysis results and markdown reports

### Individual Analysis Steps

You can also run individual components:

```bash
# Calculate edit distances for human and machine data
python scripts/edit_distance.py
python analysis/utils/calculate_human_edit_distances.py

# Aggregate all data
python analysis/utils/data_aggregation.py

# Generate figures only
python analysis/visualization/generate_figures.py

# Run statistical analyses
python analysis/statistical/run_all_analyses.py
```

### Visualize Participant Details

To generate detailed visual breakdowns for individual human participant trials:

```bash
make viz
```

This creates PNG files in `analysis_results/participant_visualizations/` showing each participant's:
- Step-by-step maze solution
- Shape recognition response
- Generated maze (if applicable)

### Clean Up

To remove all generated files and caches:

```bash
make clean
```

This will delete the `analysis_results/` directory, `__pycache__` directories, and other temporary files.

### Run Human Experiments

To run the human experiment interface (after fixing the imports):

```bash
cd experiments
python human_test.py
```

This launches a Gradio web interface for collecting human maze-solving data that matches the format used in the analysis pipeline. The interface includes:
- Maze solving with directional controls
- Shape recognition tasks
- Coordinate-based maze generation
- Automatic result saving to `data/human_results/`

**Note**: Requires `gradio` package: `pip install gradio`

## Configuration

All project settings are centralized in `config.py`:

- **File paths**: Data directories, output locations
- **Experiment parameters**: 
  - Maze sizes: 5×5, 7×7
  - Maze shapes: square, cross, spiral, triangle, C, Z
  - Encoding types: matrix, coord_list, vision
- **Models tested**:
  - Claude 3.5 Sonnet (Anthropic)
  - Claude Opus (Anthropic)
  - GPT-4o (OpenAI)
  - Llama 3.1 (Meta)
- **Vision capabilities**: Only Claude models support image inputs

## Data Format

### Human Data
- JSON files in `data/human_results/<participant_id>/`
- Contains moves, completion times, recognized shapes, and generated mazes

### Machine Data
- Text files in `data/machine_results/<encoding>_<model>_results/`
- Organized by maze size, shape, and trial number
- Contains model outputs with SUCCESS/FAIL markers

### Edit Distances
- Calculated using XOR-based path comparison
- Normalized to 0-1 range (0 = perfect match, 1 = completely different)
- Cached for performance optimization

## Recent Updates

- **Fixed error bars in Figure 1**: Replaced hardcoded confidence intervals with proper binomial confidence intervals using Wilson score method - error bars now vary realistically based on sample size and success rates
- **Updated human experiment interface**: Fixed all imports in `experiments/human_test.py` to work with current repository structure, updated path handling to use `pathlib.Path` objects
- **Added comprehensive .gitignore**: Created industry-standard `.gitignore` file excluding `__pycache__`, environment files, IDE files, and other development artifacts
- **Fixed color mapping**: Each model variant now has a unique color in Figure 1
- **Resolved edit distance loading**: Machine edit distances now properly appear in Figure 3
- **Improved data aggregation**: Better handling of vision model results
- **Enhanced visualization**: Clearer labeling and consistent color schemes across figures
- **Repository cleanup**: Removed existing `__pycache__` directories and ensured clean git history

## Troubleshooting

### Environment Issues
If you encounter numpy/pandas compatibility errors:
```bash
conda remove -n spiral --all -y
conda create -n spiral python=3.10 numpy pandas matplotlib seaborn scipy -y
conda activate spiral
make install
```

### Import Errors in Human Experiments
If you encounter import errors when running `experiments/human_test.py`:
```bash
cd experiments
python -c "import sys; sys.path.append('..'); from core.maze_generator import all_square_path_mazes; print('Imports working')"
```
The imports have been fixed to work with the current repository structure using absolute imports from the `core` module.

### Missing Dependencies for Human Interface
If running human experiments fails with missing gradio:
```bash
pip install gradio
```

### Error Bars Look Identical in Figure 1
This has been fixed! Error bars now use proper binomial confidence intervals and will vary based on actual data variance and sample sizes.

### Git Issues with Cache Files
The repository now includes a comprehensive `.gitignore` file that excludes:
- `__pycache__/` directories  
- Environment files (`.env`)
- IDE configuration files
- Temporary and cache files

If you have existing cache files, run:
```bash
make clean
```

### Missing Edit Distances
If Figure 3 shows only human data:
1. Ensure edit distance files exist in `analysis_results/processed_data/`
2. Re-run: `python scripts/edit_distance.py`
3. Re-aggregate: `python analysis/utils/data_aggregation.py`

### API Keys
If running new inferences, ensure your `.env` file in `infer/` contains valid API keys for the providers you're using.