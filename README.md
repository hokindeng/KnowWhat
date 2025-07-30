# Know What, Know How: Disentangling Conceptual and Procedural Knowledge in Mazes

A comprehensive research framework for studying maze navigation capabilities in humans and AI models, examining the dissociation between procedural knowledge (maze solving) and conceptual knowledge (shape recognition and generation). This project provides a full pipeline from data aggregation and cleaning to robust statistical analysis and publication-quality figure generation.

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

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd KnowWhatKnowHow
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

## Configuration

All project settings are centralized in `config.py`:

- **File paths**: Data directories, output locations
- **Experiment parameters**: 
  - Maze sizes: 5×5, 7×7
  - Maze shapes: square, cross, spiral, triangle, C, Z
  - Encoding types: matrix, coord_list, vision
- **Model mappings**:
  - `LLM_MODELS`: Maps model names to providers
  - `MODEL_FAMILY`: Groups models by family for visualization
  - `VISION_MODELS`: Specific vision-capable models

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

- **Fixed color mapping**: Each model variant now has a unique color in Figure 1
- **Resolved edit distance loading**: Machine edit distances now properly appear in Figure 3
- **Improved data aggregation**: Better handling of vision model results
- **Enhanced visualization**: Clearer labeling and consistent color schemes across figures

## Troubleshooting

### Environment Issues
If you encounter numpy/pandas compatibility errors:
```bash
conda remove -n spiral --all -y
conda create -n spiral python=3.10 numpy pandas matplotlib seaborn scipy -y
conda activate spiral
make install
```

### Missing Edit Distances
If Figure 3 shows only human data:
1. Ensure edit distance files exist in `analysis_results/processed_data/`
2. Re-run: `python scripts/edit_distance.py`
3. Re-aggregate: `python analysis/utils/data_aggregation.py`

### API Keys
If running new inferences, ensure your `.env` file in `infer/` contains valid API keys for the providers you're using.