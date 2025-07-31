#!/usr/bin/env python3
"""
Generate publication-quality figures for maze understanding analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (AGGREGATED_DATA_FILE, FIGURES_DIR, STATISTICAL_RESULTS_FILE, 
                   HUMAN_RESULTS_DIR, MACHINE_RESULTS_DIR, setup_logging)

# Setup logger
logger = setup_logging(__name__)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ==================== Configuration ====================
# Remove hardcoded paths - now using config imports
# BASE_DIR = Path("/Users/access/spiral_project/spiral_analysis")
# DATA_DIR = BASE_DIR / "processed_data"
# FIG_DIR = BASE_DIR / "figures"

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
FONT_SIZE = 12

# Color schemes
PARTICIPANT_COLORS = {
    'Human': '#2E7D32',  # Green

    # Unique colors for each model variant
    'Claude-Sonnet (vision)': '#1f77b4', # Muted Blue
    'Claude-Opus (vision)': '#aec7e8', # Light Blue
    'Claude (matrix)': '#ff7f0e', # Safety Orange
    'Claude (coord_list)': '#ffbb78', # Light Orange

    'GPT-4 (matrix)': '#d62728', # Brick Red
    'GPT-4 (coord_list)': '#ff9896', # Light Red

    'Llama (matrix)': '#9467bd', # Muted Purple
    'Llama (coord_list)': '#c5b0d5', # Light Purple

    'Gemini (matrix)': '#8c564b', # Brown
    'Gemini (coord_list)':'#c49c94', # Light Brown
}

TASK_ORDER = ['solve', 'recognize', 'generate']
TASK_LABELS = {
    'solve': 'Solving',
    'recognize': 'Recognition', 
    'generate': 'Generation'
}

SHAPE_ORDER = ['square', 'cross', 'spiral', 'triangle', 'C', 'Z']
SHAPE_LABELS = {
    'square': 'Square',
    'cross': 'Cross',
    'spiral': 'Spiral',
    'triangle': 'Triangle',
    'C': 'C-shape',
    'Z': 'Z-shape'
}

# ==================== Helper Functions ====================

def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    """Calculate bootstrap confidence intervals."""
    bootstrapped = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrapped.append(np.mean(sample))
    
    lower = np.percentile(bootstrapped, (100-ci)/2)
    upper = np.percentile(bootstrapped, 100-(100-ci)/2)
    
    return lower, upper

def calculate_binomial_ci(success_rate, n_trials, confidence=0.95):
    """Calculate binomial confidence interval using Wilson score method."""
    if n_trials == 0:
        return 0, 0
    
    from scipy import stats
    
    # Number of successes
    n_success = int(success_rate * n_trials)
    
    # Wilson score interval
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    p = success_rate
    n = n_trials
    
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    
    lower_bound = (centre_adjusted_probability - adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + adjusted_standard_deviation) / denominator
    
    # Return error (distance from mean)
    lower_error = max(0, success_rate - lower_bound)
    upper_error = max(0, upper_bound - success_rate)
    
    return lower_error, upper_error

def load_data():
    """Load processed data."""
    logger.info("Loading aggregated data from config paths")
    df = pd.read_csv(AGGREGATED_DATA_FILE)
    return df

# ==================== Figure 1: Overall Performance ====================

def create_figure1_overall_performance(overall_rates):
    """
    Figure 1: Overall performance across tasks (humans vs. LLMs)
    Grouped bar chart showing mean success rates with error bars
    """
    print("Creating Figure 1: Overall Performance...")
    
    # Prepare data for plotting
    plot_data = []
    
    for _, row in overall_rates.iterrows():
        for task in TASK_ORDER:
            rate_col = f'{task}_rate'
            trials_col = f'{task}_trials'
            
            if rate_col in row and trials_col in row:
                plot_data.append({
                    'Participant': row['label'],
                    'Task': TASK_LABELS[task],
                    'Success Rate': row[rate_col],
                    'N_Trials': row[trials_col]
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8)) # Widen the figure
    
    # Define participant order with new specific labels
    participant_order = ['Human', 
                         'Claude-Sonnet (vision)', 'Claude-Opus (vision)',
                         'Claude (matrix)', 'Claude (coord_list)',
                         'GPT-4 (matrix)', 'GPT-4 (coord_list)', 
                         'Llama (matrix)', 'Llama (coord_list)',
                         'Gemini (matrix)', 'Gemini (coord_list)']
    
    # Filter to existing participants
    existing_participants = [p for p in participant_order if p in plot_df['Participant'].unique()]
    
    # Set up bar positions dynamically
    n_groups = len(existing_participants)
    n_tasks = len(TASK_ORDER)
    total_width = 0.8  # Total width for all bars in a group
    bar_width = total_width / n_groups
    
    x = np.arange(n_tasks)
    
    # Plot bars for each participant
    for i, participant in enumerate(existing_participants):
        participant_data = plot_df[plot_df['Participant'] == participant]
        
        means = []
        lower_errors = []
        upper_errors = []
        
        for task in TASK_LABELS.values():
            task_data = participant_data[participant_data['Task'] == task]
            if not task_data.empty:
                success_rate = task_data['Success Rate'].values[0]
                n_trials = task_data['N_Trials'].values[0]
                
                means.append(success_rate)
                
                # Calculate proper confidence intervals
                lower_err, upper_err = calculate_binomial_ci(success_rate, n_trials)
                lower_errors.append(lower_err)
                upper_errors.append(upper_err)
            else:
                means.append(0)
                lower_errors.append(0)
                upper_errors.append(0)
        
        # Determine color dynamically
        color = PARTICIPANT_COLORS.get(participant, '#888888')
        
        # Plot bars with asymmetric error bars
        offset = (i - n_groups / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width, 
                      label=participant, color=color, alpha=0.8,
                      yerr=[lower_errors, upper_errors], capsize=5)
    
    # Customize plot
    ax.set_ylabel('Success Rate', fontsize=14)
    ax.set_xlabel('Task Type', fontsize=14)
    ax.set_title('Task Performance: Humans vs. Language Models', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS.values())
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(axis='y', alpha=0.3)
    
    # Add significance indicators (simplified - would need actual stats)
    # This is a placeholder for where statistical significance would be shown
    
    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
             frameon=True, fancybox=True, shadow=True)
    
    # Add sample size annotations
    y_offset = -0.15
    for i, participant in enumerate(existing_participants):
        participant_data = plot_df[plot_df['Participant'] == participant]
        for j, task in enumerate(TASK_LABELS.values()):
            task_data = participant_data[participant_data['Task'] == task]
            if not task_data.empty:
                n_trials = task_data['N_Trials'].values[0]
                offset = (i - n_groups/2 + 0.5) * bar_width
                ax.text(j + offset, y_offset, f'n={n_trials}', 
                       ha='center', va='top', fontsize=8, rotation=90)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = FIGURES_DIR / f"figure1_overall_performance.{FIGURE_FORMAT}"
    plt.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    
    plt.close()

# ==================== Figure 2: Shape and Format Effects ====================

def create_figure2_shape_format_effects(aggregated_data):
    """
    Figure 2: Impact of Maze Shape and Presentation Format on LLM Task Performance
    Faceted plot showing performance by shape and format
    """
    print("Creating Figure 2: Shape and Format Effects...")
    
    # Filter to LLM data only and map to model families
    from config import MODEL_FAMILY
    llm_data = aggregated_data[aggregated_data['participant_type'] == 'llm'].copy()
    llm_data['family'] = llm_data['model_name'].apply(lambda m: MODEL_FAMILY.get(m, m))

    # We want one panel per family (Claude-4, GPT-4, etc.)
    models = sorted(llm_data['family'].unique())
    tasks = TASK_ORDER
    
    if not models:
        print("Warning: No LLM data found for Figure 2.")
        return

    fig, axes = plt.subplots(len(tasks), len(models), figsize=(4 * len(models), 10), squeeze=False)
    
    for row, task in enumerate(tasks):
        for col, model in enumerate(models):
            ax = axes[row, col] if len(tasks) > 1 else axes[col]
            
            # Get data for this model family and task
            model_data = llm_data[llm_data['family'] == model]
            
            # Prepare data for plotting
            plot_data = []
            for _, row_data in model_data.iterrows():
                rate_col = f'{task}_rate'
                if rate_col in row_data:
                    # Derive a human-readable format label
                    enc = row_data['encoding_type']
                    if enc == 'vision':
                        if row_data['model_name'] == 'claude-3.5-sonnet':
                            enc_label = 'sonnet-vision'
                        elif row_data['model_name'] == 'claude-4-opus':
                            enc_label = 'opus-thinking'
                        else:
                            enc_label = 'vision'
                    else:
                        enc_label = enc  # matrix / coord_list

                    plot_data.append({
                        'Shape': row_data['maze_shape'],
                        'Format': enc_label,
                        'Success Rate': row_data[rate_col],
                        'Size': row_data['maze_size']
                    })
            
            plot_df = pd.DataFrame(plot_data)
            
            if not plot_df.empty:
                # Create grouped bar plot
                shape_format_means = plot_df.groupby(['Shape', 'Format'])['Success Rate'].mean().reset_index()
                
                # Pivot for easier plotting
                # Ensure a consistent format order
                format_order = ['coord_list', 'matrix', 'sonnet-vision', 'opus-thinking']
                pivot_df = shape_format_means.pivot(index='Shape', columns='Format', values='Success Rate')
                # Reindex columns to desired order if they exist
                pivot_df = pivot_df.reindex(columns=[f for f in format_order if f in pivot_df.columns])
                
                # Plot
                pivot_df.plot(kind='bar', ax=ax, width=0.8)
                
                # Customize subplot
                ax.set_ylim(0, 1.05)
                ax.set_xlabel('')
                
                if col == 0:
                    ax.set_ylabel(f'{TASK_LABELS[task]}\nSuccess Rate', fontsize=12)
                else:
                    ax.set_ylabel('')
                
                if row == 0:
                    ax.set_title(f'{model}', fontsize=14)
                
                if row == len(tasks) - 1:
                    ax.set_xlabel('Maze Shape', fontsize=12)
                    ax.set_xticklabels([SHAPE_LABELS[s] for s in SHAPE_ORDER], rotation=45, ha='right')
                else:
                    ax.set_xticklabels([])
                
                ax.legend(title='Format', loc='upper right', fontsize=10)
                ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('LLM Performance by Maze Shape and Presentation Format', fontsize=16, y=0.995)
    plt.tight_layout()
    
    # Save figure
    fig_path = FIGURES_DIR / f"figure2_shape_format_effects.{FIGURE_FORMAT}"
    plt.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    
    plt.close()

# ==================== Figure 3: Edit Distance Analysis ====================

def create_figure3_edit_distance(aggregated_data):
    """
    Figure 3: Edit distance analysis for generation task
    Raincloud plot showing distribution of edit distances
    """
    print("Creating Figure 3: Edit Distance Analysis...")
    
    # Check if edit distance data exists
    if 'edit_distance_mean' not in aggregated_data.columns:
        print("No edit distance data available in the dataset!")
        print("Skipping Figure 3. Run calculate_edit_distances.py to generate edit distance data.")
        return
    
    # Filter to generation task with edit distances
    gen_data = aggregated_data[(aggregated_data['edit_distance_mean'].notna())].copy()
    
    if len(gen_data) == 0:
        print("No data with edit distances available to plot!")
        print("Skipping Figure 3.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for plotting, including humans now
    from config import MODEL_FAMILY
    plot_data = []
    for _, row in gen_data.iterrows():
        if row['participant_type'] == 'llm':
            # Match the label generation from data_aggregation.py
            family = MODEL_FAMILY.get(row['model_name'], row['model_name'])
            if row['encoding_type'] == 'vision':
                if 'sonnet' in row['model_name']:
                    label = f"{family}-Sonnet (vision)"
                elif 'opus' in row['model_name']:
                    label = f"{family}-Opus (vision)"
                else:
                    label = f"{family} (vision)"
            else:
                label = f"{family} ({row['encoding_type']})"
            plot_data.append({
                'Model': label,
                'Edit Distance': row['edit_distance_mean'],
                'Shape': row['maze_shape']
            })
        elif row['participant_type'] == 'human':
            plot_data.append({
                'Model': 'Human',
                'Edit Distance': row['edit_distance_mean'],
                'Shape': row['maze_shape']
            })

    plot_df = pd.DataFrame(plot_data)

    if plot_df.empty:
        print("Warning: Could not create dataframe for Figure 3. No valid edit distance data found.")
        return

    # Create violin plot with points
    model_order = ['Human', 
                   'Claude-Sonnet (vision)', 'Claude-Opus (vision)',
                   'Claude (matrix)', 'Claude (coord_list)',
                   'GPT-4 (matrix)', 'GPT-4 (coord_list)',
                   'Llama (matrix)', 'Llama (coord_list)',
                   'Gemini (matrix)', 'Gemini (coord_list)']
    
    existing_models = [m for m in model_order if m in plot_df['Model'].unique()]
    
    # Violin plot
    sns.violinplot(data=plot_df, x='Model', y='Edit Distance', 
                   order=existing_models, ax=ax, inner='box', alpha=0.6)
    
    # Add individual points
    sns.stripplot(data=plot_df, x='Model', y='Edit Distance',
                  order=existing_models, ax=ax, size=4, alpha=0.7, jitter=True)
    
    # Customize plot
    ax.set_ylabel('Edit Distance (0 = perfect match)', fontsize=14)
    ax.set_xlabel('Model and Encoding Type', fontsize=14)
    ax.set_title('Edit Distance Distribution for Generation Task', fontsize=16, pad=20)
    ax.set_ylim(-0.05, 1.05)
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add horizontal line at 0 for reference
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Perfect match')
    
    # Add mean values as text
    for i, model in enumerate(existing_models):
        model_data = plot_df[plot_df['Model'] == model]['Edit Distance']
        if len(model_data) > 0:
            mean_val = model_data.mean()
            ax.text(i, 1.02, f'μ={mean_val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = FIGURES_DIR / f"figure3_edit_distance.{FIGURE_FORMAT}"
    plt.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    
    plt.close()

# ==================== Supplementary Figures ====================

def create_supplementary_figures(aggregated_data, all_data):
    """Create additional figures for supplementary materials."""
    print("\nCreating supplementary figures...")
    
    # S1: Performance by maze size, broken down by model
    fig, ax = plt.subplots(figsize=(12, 8))
    
    size_data = aggregated_data.groupby(['participant_type', 'model_name', 'encoding_type', 'maze_size']).agg({
        'solve_rate': 'mean',
        'recognize_rate': 'mean',
        'generate_rate': 'mean'
    }).reset_index()
    
    size_data['label'] = size_data.apply(
        lambda row: 'Human' if row['participant_type'] == 'human' 
        else f"{row['model_name']} ({row['encoding_type']})", axis=1
    )

    # Plot size effects for each participant group
    for label in size_data['label'].unique():
        participant_data = size_data[size_data['label'] == label]
        if not participant_data.empty:
            for task in ['solve_rate', 'recognize_rate', 'generate_rate']:
                if task in participant_data.columns:
                    # Check for both sizes
                    values_5x5 = participant_data[participant_data['maze_size'] == '5x5'][task]
                    values_7x7 = participant_data[participant_data['maze_size'] == '7x7'][task]
                    
                    if not values_5x5.empty and not values_7x7.empty:
                        ax.plot([0, 1], [values_5x5.mean(), values_7x7.mean()], 'o-', 
                               label=f'{label} - {task.replace("_rate", "")}')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['5x5', '7x7'])
    ax.set_xlabel('Maze Size', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Supplementary Figure: Performance by Maze Size for Each Participant Group', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    
    fig_path = FIGURES_DIR / f"supp_figure_size_effects.{FIGURE_FORMAT}"
    plt.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()

# ==================== Main Function ====================

def main():
    """Generate all figures."""
    print("="*60)
    print("Generating Figures for Maze Understanding Analysis")
    print("="*60)
    
    # Ensure output directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    try:
        # Load aggregated data from pickle
        aggregated_data = pd.read_pickle(AGGREGATED_DATA_FILE)
        
        # Load the overall_rates file which has the correct labels
        from config import OVERALL_RATES_FILE
        if OVERALL_RATES_FILE.exists():
            overall_rates = pd.read_pickle(OVERALL_RATES_FILE)
        else:
            # Fallback for older data, now with corrected labeling
            overall_rates = aggregated_data.groupby(['participant_type', 'model_name', 'encoding_type']).agg({
                'solve_rate': 'mean',
                'recognize_rate': 'mean',
                'generate_rate': 'mean',
                'solve_trials': 'sum',
                'recognize_trials': 'sum',
                'generate_trials': 'sum'
            }).reset_index()
            from config import MODEL_FAMILY
            # This is now the single source of truth for labels, matching the aggregation script
            def get_label(row):
                if row['participant_type'] == 'human':
                    return 'Human'
                family = MODEL_FAMILY.get(row['model_name'], row['model_name'])
                if row['encoding_type'] == 'vision':
                    if 'sonnet' in row['model_name']:
                        return f"{family}-Sonnet (vision)"
                    if 'opus' in row['model_name']:
                        return f"{family}-Opus (vision)"
                    return f"{family} (vision)"
                return f"{family} ({row['encoding_type']})"
            overall_rates['label'] = overall_rates.apply(get_label, axis=1)

        # For compatibility with existing functions, use aggregated_data as all_data
        all_data = aggregated_data
        
    except FileNotFoundError:
        print(f"Error: Aggregated data file not found at {AGGREGATED_DATA_FILE}")
        print("Please run comprehensive_analysis.py first to generate the data.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Generate main figures
    create_figure1_overall_performance(overall_rates)
    create_figure2_shape_format_effects(aggregated_data)
    create_figure3_edit_distance(aggregated_data)
    
    # Generate supplementary figures
    create_supplementary_figures(aggregated_data, all_data)
    
    print("\nAll figures generated successfully!")
    print(f"Figures saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    main() 