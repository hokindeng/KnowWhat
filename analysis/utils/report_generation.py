#!/usr/bin/env python3
"""
Create comprehensive analysis report.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (PROCESSED_DATA_DIR, FIGURES_DIR, STATISTICAL_RESULTS_FILE,
                   AGGREGATED_DATA_FILE, ANALYSIS_RESULTS_DIR, setup_logging)

# Setup logger
logger = setup_logging(__name__)

def load_results():
    """Load all analysis results."""
    try:
        overall_rates = pd.read_pickle(PROCESSED_DATA_DIR / "overall_rates.pkl")
        aggregated_data = pd.read_pickle(PROCESSED_DATA_DIR / "aggregated_data.pkl")
        
        # Load statistical results if available
        stats_summary_path = PROCESSED_DATA_DIR / STATISTICAL_RESULTS_FILE
        stats_summary = None
        if stats_summary_path.exists():
            with open(stats_summary_path, 'r') as f:
                stats_summary = f.read()
        
        return overall_rates, aggregated_data, stats_summary
    except Exception as e:
        print(f"Error loading results: {e}")
        return None, None, None

def get_generation_quality_summary(aggregated_data):
    """Generate a dynamic summary paragraph for generation quality."""
    if 'edit_distance_mean' not in aggregated_data.columns:
        return ""
        
    edit_data = aggregated_data[aggregated_data['edit_distance_mean'].notna()]
    if edit_data.empty:
        return ""

    human_data = edit_data[edit_data['participant_type'] == 'human']
    llm_data = edit_data[edit_data['participant_type'] == 'llm']

    summary_parts = []
    if not human_data.empty:
        human_mean_dist = human_data['edit_distance_mean'].mean()
        summary_parts.append(f"humans generate highly canonical maze representations (mean distance: {human_mean_dist:.3f})")

    if not llm_data.empty:
        llm_summary = []
        # Get top 2-3 LLMs by performance (lowest mean edit distance)
        top_llms = llm_data.groupby('model_name')['edit_distance_mean'].mean().nsmallest(3)
        for model, dist in top_llms.items():
            llm_summary.append(f"{model}: {dist:.3f}")
        
        if llm_summary:
            llm_text = ", ".join(llm_summary)
            summary_parts.append(f"while LLMs show greater variation from exemplars ({llm_text}), indicating differences in conceptual precision.")

    if not summary_parts:
        return ""

    full_summary = "4. **Generation Quality**: Edit distance analysis reveals that " + ", ".join(summary_parts)
    return full_summary

def generate_report(overall_rates, aggregated_data, stats_summary):
    """Generate the analysis report."""
    report = []
    
    # Header
    report.append("# Maze Understanding Analysis: Humans vs. Language Models")
    report.append(f"\n*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    report.append("This analysis examines procedural versus conceptual knowledge in maze understanding tasks, ")
    report.append("comparing human performance with various language models (LLMs) across three tasks: ")
    report.append("maze solving (procedural), shape recognition (conceptual), and maze generation (conceptual). ")
    report.append("Edit distance analysis quantifies how closely generated mazes match canonical exemplars.\n")
    
    # Methodology Note
    report.append("### Data Processing Notes\n")
    report.append("- **Duplicate Trial Handling**: Some human participants completed the same maze multiple times. ")
    report.append("For analysis, we selected the best attempt for each participant-maze combination based on ")
    report.append("overall task performance (solving, recognition, generation success) and edit distance for generation tasks.\n")
    report.append("- **Expected Data**: Each participant should complete 12 unique mazes (2 sizes × 6 shapes).\n")
    
    # Key Findings
    report.append("## Key Findings\n")
    
    # 1. Overall Performance
    report.append("### 1. Overall Task Performance\n")
    
    # Create summary table
    summary_data = []
    for _, row in overall_rates.iterrows():
        summary_data.append({
            'Participant': row['label'],
            'Solve Rate': f"{row['solve_rate']:.1%}" if 'solve_rate' in row else "N/A",
            'Recognition Rate': f"{row['recognize_rate']:.1%}" if 'recognize_rate' in row else "N/A",
            'Generation Rate': f"{row['generate_rate']:.1%}" if 'generate_rate' in row else "N/A"
        })
    
    summary_df = pd.DataFrame(summary_data)
    report.append(summary_df.to_markdown(index=False))
    report.append("")
    
    # 2. Procedural vs Conceptual Dissociation
    report.append("\n### 2. Procedural-Conceptual Knowledge Dissociation\n")
    
    # Calculate dissociation scores
    dissociation_scores = []
    for _, row in overall_rates.iterrows():
        if all(col in row for col in ['solve_rate', 'recognize_rate', 'generate_rate']):
            procedural = row['solve_rate']
            conceptual_avg = (row['recognize_rate'] + row['generate_rate']) / 2
            dissociation = procedural - conceptual_avg
            
            dissociation_scores.append({
                'Participant': row['label'],
                'Procedural (Solve)': f"{procedural:.1%}",
                'Conceptual (Avg)': f"{conceptual_avg:.1%}",
                'Dissociation Score': f"{dissociation:+.1%}"
            })
    
    if dissociation_scores:
        dissociation_df = pd.DataFrame(dissociation_scores)
        report.append(dissociation_df.to_markdown(index=False))
        report.append("\n*Note: Positive dissociation scores indicate better procedural than conceptual performance.*\n")
    
    # 3. Effect of Presentation Format
    report.append("### 3. Effect of Presentation Format\n")
    
    # Focus on models with multiple formats
    format_effects = []
    for model in ['claude-3.5-sonnet', 'gpt-4o', 'llama-3.1']:
        model_data = aggregated_data[aggregated_data['model_name'] == model]
        if len(model_data) > 0:
            for task in ['solve', 'recognize', 'generate']:
                rate_col = f'{task}_rate'
                if rate_col in model_data.columns:
                    by_format = model_data.groupby('encoding_type')[rate_col].mean()
                    for fmt, rate in by_format.items():
                        format_effects.append({
                            'Model': model,
                            'Task': task.capitalize(),
                            'Format': fmt,
                            'Success Rate': f"{rate:.1%}"
                        })
    
    if format_effects:
        format_df = pd.DataFrame(format_effects)
        # Pivot for better readability
        format_pivot = format_df.pivot_table(
            index=['Model', 'Task'], 
            columns='Format', 
            values='Success Rate',
            aggfunc='first'
        )
        report.append(format_pivot.to_markdown())
        report.append("")
    
    # 4. Effect of Maze Shape
    report.append("### 4. Effect of Maze Shape\n")
    
    # Aggregate by shape
    shape_effects = aggregated_data.groupby(['participant_type', 'maze_shape']).agg({
        'solve_rate': 'mean',
        'recognize_rate': 'mean', 
        'generate_rate': 'mean'
    }).round(3)
    
    report.append("Average performance by maze shape:\n")
    report.append(shape_effects.to_markdown())
    report.append("")
    
    # 5. Edit Distance Analysis
    report.append("### 5. Edit Distance Analysis (Generation Task)\n")
    
    if 'edit_distance_mean' not in aggregated_data.columns:
        report.append("*Edit distance data not available. Run calculate_edit_distances.py to generate this data.*\n")
    else:
        edit_dist_data = aggregated_data[aggregated_data['edit_distance_mean'].notna()]
        if len(edit_dist_data) > 0:
            edit_summary = edit_dist_data.groupby(['model_name', 'encoding_type'])['edit_distance_mean'].agg([
                'mean', 'std', 'min', 'max'
            ]).round(3)
            
            report.append("Edit distance statistics by model and encoding:\n")
            report.append(edit_summary.to_markdown())
            report.append("\n*Note: Lower edit distance indicates generated mazes closer to valid exemplars.*\n")
        else:
            report.append("*No edit distance data available in the dataset.*\n")
    
    # Statistical Analysis Summary
    if stats_summary:
        report.append("## Statistical Analysis Summary\n")
        report.append("```")
        report.append(stats_summary[:2000])  # First 2000 chars
        if len(stats_summary) > 2000:
            report.append("\n... (truncated)")
        report.append("```\n")
    
    # Appendix
    report.append("## Appendix: Data Summary\n")
    
    # Sample sizes
    report.append("### Sample Sizes\n")
    
    # Use the 'label' for grouping to ensure family names are used
    overall_rates['group_label'] = overall_rates['label'].apply(lambda x: x.split(' (')[0])
    sample_sizes = overall_rates.groupby('group_label').size()
    
    report.append(f"Total experimental conditions analyzed: {len(aggregated_data)}\n")
    report.append("\nConditions per participant type:\n")
    report.append(sample_sizes.to_markdown())
    
    return "\n".join(report)

def main():
    """Generate comprehensive analysis report."""
    logger.info("Generating analysis report...")
    
    # Load results
    overall_rates, aggregated_data, stats_summary = load_results()
    
    if overall_rates is None or aggregated_data is None:
        print("Failed to load analysis results. Make sure to run the analysis pipeline first.")
        return
    
    # Generate report
    report_content = generate_report(overall_rates, aggregated_data, stats_summary)
    
    # Save report
    REPORT_PATH = ANALYSIS_RESULTS_DIR / "analysis_report.md"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(REPORT_PATH, 'w') as f:
        f.write(report_content)
    
    print(f"Report saved to: {REPORT_PATH}")
    
    # Also create a summary CSV for easy viewing
    summary_path = PROCESSED_DATA_DIR / "analysis_summary.csv"
    overall_rates.to_csv(summary_path, index=False)
    print(f"Summary CSV saved to: {summary_path}")

if __name__ == "__main__":
    main() 