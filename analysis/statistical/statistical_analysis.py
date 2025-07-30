#!/usr/bin/env python3
"""
Robust GLMM Analysis with better handling of separation and convergence issues.
Implements the analysis plan with appropriate simplifications.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LogisticRegression
import warnings
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import PROCESSED_DATA_DIR, STATISTICAL_RESULTS_FILE, setup_logging

warnings.filterwarnings('ignore')
logger = setup_logging(__name__)

def prepare_data_for_glmm(df):
    """Prepare and clean data for GLMM analysis."""
    # Create maze exemplar ID
    df['maze_exemplar_id'] = df['maze_file'].astype(str) + "_" + df['maze_size'] + "_" + df['maze_shape']
    
    # Create subject ID
    df['subject_id'] = df.apply(lambda x: 
        x['participant_id'] if x['participant_type'] == 'human' 
        else f"{x['model_name']}_{x['encoding_type']}", axis=1)
    
    # Extract base model names for LLMs
    df['base_model'] = df['model_name'].apply(lambda x: 
        'human' if x == 'human' else x.split('-')[0] if '-' in x else x)
    
    # Create binary encoding for key predictors
    df['is_human'] = (df['participant_type'] == 'human').astype(int)
    df['is_7x7'] = (df['maze_size'] == '7x7').astype(int)
    
    return df

def run_simplified_logistic_glmm(df, task='solve'):
    """
    Run simplified logistic GLMM that handles separation better.
    """
    logger.info(f"Running simplified logistic GLMM for {task.upper()} task")
    
    # Filter to task
    task_df = df[df['task'] == task].copy()
    
    # Check for separation
    success_by_group = task_df.groupby(['participant_type', 'encoding_type'])['success'].agg(['mean', 'count'])
    logger.info(f"Success rates for {task} task:\n{success_by_group}")
    
    # If humans have perfect/near-perfect success, analyze LLMs separately
    human_success_rate = task_df[task_df['participant_type'] == 'human']['success'].mean()
    if human_success_rate > 0.95:
        logger.info(f"Humans have {human_success_rate:.1%} success rate - analyzing LLMs separately")
        task_df = task_df[task_df['participant_type'] == 'llm'].copy()
    
    result_dict = { 'task': task, 'method': None, 'coefficients': {}, 'odds_ratios': {}, 'p_values': {}, 'model': None, 'summary_text': [] }
    
    try:
        formula_main = "success ~ C(base_model) + C(encoding_type) + C(maze_size) + C(maze_shape)"
        model = smf.mixedlm(formula_main, task_df, groups=task_df["subject_id"])
        result = model.fit(method='bfgs', maxiter=500)
        result_dict['summary_text'].append(str(result.summary()))
        
        params = result.params
        for param in params.index:
            if param != 'Intercept' and param != 'Group Var':
                coef, se = params[param], result.bse[param]
                or_val, p_val = np.exp(coef), result.pvalues[param]
                result_dict['coefficients'][param], result_dict['odds_ratios'][param], result_dict['p_values'][param] = coef, or_val, p_val

        result_dict.update({'method': 'mixedlm', 'model': result})
        
        formula_int = "success ~ C(base_model) * C(encoding_type) + C(maze_size) + C(maze_shape)"
        model2 = smf.mixedlm(formula_int, task_df, groups=task_df["subject_id"])
        result2 = model2.fit(method='bfgs', maxiter=500)
        
        lr_stat, df_diff = 2 * (result2.llf - result.llf), len(result2.params) - len(result.params)
        p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
        
        if p_value < 0.05:
            logger.info("Interaction is significant - using Model 2")
            result_dict['model'] = result2
            result_dict['summary_text'].append(str(result2.summary()))
        else:
            logger.info("Interaction not significant - using Model 1")
            
    except Exception as e:
        logger.warning(f"Mixed model failed for {task}: {e}. Falling back to fixed effects logistic regression.")
        X = pd.get_dummies(task_df[['base_model', 'encoding_type', 'maze_size', 'maze_shape']], drop_first=True)
        y = task_df['success']
        
        model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
        model.fit(X, y)
        
        for i, feature in enumerate(X.columns):
            coef = model.coef_[0][i]
            result_dict['coefficients'][feature], result_dict['odds_ratios'][feature] = coef, np.exp(coef)
        
        result_dict.update({'method': 'sklearn_logistic', 'model': model})
    
    return result_dict

def analyze_edit_distances_beta(df):
    """
    Analyze edit distances using beta regression approach.
    """
    logger.info("Analyzing edit distances using beta regression")
    gen_df = df[(df['task'] == 'generate') & (df['edit_distance'].notna()) & (df['participant_type'] == 'llm')].copy()
    
    if len(gen_df) == 0:
        logger.warning("No LLM edit distance data available for analysis.")
        return None
    
    gen_df['edit_dist_adj'] = gen_df['edit_distance'].clip(1e-6, 1-1e-6)
    gen_df['logit_edit'] = np.log(gen_df['edit_dist_adj'] / (1 - gen_df['edit_dist_adj']))
    
    formula = "logit_edit ~ C(base_model) + C(encoding_type) + C(maze_shape) + C(maze_size)"
    
    try:
        model = smf.mixedlm(formula, gen_df, groups=gen_df["maze_exemplar_id"])
        result = model.fit()
        return result
    except Exception as e:
        logger.error(f"Beta regression (mixed model on logit) failed: {e}")
        return None

def main():
    """Main analysis pipeline."""
    logger.info("Starting robust GLMM analysis")
    
    glmm_data_path = PROCESSED_DATA_DIR / "comprehensive_glmm_data.pkl"
    if not glmm_data_path.exists():
        logger.error(f"GLMM data not found at {glmm_data_path}. Run prepare_glmm_data.py first.")
        return
        
    df = pd.read_pickle(glmm_data_path)
    df = prepare_data_for_glmm(df)
    
    results = {}
    for task in ['solve', 'recognize', 'generate']:
        result = run_simplified_logistic_glmm(df, task)
        if result:
            results[f'{task}_glmm'] = result
    
    edit_result = analyze_edit_distances_beta(df)
    if edit_result:
        results['edit_distance_beta'] = edit_result
    
    # Save comprehensive summary
    with open(STATISTICAL_RESULTS_FILE, "w") as f:
        f.write("ROBUST GLMM ANALYSIS - DETAILED RESULTS\n\n")
        
        for key, result_data in results.items():
            f.write(f"\n{'='*60}\nRESULTS FOR: {key.upper()}\n{'='*60}\n")
            if isinstance(result_data, dict):
                f.write(f"Method used: {result_data.get('method', 'N/A')}\n\n")
                if result_data.get('summary_text'):
                    f.write("MODEL SUMMARY:\n")
                    f.write("\n".join(result_data['summary_text']))
            elif hasattr(result_data, 'summary'):
                f.write(str(result_data.summary()))
            f.write("\n\n")

    logger.info(f"Statistical analysis complete. Detailed summary saved to {STATISTICAL_RESULTS_FILE}")

if __name__ == "__main__":
    main() 