# Maze Understanding Analysis: Humans vs. Language Models

*Report generated: 2025-07-29 12:12:06*

## Executive Summary

This analysis examines procedural versus conceptual knowledge in maze understanding tasks, 
comparing human performance with various language models (LLMs) across three tasks: 
maze solving (procedural), shape recognition (conceptual), and maze generation (conceptual). 
Edit distance analysis quantifies how closely generated mazes match canonical exemplars.

### Data Processing Notes

- **Duplicate Trial Handling**: Some human participants completed the same maze multiple times. 
For analysis, we selected the best attempt for each participant-maze combination based on 
overall task performance (solving, recognition, generation success) and edit distance for generation tasks.

- **Expected Data**: Each participant should complete 12 unique mazes (2 sizes × 6 shapes).

## Key Findings

### 1. Overall Task Performance

| Participant         | Solve Rate   | Recognition Rate   | Generation Rate   |
|:--------------------|:-------------|:-------------------|:------------------|
| Human               | 99.4%        | 99.7%              | 99.4%             |
| Claude (coord_list) | 75.9%        | 7.6%               | 0.0%              |
| Claude (matrix)     | 38.3%        | 4.7%               | 13.0%             |
| Claude (vision)     | 56.5%        | 0.9%               | 0.0%              |
| Claude (coord_list) | 75.9%        | 7.6%               | 0.0%              |
| Claude (matrix)     | 38.3%        | 4.7%               | 13.0%             |
| Claude (vision)     | 91.7%        | 1.5%               | 0.0%              |
| GPT-4 (coord_list)  | 84.0%        | 19.0%              | 2.7%              |
| GPT-4 (matrix)      | 72.7%        | 13.2%              | 8.5%              |
| Llama (coord_list)  | 53.9%        | 8.7%               | 0.2%              |
| Llama (matrix)      | 20.2%        | 3.9%               | 2.6%              |


### 2. Procedural-Conceptual Knowledge Dissociation

| Participant         | Procedural (Solve)   | Conceptual (Avg)   | Dissociation Score   |
|:--------------------|:---------------------|:-------------------|:---------------------|
| Human               | 99.4%                | 99.5%              | -0.2%                |
| Claude (coord_list) | 75.9%                | 3.8%               | +72.1%               |
| Claude (matrix)     | 38.3%                | 8.8%               | +29.5%               |
| Claude (vision)     | 56.5%                | 0.5%               | +56.0%               |
| Claude (coord_list) | 75.9%                | 3.8%               | +72.1%               |
| Claude (matrix)     | 38.3%                | 8.8%               | +29.5%               |
| Claude (vision)     | 91.7%                | 0.8%               | +91.0%               |
| GPT-4 (coord_list)  | 84.0%                | 10.9%              | +73.1%               |
| GPT-4 (matrix)      | 72.7%                | 10.9%              | +61.9%               |
| Llama (coord_list)  | 53.9%                | 4.4%               | +49.4%               |
| Llama (matrix)      | 20.2%                | 3.2%               | +16.9%               |

*Note: Positive dissociation scores indicate better procedural than conceptual performance.*

### 3. Effect of Presentation Format

|                                    | coord_list   | matrix   | vision   |
|:-----------------------------------|:-------------|:---------|:---------|
| ('claude-3.5-sonnet', 'Generate')  | 0.0%         | 13.0%    | 0.0%     |
| ('claude-3.5-sonnet', 'Recognize') | 7.6%         | 4.7%     | 0.9%     |
| ('claude-3.5-sonnet', 'Solve')     | 75.9%        | 38.1%    | 56.5%    |
| ('gpt-4o', 'Generate')             | 2.7%         | 8.5%     | nan      |
| ('gpt-4o', 'Recognize')            | 19.0%        | 13.2%    | nan      |
| ('gpt-4o', 'Solve')                | 84.0%        | 72.7%    | nan      |
| ('llama-3.1', 'Generate')          | 0.2%         | 2.6%     | nan      |
| ('llama-3.1', 'Recognize')         | 8.7%         | 3.9%     | nan      |
| ('llama-3.1', 'Solve')             | 53.9%        | 20.2%    | nan      |

### 4. Effect of Maze Shape

Average performance by maze shape:

|                       |   solve_rate |   recognize_rate |   generate_rate |
|:----------------------|-------------:|-----------------:|----------------:|
| ('human', 'C')        |        0.98  |        1         |       0.99      |
| ('human', 'Z')        |        1     |        1         |       1         |
| ('human', 'cross')    |        1     |        1         |       1         |
| ('human', 'spiral')   |        1     |        1         |       1         |
| ('human', 'square')   |        1     |        0.99      |       1         |
| ('human', 'triangle') |        0.98  |        1         |       0.98      |
| ('llm', 'C')          |        0.768 |        0.0203333 |       0.053     |
| ('llm', 'Z')          |        0.622 |        0.349     |       0.0453333 |
| ('llm', 'cross')      |        0.473 |        0.0206667 |       0.024     |
| ('llm', 'spiral')     |        0.43  |        0.0393333 |       0.049234  |
| ('llm', 'square')     |        0.718 |        0.001     |       0.037     |
| ('llm', 'triangle')   |        0.631 |        0         |       0.032     |

### 5. Edit Distance Analysis (Generation Task)

Edit distance statistics by model and encoding:

|                     |   mean |   std |   min |   max |
|:--------------------|-------:|------:|------:|------:|
| ('human', 'visual') |  0.057 | 0.093 |     0 |   0.4 |

*Note: Lower edit distance indicates generated mazes closer to valid exemplars.*

## Statistical Analysis Summary

```
ROBUST GLMM ANALYSIS - DETAILED RESULTS


============================================================
RESULTS FOR: SOLVE_GLMM
============================================================
Method used: sklearn_logistic

MODEL SUMMARY:
                Mixed Linear Model Regression Results
======================================================================
Model:                  MixedLM     Dependent Variable:     success   
No. Observations:       17982       Method:                 REML      
No. Groups:             10          Scale:                  0.1754    
Min. group size:        1791        Log-Likelihood:         -9915.4793
Max. group size:        1800        Converged:              No        
Mean group size:        1798.2                                        
----------------------------------------------------------------------
                           Coef.  Std.Err.    z    P>|z| [0.025 0.975]
----------------------------------------------------------------------
Intercept                   0.900    0.095   9.474 0.000  0.714  1.087
C(base_model)[T.gpt]        0.213    0.134   1.590 0.112 -0.050  0.476
C(base_model)[T.llama]     -0.200    0.134  -1.495 0.135 -0.463  0.062
C(encoding_type)[T.matrix] -0.301    0.109  -2.750 0.006 -0.515 -0.086
C(encoding_type)[T.vision]  0.020    0.145   0.139 0.889 -0.263  0.304
C(maze_size)[T.7x7]        -0.036    0.006  -5.836 0.000 -0.049 -0.024
C(maze_shape)[T.Z]         -0.146    0.011 -13.533 0.000 -0.168 -0.125
C(maze_shape)[T.cross]     -0.295    0.011 -27.282 0.000 -0.316 -0.274
C(maze_shape)[T.spiral]    -0.337    0.011 -31.149 0.000 -0.359 -0.316
C(maze_shape)[T.square]    -0.051    0.011  -4.686 0.000 -0.072 -0.029
C(maze_shape)[T.triangle]  -0.137    0.011 -12.701 0.000 -0.159 -0.116
Group Var                   0.024    0.055                            
======================================================================



============================================================
RESULTS FOR: RE

... (truncated)
```

## Appendix: Data Summary

### Sample Sizes

Total experimental conditions analyzed: 420


Conditions per participant type:

| group_label   |   0 |
|:--------------|----:|
| Claude        |   6 |
| GPT-4         |   2 |
| Human         |   1 |
| Llama         |   2 |