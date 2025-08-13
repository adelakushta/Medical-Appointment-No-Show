# Medical-Appointment-No-Show Prediction

## Overview

This repository contains a comprehensive comparison of machine learning classification algorithms including Logistic Regression, Decision Trees, and Neural Oblivious Decision Ensembles (NODE) for predicting medical appointment no-shows. The project addresses the critical healthcare challenge of missed appointments, which cause inefficiencies and increased costs in healthcare systems.

## Repository Structure

- **Decision_Trees/**
  - `Decision_Trees.ipynb`
  - `decision_tree_tuning_results.csv`

- **Logistic_Regression/**
  - `Logistic_Regression.ipynb`
  - `logistic_regression_results.csv`
  - `logistic_regression_results_with_SMOTE.csv`

- **NODE/**
  - `NODE.ipynb`

- `LICENSE`
- `README.md`


## Dataset Information 

- Source: Kaggle Medical Appointment No-Show Dataset
- Size: 110,527 records
- Target Variable: No-show (0 = Showed up, 1 = Missed appointment)
- Challenge: Imbalanced classes (~20% no-shows, ~80% show-ups)
- Note: Dataset not included due to licensing. Download from Kaggle and place it accordingly before running the code.

## Feature Engineering 

Key features engineered from the raw dataset:
- Age (cleaned and validated)
- Gender (encoded)
- DaysBetween: Days between scheduling and appointment
- ScheduledWeekday: Day of week appointment was scheduled
- IsWeekendAppointment: Boolean flag for weekend appointments
- AppointmentWeekday: Day of week appointment occurs
- Additional behavioral and temporal features for improved prediction

## Modeling Pipeline

The project follows a structured pipeline:
- Project Definition and Scope
- Data Collection
- Exploratory Data Analysis (EDA) – Before Feature Engineering
- Feature Engineering
- Exploratory Data Analysis (EDA) – After Feature Engineering
- Data Preprocessing (handling missing data, encoding, normalization, train-test split, and SMOTE oversampling)
- Logistic Regression: Open Logistic_Regression/Logistic_Regression.ipynb
- Decision Trees:   Open Decision_Trees/Decision_Trees.ipynb
- NODE:  Open NODE/NODE.ipynb
- Conclusion and Recommendations

## Key Results 

Model Performance Summary

| Algorithm | Scenario | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|----------|-----------|--------|----------|
| Logistic Regression | Without SMOTE| 79.8% | High | ~0.00 | Poor |
| Logistic Regression | With SMOTE | Lower | Decreased | Improved | Better |
| Decision Trees | Without SMOTE | 80.0% | Low | Poor | 0.02 | 
| Decision Trees | With SMOTE + Threshold 0.2 | 56.0% | 30% | 84% |Improved |
| NODE | Without SMOTE | 80.0% | 58% | 50% | 0.45 |
| NODE | With SMOTE | 70.0% | 56% | 57% | 0.57 |

## Visualizations

The repository includes comprehensive visualizations:
- Class distribution analysis before and after SMOTE
- Confusion matrices for all models and scenarios
- Feature importance plots for tree-based models
- Hyperparameter tuning results and performance comparisons
- ROC curves and precision-recall curves
- Training curves for NODE model convergence

## Technologies Used

- Python 3.x
- Scikit-learn - Traditional ML algorithms and preprocessing
- PyTorch - NODE model implementation
- Pandas & NumPy - Data manipulation and numerical computing
- SMOTE (imbalanced-learn) - Class imbalance handling
- Matplotlib/Seaborn - Comprehensive data visualization
- Jupyter Notebook - Interactive development and analysis

## Key Findings

Critical Insights
- Class imbalance significantly impacts model performance: All models initially achieved high accuracy (~80%) but failed to detect no-shows
- SMOTE preprocessing is essential: Improved recall substantially across all algorithms
- Healthcare domain prioritizes recall over precision: Better to have false alarms than missed no-shows
- Custom thresholds enhance minority class detection: Decision trees with 0.2 threshold achieved 84% recall
- NODE shows promise for tabular healthcare data: Balanced performance with proper preprocessing

## Algorithm-Specific Learnings

- Logistic Regression: Limited effectiveness on this imbalanced healthcare dataset despite extensive tuning
- Decision Trees: Excellent recall (84%) after SMOTE and threshold adjustment, making it suitable for no-show detection
- NODE: Best balanced performance with SMOTE, demonstrating potential for complex feature interactions

## Healthcare Applications

This project demonstrates practical ML applications in healthcare:
- Targeted patient reminders for high-risk no-show predictions
- Resource allocation based on predicted attendance patterns
- Appointment scheduling optimization to minimize gaps
- Cost reduction through better appointment management

## Documentation

For detailed methodology, implementation notes, and results analysis:
- Each algorithm folder contains comprehensive Jupyter notebooks
- Results are saved as Excel files with detailed performance metrics
- All hyperparameter combinations and their outcomes are documented

## Contact

Author: Adela Kushta 

Email: adelakushta05@gmail.com
