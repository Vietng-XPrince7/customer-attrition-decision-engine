"""
Customer Attrition Decision Engine
===================================
Dual-Stage Revenue Defense & Profit-Driven Attrition Engine.

Modules
-------
config          : Paths, constants, column registries, financial parameters
data_loader     : Data ingestion, schema validation, and merge pipeline
features        : Custom Scikit-learn transformers for domain feature engineering
models          : Stage-1 regression/classifier and Stage-2 stacking ensemble
explainability  : SHAP-based model interpretation utilities
decision_engine : 2D Risk-Value matrix, Expected Profit optimisation
"""

__version__ = "0.1.0"
__author__ = "Financial Data Science Team"

