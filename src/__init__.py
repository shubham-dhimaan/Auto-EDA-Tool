"""
Auto EDA Package
----------------
A pipeline-based Automated Exploratory Data Analysis & Data Cleaning engine.
"""

from src.cleaner import get_cleaning_recommendations, generate_cleaning_functions, inspect_dataset
from src.runner import execute_steps, code_repair
from src.analyser import get_analysis_recommendations_basic, generate_analysis_functions
from src.report_compressor import compress_report
from src.evaluator import evaluate_cleaning
from src.pipeline import AutoEDAPipeline

__all__ = [
    "AutoEDAPipeline",
    "get_cleaning_recommendations",
    "generate_cleaning_functions",
    "inspect_dataset",
    "execute_steps",
    "code_repair",
    "get_analysis_recommendations_basic",
    "generate_analysis_functions",
    "compress_report",
    "evaluate_cleaning",
]
