"""
pipeline.py
-----------
Deterministic, pipeline-based orchestrator for Auto EDA.
Coordinates data inspection, profiling/compression, cleaning plan generation,
code generation, resilient execution with self-repair, and downstream analysis.
"""

import json
from pathlib import Path
import pandas as pd

from src.cleaner import get_cleaning_recommendations, generate_cleaning_functions, inspect_dataset
from src.runner import execute_steps
from src.analyser import get_analysis_recommendations_basic, generate_analysis_functions
from src.evaluator import evaluate_cleaning
from src.report_compressor import compress_report


class AutoEDAPipeline:
    """
    Pipeline orchestrator that executes end-to-end data cleaning and analysis
    without relying on autonomous, unbounded agent loops.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, df_or_path, profiling_report_path: str = None) -> dict:
        """
        Runs the full Auto EDA pipeline.

        Args:
            df_or_path: Path to a CSV file or an existing pandas DataFrame.
            profiling_report_path (optional): Path to a pre-generated ydata-profiling JSON report.

        Returns:
            dict containing:
                - "original_df": Original dataframe
                - "cleaned_df": Cleaned dataframe
                - "cleaning_insights": Generated recommendations text
                - "cleaning_functions_json": Generated cleaning code json
                - "evaluation": Evaluation results of cleaning
                - "analysis_insights": Recommended analytical steps
        """
        # 1. Load Data
        if isinstance(df_or_path, (str, Path)):
            print(f"[PIPELINE] Loading dataset from: {df_or_path}")
            df_original = pd.read_csv(df_or_path)
        elif isinstance(df_or_path, pd.DataFrame):
            df_original = df_or_path.copy()
        else:
            raise TypeError("df_or_path must be a file path string or pandas DataFrame")

        print(f"[PIPELINE] Input shape: {df_original.shape}")

        # 2. Inspect Dataset & Metadata
        inspection_df, mapping_df = inspect_dataset(df_original)
        df_metadata = {
            "shape": df_original.shape,
            "columns": list(df_original.columns),
            "dtypes": {col: str(dtype) for col, dtype in df_original.dtypes.items()}
        }

        # 3. Handle Profiling Report (or generate minimal summary if not supplied)
        if profiling_report_path and Path(profiling_report_path).exists():
            report_file_to_use = profiling_report_path
        else:
            # Create a lightweight fallback profiling JSON report for cleaner.py
            fallback_report = {
                "variables": {col: {"type": str(dtype)} for col, dtype in df_original.dtypes.items()}
            }
            temp_report_path = self.output_dir / "temp_profiling_report.json"
            with open(temp_report_path, "w", encoding="utf-8") as f:
                json.dump(fallback_report, f)
            report_file_to_use = str(temp_report_path)

        # 4. Generate Cleaning Recommendations
        print("[PIPELINE] Step 1/4: Generating data cleaning recommendations...")
        cleaning_insights = get_cleaning_recommendations(report_file_to_use, df_original)

        # 5. Generate Cleaning Functions (JSON)
        print("[PIPELINE] Step 2/4: Generating Python cleaning functions...")
        cleaning_functions_json = generate_cleaning_functions(cleaning_insights, df_metadata)

        # 6. Execute Cleaning Functions with Self-Repair
        print("[PIPELINE] Step 3/4: Executing cleaning functions with runner & error repair...")
        cleaned_df = execute_steps(cleaning_functions_json, df_original, df_metadata)

        # Save cleaned dataset
        cleaned_path = self.output_dir / "cleaned_data.csv"
        cleaned_df.to_csv(cleaned_path, index=False)
        print(f"[PIPELINE] Cleaned dataset saved to: {cleaned_path}")

        # 7. Evaluate Cleaning
        print("[PIPELINE] Evaluating cleaning results against original issues...")
        eval_results = None
        try:
            eval_results = evaluate_cleaning(cleaning_insights, df_original, cleaned_df)
        except Exception as e:
            print(f"[PIPELINE] Evaluation warning: {e}")

        # 8. Analysis & Insights on Cleaned Data
        print("[PIPELINE] Step 4/4: Generating exploratory data analysis recommendations...")
        analysis_insights = get_analysis_recommendations_basic(cleaned_df)

        return {
            "original_df": df_original,
            "cleaned_df": cleaned_df,
            "cleaning_insights": cleaning_insights,
            "cleaning_functions_json": cleaning_functions_json,
            "evaluation": eval_results,
            "analysis_insights": analysis_insights
        }
