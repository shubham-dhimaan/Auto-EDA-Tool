"""
Auto EDA - Command Line Interface
"""

import argparse
from pathlib import Path
from src.pipeline import AutoEDAPipeline

def main():
    parser = argparse.ArgumentParser(description="Run Auto EDA pipeline on a dataset.")
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        required=True,
        help="Path to the raw CSV dataset."
    )
    parser.add_argument(
        "--report",
        "-r",
        type=str,
        default=None,
        help="Optional path to pre-generated ydata-profiling JSON report."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output",
        help="Directory to save pipeline outputs (default: output)."
    )

    args = parser.parse_args()

    pipeline = AutoEDAPipeline(output_dir=args.output)
    results = pipeline.run(df_or_path=args.data, profiling_report_path=args.report)
    print("\n[PIPELINE COMPLETE] Analysis recommendations summary:")
    print(results["analysis_insights"][:500] + ("..." if len(results["analysis_insights"]) > 500 else ""))

if __name__ == "__main__":
    main()
