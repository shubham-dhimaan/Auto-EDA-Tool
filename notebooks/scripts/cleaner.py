
"""
cleaner.py

Utilities for analyzing data quality reports and generating
Pandas data-cleaning functions with an AI model.
"""
# --- Standard library imports ---
import json
import pandas as pd
import os

# --- Third-party imports ---
from dotenv import load_dotenv
from google import genai
from google.genai import types

from scripts import skills_loader as sl


# --- Configuration ---

# Load environment variables from the .env file.
load_dotenv()

# Initialize the Gemini client for use by this module's functions.
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# --- 1. Cleaning recommendations ---


def get_cleaning_recommendations(file_path, df):


    """
    Analyze a ydata-profiling JSON report and recommend
    data-cleaning actions.

    Args:
        file_path (str): Path to the JSON profiling report.
        df (pd.DataFrame): The DataFrame to analyze.
        client: AI client used to generate recommendations.

    Returns:
        str: A summary of recommended data-cleaning actions.
    """
    # Load the JSON report
    with open(file_path, "r", encoding="utf-8") as f:
     mod_report = json.load(f)

    report, mapping_for_categorical = inspect_dataset(df)

    prompt = f"""
    Analyze the ydata-profiling report, the inspection report, and mapping report (for categorical columns) 
    and briefly identify what data cleaning is needed.

    Focus on:
    - Missing values
    - Duplicate rows
    - Incorrect data types
    - Outliers or suspicious values
    - Inconsistent categorical values
    - Other data quality issues

    For each issue, mention the relevant column and a brief
    recommended action.

    Only recommend actions supported by the reports.
    If the mapping report indicates a categorical column has inconsistent values, suggest standardizing them baesed on the unique value counts.
    Give detailed instructions for for categorigcal columns with inconsistent values, including how to standardize based on what errors are present in the mapping report. If the mapping report indicates a categorical column has consistent values, do not suggest any cleaning for that column.
    You can ignore normalizing any numerical columns in the mapping report, instead they shoukd be converted into numeric format.

    Do not write Python code or perform the cleaning.
    If an issue cannot be determined from the report, say so.
    You goal is to give detailed enough steps so that python functions can be generated to perform the cleaning.

    Return a concise, readable summary.

    Profiling report:
    {json.dumps(mod_report)}

    Inspection report:
    {report.to_dict(orient="records")}

    Mapping report:
    {mapping_for_categorical.to_dict(orient="records")}

    """

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text


# --- 2. Generate cleaning functions ---


def generate_cleaning_functions(cleaning_insights, df_metadata):

    """
    Generate Python cleaning functions from an analysis
    and DataFrame metadata and df insepction report

    Args:
        cleaning_insights (str): Summary of cleaning issues.
        df_metadata (dict): Metadata describing the DataFrame.

    Returns:
        str: JSON-formatted response containing function
        names, affected columns, reasons, and source code.
    """

    
    skills_loaded = sl.load_skill_text(cleaning_insights)

    prompt = f"""
    You are an expert Python anlayst who specializes in data-cleaning function generation.

    INPUT 1: Cleaning_insights
    {cleaning_insights}

    INPUT 2: DATAFRAME METADATA
    {json.dumps(df_metadata, default=str)}

    INPUT 3: Skills at your disposal. FOLLOW THESE SKILLS (reference recipes, copy their style and rules):
    {skills_loaded if skills_loaded else "(no matching skill, use plain Pandas best practice)"}

    YOUR TASK:
    Generate small Python functions for basic data-cleaning
    actions supported by the analysis and metadata.

    RULES:
    1. Generate functions only for actionable cleaning issues.
    2. Do not generate functions for correlation analysis,
       feature selection, or model-training strategies.
    3. Use the exact DataFrame column names provided.
    4. Each function must accept df as its only argument.
    5. Each function must return a cleaned DataFrame.
    6. Do not modify the original DataFrame.
    7. Do not invent columns or assume unsupported facts.
    8. Flag ambiguous actions for human review.
    9. Do not automatically delete rows or cap outliers
       without a justified rule.
    10. Generate basic operations such as handling missing
        values, correcting confirmed data types, and replacing
        confirmed invalid values.
    11. Do not include imports or Markdown code fences.
    12. Use the skills provided, but do not invent new skills or recipes.

    RETURN FORMAT:
    Return valid JSON with this structure:
    {{
        "functions": [
            {{
                "name": "function_name",
                "columns": ["column_name"],
                "action": "short_action_name",
                "reason": "why this function is needed",
                "requires_review": true,
                "code": "complete Python function as a string"
            }}
        ]
    }}

    Return an empty functions list if no safe, actionable
    cleaning functions can be generated.
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=8192,
            response_mime_type="application/json"
        )
    )

    return response.text

def inspect_dataset(df):
    report = []
    mapping = []

    for column in df.columns:
        series = df[column]

        report.append({
            "column": column,
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(
                series.isna().mean() * 100, 2
            ),
            "unique_values": int(series.nunique()),
            "duplicate_count": int(series.duplicated().sum()),
            "sample_values": series.dropna().head(5).tolist()
        })

    for i in report:
        if i["unique_values"] < 50:
            mapping.append({
                "column": i["column"],
                "unique_values": df[i["column"]].value_counts().to_dict()
            })

    return pd.DataFrame(report), pd.DataFrame(mapping)