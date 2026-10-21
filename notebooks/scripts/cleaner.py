
"""
cleaner.py

Utilities for analyzing data quality reports and generating
Pandas data-cleaning functions with an AI model.
"""
# --- Standard library imports ---
import json
import os

# --- Third-party imports ---
from dotenv import load_dotenv
from google import genai
from google.genai import types


# --- Configuration ---

# Load environment variables from the .env file.
load_dotenv()

# Initialize the Gemini client for use by this module's functions.
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# --- 1. Cleaning recommendations ---


def get_cleaning_recommendations(file_path):


    """
    Analyze a ydata-profiling JSON report and recommend
    data-cleaning actions.

    Args:
        file_path (str): Path to the JSON profiling report.
        client: AI client used to generate recommendations.

    Returns:
        str: A summary of recommended data-cleaning actions.
    """
    # Load the JSON report
    with open(file_path, "r", encoding="utf-8") as f:
     mod_report = json.load(f)

    prompt = f"""
    Analyze this ydata-profiling report and briefly identify
    what data cleaning is needed.

    Focus on:
    - Missing values
    - Duplicate rows
    - Incorrect data types
    - Outliers or suspicious values
    - Inconsistent categorical values
    - Other data quality issues

    For each issue, mention the relevant column and a brief
    recommended action.

    Only recommend actions supported by the report.
    Do not write Python code or perform the cleaning.
    If an issue cannot be determined from the report, say so.

    Return a concise, readable summary.

    Profiling report:
    {json.dumps(mod_report)}
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text


# --- 2. Generate cleaning functions ---


def generate_cleaning_functions(cleaning_analysis, df_metadata):

    """
    Generate Python cleaning functions from an analysis
    and DataFrame metadata.

    Args:
        cleaning_analysis (str): Summary of cleaning issues.
        df_metadata (dict): Metadata describing the DataFrame.

    Returns:
        str: JSON-formatted response containing function
        names, affected columns, reasons, and source code.
    """

    prompt = f"""
    You are a Python data-cleaning function generator.

    INPUT 1: CLEANING ANALYSIS
    {cleaning_analysis}

    INPUT 2: DATAFRAME METADATA
    {json.dumps(df_metadata, default=str)}

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