
import pandas as pd
import os
from google import genai
from google.genai import types

from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key= os.environ["GEMINI_API_KEY"]
)


def get_analysis_recommendations_basic(df):
    
    """
    Generate recommendations for basic exploratory data analysis (EDA)
    based on the columns and data types of a cleaned DataFrame.
    
    Args:

    """
    prompt = f"""
    
    You are a expert data analyst planning basic exploratory data analysis (EDA)
    on a cleaned pandas DataFrame.

    Analyze the available columns, their data types, and their analytical
    value. Recommend relevant analyses to understand the dataset and identify
    interesting patterns.

    Focus on:
    - Descriptive statistics: mean, median, min, max, standard deviation,
      and quartiles.
    - Distributions, value counts, and proportions.
    - Group-wise comparisons and aggregations.
    - Correlations between relevant numerical variables.
    - Date-based trends where applicable.
    - Unusual patterns or extreme values worth investigating.

    For each recommendation, specify:
    1. Analysis objective.
    2. Exact column names and pandas data types.
    3. Operations and specific pandas/NumPy functions.
    4. Relevant parameters or aggregation methods.
    5. Expected output and what it could reveal.

    Use only existing columns. Consider their analytical roles, not just
    their data types. Exclude identifiers from meaningless numerical
    analysis. Avoid redundant or irrelevant operations.

    Do not generate Python code, modify the DataFrame, or claim findings
    before analyses are executed. Provide clear, specific steps that a
    separate code generator can translate into executable Python/Pandas code.

    DataFrame shape: {df.shape}
    Column data types:
    {df.dtypes.to_string()}
    Sample Data Rows: {df.head(30)}
    
    """

    response = client.models.generate_content(
        model = "gemini-3.5-flash-lite",
        contents = prompt
    )

    return response.text


def generate_analysis_functions(analysis_insights, df):
    """
    Generate Python functions for basic EDA based on analysis insights
    and DataFrame metadata.

    Args:
        analysis_insights (str): Summary of potential analyses and findings.
        df_metadata (dict): Metadata describing the DataFrame.

    Returns:
        str: JSON string containing generated functions and their metadata,
             following the existing output structure.
    """

    prompt = f"""
    You are an expert Python data analyst specializing in exploratory
    data analysis (EDA).

    INPUT 1: ANALYSIS INSIGHTS
    {analysis_insights}

    INPUT 2: SAMPLE of DATA FRAME
    {df.head(30)}

    YOUR TASK:
    Generate small, independent Python functions to perform basic EDA
    supported by the provided analysis insights and DataFrame metadata.

    RULES:
    1. Generate functions only for relevant, actionable analysis steps.
    2. Focus on basic EDA, such as descriptive statistics, value
       distributions, group-by summaries, trends, and comparisons.
    3. Use the exact DataFrame column names provided.
    4. Each function must accept df as its only argument.
    5. Each function must return its analysis result.
    6. Do not modify the original DataFrame.
    7. Do not invent columns, assume unsupported facts, or fabricate findings.
    8. Avoid redundant functions that perform essentially the same analysis.
    9. Keep functions small, focused, and readable.
    10. Use appropriate Pandas operations and handle potential errors
        where reasonably necessary.
    11. Do not generate data-cleaning functions, model-training strategies,
        or feature-selection procedures.
    12. Do not include imports or Markdown code fences.

    RETURN FORMAT:
    Return valid JSON with this exact structure:
    {{
        "functions": [
            {{
                "name": "function_name",
                "columns": ["column_name"],
                "action": "short_action_name",
                "reason": "why this analysis is useful",
                "requires_review": false,
                "code": "complete Python function as a string"
            }}
        ]
    }}

    Return an empty functions list if no relevant analysis functions
    can be generated.
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


def get_analysis_recommendations_advanced(output, df):
    
    """
    Generate recommendations for basic exploratory data analysis (EDA)
    based on the columns and data types of a cleaned DataFrame.
    
    Args:

    """
    prompt = f"""
    
    You are a expert data analyst planning basic exploratory data analysis (EDA)
    on a cleaned pandas DataFrame.

    Analyze the available columns, their data types, and their analytical
    value. Recommend relevant analyses to understand the dataset and identify
    interesting patterns.

    Focus on:
    - Descriptive statistics: mean, median, min, max, standard deviation,
      and quartiles.
    - Distributions, value counts, and proportions.
    - Group-wise comparisons and aggregations.
    - Correlations between relevant numerical variables.
    - Date-based trends where applicable.
    - Unusual patterns or extreme values worth investigating.

    For each recommendation, specify:
    1. Analysis objective.
    2. Exact column names and pandas data types.
    3. Operations and specific pandas/NumPy functions.
    4. Relevant parameters or aggregation methods.
    5. Expected output and what it could reveal.

    Use only existing columns. Consider their analytical roles, not just
    their data types. Exclude identifiers from meaningless numerical
    analysis. Avoid redundant or irrelevant operations.

    Do not generate Python code, modify the DataFrame, or claim findings
    before analyses are executed. Provide clear, specific steps that a
    separate code generator can translate into executable Python/Pandas code.

    Output of basic EDA: {output}
    Column data types:
    {df.dtypes.to_string()}
    Sample Data Rows: {df.head(30)}
    
    """

    response = client.models.generate_content(
        model = "gemini-3.5-flash-lite",
        contents = prompt
    )

    return response.text