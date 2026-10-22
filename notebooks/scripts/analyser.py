
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
    Generate Python functions for EDA based on analysis insights
    and DataFrame metadata.

    Args:
        analysis_insights (str): Summary of potential analyses and findings.
        df (dataframe): Metadata describing the DataFrame.

    Returns:
        str: JSON string containing generated functions and their metadata,
             following the existing output structure.
    """

    prompt = f"""
    You are an expert Python data analyst specializing in exploratory
    data analysis (EDA). You translate analysis insights, findings, and
    recommended analysis steps into small, reliable Python functions
    that carry out the exploration.

    INPUT 1: ANALYSIS INSIGHTS / RECOMMENDED STEPS
    {analysis_insights}

    INPUT 2: SAMPLE of DATA FRAME
    {df.head(30)}

    YOUR TASK:
    Read the insights or recommended steps and generate small, independent
    Python functions that carry out the exploratory analysis they call
    for. Base every function on what the insights ask for and on what the
    DataFrame sample actually contains. Cover whatever EDA is relevant,
    such as descriptive statistics, distributions, group-by summaries,
    comparisons, correlations, trends, and checks of data quality or
    structure, whether the insights request them explicitly or imply them.

    RULES:
    1. Generate functions only for relevant, actionable analysis steps.
    2. Every function must be exploratory. Do not generate data-cleaning
       functions, model-training strategies, or feature-selection
       procedures.
    3. Use the exact DataFrame column names provided.
    4. Each function must accept df as its only argument.
    5. Each function must return its analysis result.
    6. Do not modify the original DataFrame. Work on copies or derived
       objects.
    7. Do not invent columns, assume unsupported facts, or fabricate
       findings. If a step cannot be supported by the available columns,
       skip it.
    8. Avoid redundant functions that perform essentially the same
       analysis.
    9. Keep functions small, focused, and readable.
    10. Use appropriate Pandas operations and handle potential errors
        where reasonably necessary.
    11. Do not include imports or Markdown code fences.
    12. Set "requires_review" to true when a function depends on an
        assumption about the data, such as an inferred column meaning or
        an ambiguous step in the insights. Otherwise set it to false.

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


def format_eda_results(output, max_chars_per_step=3000):
    """Turn the list of executed step results into compact text for the prompt."""
    blocks = []
    for i, step in enumerate(output, 1):
        result = str(step.get("output", ""))
        if len(result) > max_chars_per_step:
            result = result[:max_chars_per_step] + "\n...[truncated]"
        blocks.append(
            f"--- Step {i}: {step.get('name')} ---\n"
            f"Actions: {step.get('actions')}\n"
            f"Reason: {step.get('reason')}\n"
            f"Result:\n{result}"
        )
    return "\n\n".join(blocks)


def get_analysis_recommendations_advanced(output, df, basic_recommendations):
    """
    Generate advanced EDA recommendations grounded in the executed basic EDA.

    Args:
        output: list of dicts, one per executed basic EDA step, each with
                "name", "actions", "reason", and "output" (the actual result).
        df: cleaned pandas DataFrame. Only shape and head are used for context.
        basic_recommendations: text of the basic EDA plan that was executed.

    Returns:
        Text of follow-up recommendations for the code generator.
    """
    results_text = format_eda_results(output)

    prompt = f"""
You are an expert data analyst planning the ADVANCED stage of exploratory
data analysis (EDA) on a cleaned pandas DataFrame.

A basic EDA has already been planned and executed. Your job is to read the
actual results of each executed step, understand what they reveal, and
recommend further analyses that go deeper. Your recommendations will be
passed to a separate code generator that turns them into Python/Pandas code.

## Dataset context
Shape (rows, columns): {df.shape}
Sample rows (head):
{df.head(10).to_string()}

## Basic EDA plan that was followed
{basic_recommendations}

## Executed basic EDA steps and their actual results
{results_text}

## What to do

1. Review each executed step against its stated reason. Note what the result
   actually shows: key numbers, distributions, skew, outliers, missing or
   dominant categories, strong or weak relationships, and anything surprising.
   Also note any step whose output was empty, errored, or uninformative.
2. Identify gaps: what the basic EDA did not cover, what it hinted at but did
   not explain, and which findings need verification or a closer look.
3. Recommend advanced analyses that build directly on those findings. Examples
   of the kind of depth expected (use only what fits the data):
   - Investigating outliers or skew flagged in the results (IQR/z-score
     segmentation, log transforms, comparing groups with and without outliers).
   - Segment-level or multi-column group comparisons that explain patterns
     seen in the basic aggregates.
   - Deeper relationship analysis where correlations were notable (partial
     views by category, non-linear checks, Spearman vs Pearson).
   - Categorical associations, rare-category handling, cross-tabulations.
   - Time-based analysis (trends, seasonality, rolling windows) if date
     columns exist.
   - Data quality follow-ups (duplicates, inconsistent values, suspicious
     ranges) that the results suggest.

## Rules
- Use only columns that exist in the dataset. Use exact column names.
- Base every recommendation on something observed in the results above.
  Cite the finding it follows from.
- Do NOT repeat analyses that were already executed.
- Do NOT claim new findings. You have only seen the basic results.
- Do NOT write Python code or modify the DataFrame.
- Exclude identifier columns from numerical analysis.
- Prefer a small number of high-value recommendations (roughly 5 to 8) over
  a long generic list.

## Output format
First, a short section titled "Key observations" with one to two lines per
executed step.

Then a section titled "Recommended advanced analyses". For each one give:
1. Objective: what question it answers.
2. Based on: the specific finding from the basic results that motivates it.
3. Columns and dtypes: exact column names and pandas data types.
4. Operations: specific pandas/NumPy/SciPy functions, with parameters or
   aggregation methods.
5. Expected output: the form of the result (table, series, plot data) and
   what it could reveal.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )

    return response.text



def final_insights_summariser(
    basic_eda_recommendations,
    adv_eda_recommendations,
    basic_eda_output,
    adv_eda_output,
):
    """
    Summarise basic and advanced EDA results into 10-15 final insights.

    Returns the insights as plain text (a numbered list).
    """
    prompt = f"""
    You are an expert data analyst who turns raw exploratory data analysis
    (EDA) results into clear, accurate, decision-ready insights.

    INPUT 1: BASIC EDA RECOMMENDATIONS
    {basic_eda_recommendations}

    INPUT 2: ADVANCED EDA RECOMMENDATIONS
    {adv_eda_recommendations}

    INPUT 3: BASIC EDA OUTPUT
    {format_eda_results(basic_eda_output)}

    INPUT 4: ADVANCED EDA OUTPUT
    {format_eda_results(adv_eda_output)}

    YOUR TASK:
    Read the recommendations to understand what each analysis was meant to
    answer, then read the outputs to see what the data actually showed.
    Produce 10 to 15 final insights that summarise what was learned from
    the data.

    RULES:
    1. Base every insight only on the EDA outputs. Do not invent numbers,
       columns, or findings, and do not use outside assumptions.
    2. Cite specific evidence for each insight (values, percentages,
       counts, correlations, group differences) taken from the outputs.
    3. Prioritise insights by importance: strong patterns, large
       differences, notable outliers, meaningful relationships, and data
       quality issues that affect interpretation.
    4. Combine findings from basic and advanced outputs where they support
       each other. Do not repeat the same point in different words.
    5. Distinguish correlation from causation. Do not claim causes the
       analysis did not establish.
    6. If an output is missing, empty, or shows an error, do not build an
       insight on it.
    7. Write each insight in plain language that a non-technical reader
       can follow. One idea per insight.
    8. Do not recommend data cleaning, model training, or feature
       selection. Report what the data shows.
    9. If fewer than 10 well-supported insights exist, return only those.
       Never pad the list.

    RETURN FORMAT:
    Return plain text only, with no JSON and no Markdown code fences.
    Use a numbered list. Each item is one insight written as:

    1. Short headline: one or two sentences explaining the finding.
       Evidence: the specific numbers from the outputs that support it.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )

    return response.text.strip()