import json
import os

from google import genai
import copy

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"])


def compress_report(report):

    """
    Compresses the given report by removing unnecessary information. The goal 
    is to reduce token usage and only pass relevant information to the AI model 
    for any insights or analysis.

    Args:
        report (str): The report content to be compressed. As the output of a 
        Y-data profile report

    """
    og_report = json.loads(report)
    mod_report = copy.deepcopy(og_report)

    # deletes unnecessary keys within the variable key

    for details in mod_report["variables"].values():
        
        details.pop("value_counts_without_nan", None)
        details.pop("value_counts_index_sorted", None)
        details.pop("histogram", None)

    # deletes unnecessary keys within the report

    mod_report.pop("scatter", None)
    mod_report.pop("correlations", None)
    mod_report.pop("missing", None)
    mod_report.pop("ydata_profiling_version", None)
    mod_report.pop("ydata_profiling_config", None)
    mod_report.pop("analysis", None)

    output_report = json.dumps(mod_report, indent=4)

    token_difference(report, output_report)  # prints the token difference between the original and compressed report

    return output_report   # returns the compressed report as a JSON string


def save_compressed_report(compressed_report):

    """
    Saves the compressed report to a specified file path.

    Args:
        compressed_report (str): The compressed report content as a JSON string.
    """
    output_path = "C:\\Users\\shubh\\Desktop\\Auto EDA\\JSON reports\\compressed_report.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(compressed_report)    


def get_token_count(report):

    """
    Calculates the number of tokens in the given report.

    Args:
        report (str): The report content as a JSON string.

    Returns:
        int: The number of tokens in the report.
    """
    report_json = json.dumps(json.loads(report))

    response = client.models.count_tokens(
        model="gemini-3.5-flash-lite",
        contents=report_json
    )

    return response.total_tokens

def token_difference(original_report, compressed_report):

    """
    Compares the token count of the original and 
    compressed reports and prints the difference.

    Args:
        original_report (str): The original report content as a JSON string.
        compressed_report (str): The compressed report content as a JSON string.        
    
    """

    og_tokens = get_token_count(original_report)
    mod_tokens = get_token_count(compressed_report)

    print(f"Original report token count: {og_tokens:,}")
    print(f"Compressed report token count: {mod_tokens:,}")

    print(f"Tokens Saved: {og_tokens - mod_tokens:,}")
    