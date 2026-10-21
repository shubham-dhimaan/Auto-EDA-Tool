
"""
runner.py

Executes AI generated code functions for cleaning, analysing and visualizing data.
It also handles errors by sending broken function, with errors to an AI model for
repairs.

"""
# --- Standard library imports ---
import json
import os

# ----- Data Library imports -----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


# --- 1. Executes Coding Steps  ---


def execute_steps(json_of_functions, data, metadata):

    """
    Executes AI-generated data-functions.

    Args:
        json_of_functions (str): JSON string containing the functions to execute.
        data (pd.DataFrame): The DataFrame to clean.
        metadata (dict): Metadata describing the DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """

    copy_df = data.copy()  # Create a copy of the original DataFrame to avoid modifying it directly
    
    # Converts JSON string into dictionary
    function_dict = json.loads(json_of_functions)

    for func_data in function_dict["functions"]:
        current_function = func_data["name"]           # extracting current iteration of the function and storing in current_function (name only)
        
        try:
            function_box = {"pd": pd, "np": np}                # Add any other libraries you might need here
            exec(func_data["code"], function_box)          # extracts the code for the current function and exccutes it and stores the function object in the function_box dictionary with the name of the function as key

             # Retrieve and run the function
            ready_function = function_box[current_function]             # takes that function object and stores it in ready_function
            result = ready_function(copy_df.copy())
            
            if not isinstance(result, pd.DataFrame):
                raise TypeError(f"{current_function} returned {type(result).__name__}, expected DataFrame")

            copy_df = result                            # runs ready function on the copy of the df 

            print(f"[SUCCESS] {current_function} ran successfully.")

        except Exception as e:                                              # catches the error if the function fails to run and prints the error message 
            
            print(f"[FAILED] {current_function} — {type(e).__name__}: {e}") 
            print("Calling Code Repair function to fix the error...")
            repaired_function = code_repair(func_data, e, metadata)

            func_data = repaired_function                                   # changing current iteration with repaired function so that it can be executed 
            current_function = repaired_function["name"]
            
            try: 
                function_box = {"pd": pd, "np": np}             # Add any other libraries you might need here

                exec(repaired_function["code"], function_box)
                ready_function = function_box[current_function]
                result = ready_function(copy_df.copy())
                copy_df = result

                print(f"[SUCCESS] {current_function} ran successfully after repair.")
            
            except Exception as e2:
                print(f"[FAILED] {current_function} — {type(e2).__name__}: {e2}")
                print("Skipping this function and moving to the next one.")

        
        
    print("Cleaning process completed.")
    return copy_df                                                         # Return the cleaned DataFrame after all functions have been executed



# ------- 2. Code Repair Function -------


def code_repair(function_dict, error, metadata):
    
    prompt = f"""
    You are an expert Python code repair engineer. You specialize in debugging
    failed pandas data-cleaning functions and rewriting them so they run correctly.

    A data-cleaning function failed while being executed. Your job is to fix it.

    ## Failed function
    {json.dumps(function_dict, indent=2)}

    ## Error raised
    Type: {type(error).__name__}
    Message: {str(error)}

    ## Dataset metadata
    {metadata}

    ## How the code will be run
    - The function code is executed with exec() in a namespace that contains only
    `pd` (pandas). Any other library you need must be imported inside the function.
    - The function is then retrieved by its name and called with a single pandas
    DataFrame as its only argument.
    - It must return a pandas DataFrame. Returning None or any other type counts
    as a failure.

    ## Instructions
    1. Diagnose the root cause. Compare the error against the metadata: check
    column names, dtypes, missing values and anything the code assumes that
    the metadata contradicts.
    2. Rewrite the function so it fixes that root cause and still does what the
    original function intended. Do not add unrelated cleaning steps.
    3. Keep the function name exactly the same as in the failed function.
    4. Keep the signature to one parameter (the DataFrame).
    5. Do not modify the input DataFrame in place. Work on a copy and return it.
    6. Only reference columns that exist in the metadata. If a column the original
    code needs does not exist, handle it safely instead of raising an error.
    7. Do not print anything, read files, or access the network.

    ## Output format
    Return ONLY valid JSON, with no markdown fences and no extra text. Return a
    single function object (not a list, not wrapped in "functions") with exactly
    this structure:
    {{
        "name": "function_name",
        "columns": ["column_name"],
        "action": "short_action_name",
        "reason": "why this function is needed",
        "requires_review": true,
        "code": "complete corrected Python function as a string"
    }}
    Keep "name", "columns", "action", "reason" and "requires_review" identical to
    the failed function above. Only "code" should change.


"""
    response = client.models.generate_content(
        model = "gemini-3.5-flash-lite",
        contents = prompt,
        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens= 8192 ,
            response_mime_type="application/json")
    )
    
    repaired = json.loads(response.text)

    # Keep the original dict, take only the new code from Gemini
    return repaired

#---    3. Code runner for EDA steps that stroes every output ---- 

def execute_eda_steps(json_of_functions, data, metadata):

    """
    Executes AI-generated EDA data-functions and stroes output in a dict.

    Args:
        json_of_functions (str): JSON string containing the functions to execute.
        data (pd.DataFrame): The DataFrame to clean.
        metadata (dict): Metadata describing the DataFrame.

    Returns:
        dict: of the function ran, why it ran and the result
    """

    copy_df = data.copy()  # Create a copy of the original DataFrame to avoid modifying it directly
    
    # Converts JSON string into dictionary

    function_dict = json.loads(json_of_functions)

    output_report = []

    for func_data in function_dict["functions"]:
        current_function = func_data["name"]           # extracting current iteration of the function and storing in current_function (name only)
        
        try:
            function_box = {"pd": pd, "np": np}                # Add any other libraries you might need here
            exec(func_data["code"], function_box)          # extracts the code for the current function and exccutes it and stores the function object in the function_box dictionary with the name of the function as key

             # Retrieve and run the function
            ready_function = function_box[current_function]             # takes that function object and stores it in ready_function
            result = ready_function(copy_df.copy())

            output_report.append(
                {
                    "name": current_function,
                    "actions": func_data["action"],
                    "reason": func_data["reason"],
                    "output": result

                }
            )


            print(f"[SUCCESS] {current_function} ran successfully and output is stored.")

        except Exception as e:                                              # catches the error if the function fails to run and prints the error message 
            
            print(f"[FAILED] {current_function} — {type(e).__name__}: {e}") 
            print("Calling Code Repair function to fix the error...")
            repaired_function = code_repair(func_data, e, metadata)

            func_data = repaired_function                                   # changing current iteration with repaired function so that it can be executed 
            current_function = repaired_function["name"]
            
            try: 
                function_box = {"pd": pd, "np": np}             # Add any other libraries you might need here

                exec(repaired_function["code"], function_box)
                ready_function = function_box[current_function]
                result = ready_function(copy_df.copy())
                
                output_report.append(
                                {
                                    "name": current_function,
                                    "actions": func_data["action"],
                                    "reason": func_data["reason"],
                                    "output": result
                
                                }
                            )
                

                print(f"[SUCCESS] {current_function} ran successfully after repair.")
            
            except Exception as e2:
                print(f"[FAILED] {current_function} — {type(e2).__name__}: {e2}")
                print("Skipping this function and moving to the next one.")

        
        
    print("Analysis process completed.")
    return output_report        