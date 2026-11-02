"""
evaluator.py  -  did the cleaning actually fix the problems?

Flow (one Gemini call, plus one optional retry if the generated code crashes):

    old insights + old_df + cleaned_df
            |
            v
    build_prompt()        -> tells Gemini what to check
            |
            v
    ask_gemini()          -> Gemini returns Python check code
            |
            v
    run_check_code()      -> runs that code, collects results
            |
            v
    print_report()        -> shows FIXED / NOT FIXED per issue

Each check the model writes must append a dict to a list called `results`:
    {"issue": "short name", "passed": True/False, "detail": "numbers or examples"}
"""

import io
import os
import re
import contextlib
import traceback

import numpy as np
import pandas as pd

MODEL_NAME = "gemini-2.5-flash"   # change if you use another model


# ---------------------------------------------------------------------------
# 1. Build the prompt
# ---------------------------------------------------------------------------
def describe_df(df, name, n_rows=3):
    """Small text summary of a dataframe (keeps the prompt cheap)."""
    return (
        f"{name}: shape={df.shape}\n"
        f"dtypes:\n{df.dtypes.to_string()}\n"
        f"first {n_rows} rows:\n{df.head(n_rows).to_string()}\n"
    )


def build_prompt(insights, old_df, new_df):
    return f"""You are a data QA engineer. A dataset was cleaned. Write Python code
that checks whether each issue listed in the ORIGINAL INSIGHTS is now fixed.

ORIGINAL INSIGHTS (written before cleaning):
{insights}

{describe_df(old_df, "old_df")}
{describe_df(new_df, "new_df")}

RULES FOR YOUR CODE:
- Two dataframes already exist: `old_df` (before) and `new_df` (after). Do NOT reload data.
- `pd` (pandas), `np` (numpy) and `re` are already imported. Do NOT import anything else.
- Do not modify old_df or new_df. Do not read or write files.
- Start with:  results = []
- For EACH issue in the insights, add one entry:
    results.append({{"issue": "<short name>", "passed": <True or False>, "detail": "<numbers/examples proving it>"}})
- "passed" must be a real Python bool. True means the issue is FIXED in new_df.
- If a column from the insights no longer exists in new_df, handle it
  (e.g. passed=False with detail "column missing") instead of crashing.
- A few example checks: nulls remaining, duplicate rows, dtype is numeric/datetime,
  values outside a valid range, leading/trailing whitespace, case variants
  (e.g. compare nunique() with nunique() of the lowercased/stripped values).
- Also add one sanity check: rows in new_df vs old_df (detail only, passed=True
  unless new_df is empty).

OUTPUT: only Python code inside one ```python block. No explanation."""


# ---------------------------------------------------------------------------
# 2. Call Gemini (swap this out for a fake function when testing)
# ---------------------------------------------------------------------------
def ask_gemini(prompt):
    from google import genai   # imported here so the file loads without the SDK
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text


# ---------------------------------------------------------------------------
# 3. Clean up the model's reply and run it
# ---------------------------------------------------------------------------
def extract_code(reply):
    """Pull the code out of a ```python ... ``` block (or use the reply as is)."""
    match = re.search(r"```(?:python)?\s*(.*?)```", reply, re.DOTALL)
    return (match.group(1) if match else reply).strip()


BLOCKED = ["import ", "__import__", "open(", "exec(", "eval(", "os.", "sys.",
           "subprocess", "shutil", "to_csv", "to_excel", "read_csv", "read_excel"]


def is_safe(code):
    """Very basic guard. Returns (ok, bad_word)."""
    for word in BLOCKED:
        if word in code:
            return False, word
    return True, None


def run_check_code(code, old_df, new_df):
    """
    Run the generated code on COPIES of the dataframes.

    Returns a dict:
        results : list of {"issue", "passed", "detail"}  (empty if it failed)
        stdout  : anything the code printed
        error   : None, or the traceback text
    """
    out = {"results": [], "stdout": "", "error": None}

    ok, bad = is_safe(code)
    if not ok:
        out["error"] = f"Blocked: generated code contains '{bad}'"
        return out

    namespace = {"pd": pd, "np": np, "re": re,
                 "old_df": old_df.copy(), "new_df": new_df.copy()}
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, namespace)
        out["results"] = namespace.get("results", [])
        if not out["results"]:
            out["error"] = "Code ran but `results` is empty or missing."
    
    except Exception:
        out["error"] = traceback.format_exc()
    out["stdout"] = buffer.getvalue()
    return out


# ---------------------------------------------------------------------------
# 4. Tidy the results and print a report
# ---------------------------------------------------------------------------
def normalise_results(results):
    """Make sure every entry has the 3 keys and 'passed' is a real bool."""
    tidy = []
    for r in results:
        if not isinstance(r, dict):
            continue
        tidy.append({
            "issue": str(r.get("issue", "unnamed check")),
            "passed": bool(r.get("passed", False)),
            "detail": str(r.get("detail", "")),
        })
    return tidy


def print_report(results):
    if not results:
        print("No results to show.")
        return
    for r in results:
        tag = "FIXED    " if r["passed"] else "NOT FIXED"
        print(f"[{tag}] {r['issue']}")
        if r["detail"]:
            print(f"            {r['detail']}")
    done = sum(r["passed"] for r in results)
    print(f"\nScore: {done}/{len(results)} issues fixed")


# ---------------------------------------------------------------------------
# 5. The one function you call from the notebook
# ---------------------------------------------------------------------------
def evaluate_cleaning(insights, old_df, new_df, llm=ask_gemini, max_retries=1, show_code=False):
    """
    insights    : the text the model generated BEFORE cleaning
    old_df      : dataframe before cleaning
    new_df      : dataframe after cleaning
    llm         : function(prompt) -> text. Default is Gemini.
    max_retries : extra Gemini calls allowed IF the generated code crashes
                  (set 0 to guarantee exactly one call)
    show_code   : print the generated check code (good for learning/debugging)

    Returns a dict: {"results": [...], "all_fixed": bool, "code": str, "error": str|None}
    """
    prompt = build_prompt(insights, old_df, new_df)
    code, run = "", {"results": [], "stdout": "", "error": "not run"}

    for attempt in range(max_retries + 1):
        reply = llm(prompt)
        code = extract_code(reply)
        if show_code:
            print(f"--- generated check code (attempt {attempt + 1}) ---\n{code}\n")

        run = run_check_code(code, old_df, new_df)
        if run["error"] is None:
            break

        # Tell the model what went wrong so the retry can fix it
        prompt = (build_prompt(insights, old_df, new_df)
                  + f"\n\nYour previous code failed with:\n{run['error']}\n"
                    f"Previous code:\n{code}\nReturn a corrected version.")

    results = normalise_results(run["results"])
    print_report(results)
    if run["error"]:
        print("\nCheck code failed:\n", run["error"])

    return {
        "results": results,
        "all_fixed": bool(results) and all(r["passed"] for r in results),
        "code": code,
        "error": run["error"],
    }
