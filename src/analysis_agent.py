"""
analysis_agent.py
-----------------
Multi-agent EDA layer for AutoEDA, built on the google-genai SDK (no agent frameworks).
Works with the existing runner.py (runner.py is NOT modified).

Flow:
    cleaned DataFrame
        -> PlannerAgent   (root)  : plans which EDA steps to run
        -> EDAAgent               : for each step, writes an insight + a Python function
        -> ValidatorAgent         : checks the step + function (static safety check + LLM review)
        -> ExecutorAgent          : runs the function with runner.execute_eda_steps(),
                                    then decides accept / retry / skip
        -> report saved to output/analysis_report.json

Each "agent" here is just a Gemini call with its own system prompt and a
JSON-only reply. run_analysis() at the bottom wires them together.

Setup: needs GEMINI_API_KEY in your .env (same as the rest of AutoEDA).
"""

import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    from src import runner
except ImportError:
    from scripts import runner  # your existing runner.py (unchanged)

load_dotenv()

MODEL = "gemini-3.5-flash-lite"      # change to whichever model your notebook uses
MAX_STEPS = 6                   # how many EDA steps the planner may create
MAX_RETRIES = 2                 # how many times a step may be rewritten
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ---------------------------------------------------------------------------
# Base agent: one Gemini call with a system prompt, returns parsed JSON
# ---------------------------------------------------------------------------
class Agent:
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt

    def ask(self, prompt: str) -> dict:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                response_mime_type="application/json",   # forces a JSON reply
                temperature=0.2,
            ),
        )
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            # Rare, but do not crash the whole run if the model returns bad JSON
            return {"error": f"{self.name} returned invalid JSON", "raw": response.text}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_metadata(df: pd.DataFrame) -> dict:
    """Short description of the dataset. Agents never see all the rows.
    This same dict is passed to runner.py as its `metadata` argument."""
    return {
        "shape": list(df.shape),
        "columns": {
            col: {
                "dtype": str(df[col].dtype),
                "nulls": int(df[col].isna().sum()),
                "unique": int(df[col].nunique()),
            }
            for col in df.columns
        },
        "sample_rows": df.head(3).to_dict(orient="records"),
    }


def to_jsonable(obj, max_rows: int = 20):
    """runner.py returns whatever the function returned (DataFrame, Series,
    dict, number). This turns it into plain JSON-friendly data."""
    if isinstance(obj, pd.DataFrame):
        return obj.head(max_rows).reset_index().to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.head(max_rows).to_dict()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, max_rows) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, max_rows) for v in obj[:max_rows]]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# 1. Planner (root agent)
# ---------------------------------------------------------------------------
planner = Agent(
    "planner",
    f"""You are the lead data analyst. You get metadata of a CLEANED dataset.
Plan at most {MAX_STEPS} EDA steps that would give the most useful insights.
Cover a mix: distributions, group-bys, correlations, outliers, categories.
Only use column names that exist in the metadata.

Reply with JSON only:
{{"steps": [{{"id": 1, "question": "...", "columns": ["..."], "method": "..."}}]}}""",
)


# ---------------------------------------------------------------------------
# 2. EDA agent: writes the insight + a function for ONE step
#    The reply format matches the function objects runner.py expects.
# ---------------------------------------------------------------------------
eda_agent = Agent(
    "eda_agent",
    """You are a Python/Pandas EDA engineer. You receive ONE planned EDA step.
Write a single function that answers it.

Rules for the function:
- It takes ONE argument: df (a pandas DataFrame). Do not modify df in place.
- `pd` (pandas) and `np` (numpy) are already available. Do not import anything else.
- Do not read or write files, do not print, do not plot, and do not use
  eval, exec, open, os, sys or subprocess.
- Return a small result: a dict, a Series, or a DataFrame with <= 20 rows.
  Include the numbers needed to explain the insight.
- Handle missing values safely so the function does not crash.

Reply with JSON only, exactly this structure:
{"name": "function_name",
 "columns": ["column_used"],
 "action": "short_action_name",
 "reason": "what question this answers and why it matters",
 "requires_review": false,
 "code": "def function_name(df):\\n    ..."}""",
)


# ---------------------------------------------------------------------------
# 3. Validator: static safety check (plain Python) + LLM review
# ---------------------------------------------------------------------------
FORBIDDEN_NAMES = {"eval", "exec", "open", "compile", "__import__", "os", "sys",
                   "subprocess", "input", "print"}
ALLOWED_IMPORTS = {"pandas", "numpy"}


def static_check(code: str, function_name: str) -> list:
    """Fast rule-based check using Python's ast module. Returns a list of problems."""
    problems = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    defined = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if function_name not in defined:
        problems.append(f"Function '{function_name}' is not defined in the code.")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"Import '{alias.name}' is not allowed.")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"Import from '{node.module}' is not allowed.")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"Use of '{node.id}' is not allowed.")
    return problems


validator_llm = Agent(
    "validator",
    """You review one EDA step and its Python function before it runs.
Check:
1. Does the code actually answer the stated question?
2. Do the columns it uses exist in the dataset metadata, with suitable dtypes?
3. Will it avoid obvious errors (division by zero, wrong dtype, empty groups)?
4. Does it return a small result (dict, Series or DataFrame <= 20 rows)?

Reply with JSON only:
{"approved": true or false, "issues": ["..."], "suggestion": "how to fix, or empty"}""",
)


class ValidatorAgent:
    def validate(self, step: dict, eda_out: dict, metadata: dict) -> dict:
        # Step 1: rule-based safety check (cheap, no LLM call)
        problems = static_check(eda_out.get("code", ""), eda_out.get("name", ""))
        if problems:
            return {"approved": False, "issues": problems, "suggestion": "Fix the listed problems."}

        # Step 2: LLM review of logic and column usage
        prompt = (
            f"DATASET METADATA:\n{json.dumps(metadata, default=str)}\n\n"
            f"PLANNED STEP:\n{json.dumps(step)}\n\n"
            f"FUNCTION:\n{eda_out['code']}"
        )
        return validator_llm.ask(prompt)


# ---------------------------------------------------------------------------
# 4. Executor: runs the code via runner.execute_eda_steps, then decides
# ---------------------------------------------------------------------------
def call_runner(eda_out: dict, df: pd.DataFrame, metadata: dict) -> dict:
    """
    Wraps runner.execute_eda_steps() for ONE function.

    runner expects a JSON string like {"functions": [ {name, columns, action,
    reason, requires_review, code} ]} and returns a list of
    {"name", "actions", "reason", "output"}.

    runner already tries one AI repair on a failure. If the repair also fails,
    runner just skips the function, so an EMPTY list means "it failed".
    """
    payload = json.dumps({"functions": [eda_out]})
    try:
        report = runner.execute_eda_steps(payload, df, metadata)
    except Exception as e:  # e.g. bad JSON or API problem inside runner
        return {"ok": False, "result": None, "error": f"{type(e).__name__}: {e}"}

    if not report:
        return {"ok": False, "result": None,
                "error": "runner could not run this function, even after its automatic repair."}

    return {"ok": True, "result": to_jsonable(report[0]["output"]), "error": None}


executor_llm = Agent(
    "executor",
    """You are the executor. You get a step, the function that ran, and the runner output.
Decide:
- "accept": the output is correct and answers the question. Write a 1-2 sentence
  plain-English insight that uses the actual numbers.
- "retry": the run failed or the output is wrong, empty or misleading. Explain what to change.
- "skip": the step cannot be answered with this data. Explain why.

Reply with JSON only:
{"decision": "accept" | "retry" | "skip", "insight": "...", "feedback": "..."}""",
)


class ExecutorAgent:
    def execute(self, step: dict, eda_out: dict, df: pd.DataFrame, metadata: dict) -> dict:
        run = call_runner(eda_out, df, metadata)
        prompt = (
            f"STEP:\n{json.dumps(step)}\n\nFUNCTION:\n{eda_out['code']}\n\n"
            f"RUNNER OUTPUT:\n{json.dumps(run, default=str)[:4000]}"
        )
        decision = executor_llm.ask(prompt)
        decision["run"] = run
        return decision


# ---------------------------------------------------------------------------
# Orchestrator: wires the four agents together
# ---------------------------------------------------------------------------
def run_analysis(df: pd.DataFrame) -> dict:
    metadata = build_metadata(df)
    metadata_text = json.dumps(metadata, default=str, indent=2)
    validator = ValidatorAgent()
    executor = ExecutorAgent()

    # Root agent plans the work
    plan = planner.ask(f"DATASET METADATA:\n{metadata_text}")
    steps = plan.get("steps", [])[:MAX_STEPS]
    print(f"Planner created {len(steps)} steps.")

    results = []
    for step in steps:
        print(f"\nStep {step['id']}: {step['question']}")
        feedback = ""
        final = {"step": step, "status": "failed", "attempts": 0}

        for attempt in range(1, MAX_RETRIES + 2):
            final["attempts"] = attempt

            # EDA agent writes (or rewrites) the function
            prompt = f"DATASET METADATA:\n{metadata_text}\n\nSTEP:\n{json.dumps(step)}"
            if feedback:
                prompt += f"\n\nPREVIOUS ATTEMPT FAILED. FEEDBACK:\n{feedback}"
            eda_out = eda_agent.ask(prompt)
            if "code" not in eda_out or "name" not in eda_out:
                feedback = "Your reply must include 'name' and 'code'."
                continue

            # Validator checks it before anything runs
            check = validator.validate(step, eda_out, metadata)
            if not check.get("approved"):
                feedback = f"Validator issues: {check.get('issues')}. {check.get('suggestion', '')}"
                print(f"  attempt {attempt}: rejected by validator")
                continue

            # Executor runs it through runner.py and decides
            outcome = executor.execute(step, eda_out, df, metadata)
            decision = outcome.get("decision")
            print(f"  attempt {attempt}: executor says {decision}")

            if decision == "accept":
                final.update(status="accepted", code=eda_out["code"],
                             insight=outcome.get("insight"), result=outcome["run"]["result"])
                break
            if decision == "skip":
                final.update(status="skipped", reason=outcome.get("feedback"))
                break
            feedback = outcome.get("feedback", "Run failed.")  # retry

        results.append(final)

    report = {"plan": steps, "results": results}
    out_path = OUTPUT_DIR / "analysis_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved report to {out_path}")
    return report


if __name__ == "__main__":
    # Quick test: point this at your cleaned dataset from the cleaning step
    cleaned = pd.read_csv(OUTPUT_DIR / "cleaned_data.csv")
    run_analysis(cleaned)
