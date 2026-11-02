# Project Overview & Architecture
# Auto EDA: A Pipeline-Based Automated EDA and Data Cleaning Engine

Auto EDA is a deterministic, pipeline-driven engine that automates data profiling, cleaning, code generation, error recovery, and exploratory data analysis.

Rather than relying on unconstrained autonomous agent loops, Auto EDA uses a **deterministic pipeline structure** with a **self-healing code execution loop**:
1. **Profiling & Inspection**: Generates reports (via `ydata-profiling` and custom dataset inspectors).
2. **Actionable Recommendations**: Uses Gemini with targeted domain skills (`skills/*.md`) to identify issues (missing values, typos, incorrect data types, outliers, inconsistent categoricals).
3. **Structured Code Generation**: Converts recommendations into isolated Python cleaning functions in structured JSON.
4. **Execution & Self-Correction**: Executes generated functions on the DataFrame copy. When a runtime error occurs, the failed code, traceback, and metadata are sent to the model to repair and retry execution.
5. **Evaluation & Post-Cleaning EDA**: Verifies that original issues were resolved and recommends deep analytical breakdowns and statistics.

---

## Project Structure

```text
Auto EDA/
├── src/                         # Core Python package
│   ├── __init__.py              # Package exports
│   ├── config.py                # Environment and configuration
│   ├── pipeline.py              # AutoEDAPipeline orchestrator
│   ├── cleaner.py               # Dataset inspection, recommendations, cleaning code gen
│   ├── runner.py                # Code executor with error repair loop
│   ├── analyser.py              # Post-cleaning exploratory analysis recommendations
│   ├── evaluator.py             # Validation checks comparing pre- and post-cleaning
│   ├── report_compressor.py     # Profiling report token compression
│   ├── skills_loader.py         # Dynamic prompt skill injector
│   ├── analysis_agent.py        # Multi-stage EDA analysis runner
│   └── analysis_agent_ollama.py # Local Ollama variant for analysis
├── skills/                      # Domain knowledge reference recipes (markdown)
│   ├── categorical_cleanup.md
│   ├── date_parsing.md
│   ├── duplicates.md
│   ├── missing_values.md
│   ├── numeric_conversion.md
│   └── outliers.md
├── datasets/                    # Sample messy and clean datasets for testing
├── notebooks/                   # Development notebooks & experiments
├── main.py                      # CLI entrypoint to run the pipeline
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd "Auto EDA"
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY="your-gemini-api-key-here"
   ```

---

## How to Run

### Option 1: Via Command Line (CLI)
Run the pipeline on any CSV dataset:
```bash
python main.py --data datasets/harry_potter_school_log_messy.csv
```

With an optional pre-generated profiling JSON report:
```bash
python main.py --data datasets/harry_potter_school_log_messy.csv --report "JSON reports/compressed_report.json"
```

Cleaned data is automatically written to `output/cleaned_data.csv`.

---

### Option 2: In Python or Jupyter Notebooks

```python
from src.pipeline import AutoEDAPipeline

# Initialize the pipeline
pipeline = AutoEDAPipeline(output_dir="output")

# Run end-to-end cleaning and analysis
results = pipeline.run("datasets/harry_potter_school_log_messy.csv")

cleaned_df = results["cleaned_df"]
print("Cleaned DataFrame shape:", cleaned_df.shape)
print("\nGenerated Cleaning Actions:")
print(results["cleaning_insights"])
print("\nExploratory Analysis Recommendations:")
print(results["analysis_insights"])
```
