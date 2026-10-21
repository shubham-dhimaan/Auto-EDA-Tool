# Skill: date_parsing

## When to use
An insight mentions a date column stored as text, mixed date formats, impossible dates (e.g. 31/02/1992), or placeholders like "tbd".

## Rules
- Convert with `pd.to_datetime(..., errors="coerce", format="mixed")` so mixed formats are parsed and invalid or placeholder values become NaT instead of crashing.
- Strip spaces first.
- Do not guess or fill invalid dates. Leave them as NaT, the missing-value step decides what to do.
- Keep the result as a real datetime column, do not convert back to strings.
- Do this AFTER categorical_cleanup and BEFORE missing-value handling.

## Example
```python
if "Class_Date" in df.columns:
    df["Class_Date"] = pd.to_datetime(
        df["Class_Date"].astype("string").str.strip(),
        errors="coerce",
        format="mixed",
    )
```
