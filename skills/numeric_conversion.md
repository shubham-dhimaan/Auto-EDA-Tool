# Skill: numeric_conversion

## When to use
An insight says a column that should be a number is stored as text or categorical, or contains malformed entries like "40.0.0", "-", or "year".

## Rules
- Never call `astype(float)` or `astype(int)` directly on dirty text, it will crash.
- Clean the text first: strip spaces, pull out the first valid number with a regex.
- Convert with `pd.to_numeric(..., errors="coerce")` so unfixable values become NaN instead of crashing.
- For whole-number columns (like a year or class level) use the nullable integer type `"Int64"` after converting, so NaN is allowed.
- Do this AFTER categorical_cleanup and BEFORE outlier handling.

## Example
```python
def to_number(series):
    cleaned = (series.astype("string")
                     .str.strip()
                     .str.extract(r"(\d+(?:\.\d+)?)", expand=False))  # "40.0.0" -> "40.0"
    return pd.to_numeric(cleaned, errors="coerce")

if "Score" in df.columns:
    df["Score"] = to_number(df["Score"])

if "Year_Level" in df.columns:
    df["Year_Level"] = to_number(df["Year_Level"]).round().astype("Int64")
```
