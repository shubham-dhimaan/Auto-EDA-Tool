# Skill: missing_values

## When to use
An insight mentions missing values, nulls, NaN, or empty cells in a column.

## Rules
- Numeric column, low skew: fill with the median.
- Numeric column, more than 40% missing: drop the column instead of filling.
- Categorical column: fill with the mode, or "Unknown" if there is no clear mode.
- Always check that the column exists before touching it.
- Work on `df` directly and never reload the file.

## Example
```python
col = "age"
if col in df.columns:
    if df[col].isna().mean() > 0.40:
        df = df.drop(columns=[col])
    else:
        df[col] = df[col].fillna(df[col].median())
```
