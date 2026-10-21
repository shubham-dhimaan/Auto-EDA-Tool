# Skill: outliers

## When to use
An insight mentions outliers, anomalous, suspicious, extreme, or out-of-range values (e.g. scores of 999, 105, 120 on a 0-100 scale).

## Rules
- If the insight gives a known valid range (e.g. 0 to 100), treat anything outside it as invalid and set it to NaN. Do NOT clip it, because a score of 999 is a data entry error, not a real high score.
- If there is no known valid range, use the IQR rule (1.5 x IQR) to flag outliers, and set them to NaN only for clear errors. Otherwise leave them and print how many were flagged.
- The column must already be numeric. If not, run numeric_conversion first.
- Do this AFTER numeric_conversion and BEFORE missing-value handling, so the NaNs created here get filled together with the rest.
- Print how many values were changed.

## Example
```python
# known valid range
col, low, high = "Score", 0, 100
if col in df.columns:
    bad = (df[col] < low) | (df[col] > high)
    print(f"{col}: {int(bad.sum())} out-of-range values set to NaN")
    df.loc[bad, col] = np.nan

# no known range: IQR rule (flag only)
col = "Some_Numeric_Column"
if col in df.columns:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    flagged = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
    print(f"{col}: {int(flagged.sum())} values flagged as outliers")
```
