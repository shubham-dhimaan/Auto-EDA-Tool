# Skill: duplicates

## When to use
An insight mentions duplicate rows or repeated records.

## Rules
- Drop exact duplicate rows only, keep the first occurrence.
- Reset the index afterwards.
- Print how many rows were removed.

## Example
```python
before = len(df)
df = df.drop_duplicates(keep="first").reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate rows")
```
