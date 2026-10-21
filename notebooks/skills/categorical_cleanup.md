# Skill: categorical_cleanup

## When to use
An insight mentions inconsistent text values: extra spaces, mixed capitalization, abbreviations, synonyms, typos, or too many distinct categories.

## Rules
- Do this BEFORE type conversion, missing-value handling and duplicate removal, because it exposes hidden duplicates.
- Strip leading and trailing spaces, and collapse repeated inner spaces.
- Build a mapping dict from lowercase cleaned text to the canonical label, so abbreviations and synonyms get fixed. Fall back to `.str.title()` for anything not in the mapping.
- Do NOT use plain `.str.title()` alone on labels with small words (e.g. "Defence Against the Dark Arts"). Use the mapping for those.
- If the insight says a column should only have N valid values, set anything outside that list to NaN (the missing-value step handles it later).
- Only touch columns that exist in `df`.

## Example
```python
def clean_text(series):
    return (series.astype("string")
                  .str.strip()
                  .str.replace(r"\s+", " ", regex=True))

# 1. whitespace for every flagged column
for col in ["Student_Name", "Subject", "Attendance_Status", "Teacher", "House"]:
    if col in df.columns:
        df[col] = clean_text(df[col])

# 2. mapping of lowercase text -> canonical label
subject_map = {
    "dada": "Defence Against the Dark Arts",
    "defence against the dark arts": "Defence Against the Dark Arts",
}
if "Subject" in df.columns:
    lowered = df["Subject"].str.lower()
    df["Subject"] = lowered.map(subject_map).fillna(df["Subject"].str.title())

# 3. simple synonym fix + title case
if "Attendance_Status" in df.columns:
    status_map = {"here": "Present"}
    lowered = df["Attendance_Status"].str.lower()
    df["Attendance_Status"] = lowered.map(status_map).fillna(df["Attendance_Status"].str.title())

# 4. title case for names, then enforce a fixed set of valid values
for col in ["Student_Name", "Teacher"]:
    if col in df.columns:
        df[col] = df[col].str.title()

valid_houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
if "House" in df.columns:
    df["House"] = df["House"].str.title()
    df.loc[~df["House"].isin(valid_houses), "House"] = pd.NA
```
