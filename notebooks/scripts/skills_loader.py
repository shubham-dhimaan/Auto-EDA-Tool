"""
skills_loader.py
Picks the right "skill" files for the insights and builds the code-gen prompt.
Plug build_codegen_prompt() into your existing code generator in cleaner.py.

"""
from pathlib import Path

SKILLS_DIR = Path("skills")

# Which words in an insight point to which skill file (file name without .md)
KEYWORDS = {
    "categorical_cleanup": ["inconsistent", "whitespace", "capitaliz", "casing",
                            "typo", "categorical", "abbreviation", "synonym", "distinct values"],
    "numeric_conversion": ["numeric", "stored as text", "data type", "dtype",
                           "malformed", "integer", "convert"],
    "date_parsing": ["date", "datetime", "timestamp"],
    "outliers": ["outlier", "anomal", "extreme", "suspicious", "out of range",
                 "outside", "invalid range"],
    "missing_values": ["missing", "null", "nan", "empty", "impute"],
    "duplicates": ["duplicate", "repeated"],
}


def pick_skills(insights):

    """
    Takes in model generated insights and returns a list of relevant skill names.

    Args:
        insights (str): The string output of model generated cleaning insights.

    Returns:
        list: A list of relevant skill names.
    """
    
    text = insights.lower()
    chosen = []
    for skill_name, words in KEYWORDS.items():
        if any(w in text for w in words):
            chosen.append(skill_name)
    return chosen


def load_skill_text(insights):

    """
    Takes in model generated insights and calls pick_skills to determine relevant skill files
    and returns the text of those skill files joined together. 
    If no skills are found, returns an empty string.

    Args:
        insights (str): The string output of model generated cleaning insights.

    Returns:
        str: The text of the relevant skill files joined together, or an empty string if no

    ."""


    skill_names = pick_skills(insights)

    parts = []
    for name in skill_names:
        path = SKILLS_DIR / f"{name}.md"
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(parts)


