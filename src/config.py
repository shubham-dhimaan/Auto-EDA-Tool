"""
Configuration settings for Auto EDA pipeline.
Centralizes environment variables, API keys, and default model identifiers.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Default Models
DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-2.5-flash")
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "gemini-3.5-flash-lite")

# Directory Paths
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SKILLS_DIR = ROOT_DIR / "skills"
DATASETS_DIR = ROOT_DIR / "datasets"
