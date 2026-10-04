from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

CSV_PATH = DATA_DIR / "placement_data.csv"
DB_PATH = DATA_DIR / "placement.db"
MODEL_PATH = MODEL_DIR / "placement_model.joblib"

TARGET = "placed"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "ssc_percentage",
    "hsc_percentage",
    "cgpa",
    "internships",
    "projects",
    "certifications",
    "aptitude_score",
    "soft_skills_score",
    "backlogs",
]
CATEGORICAL_FEATURES = ["gender", "branch", "extracurricular"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
