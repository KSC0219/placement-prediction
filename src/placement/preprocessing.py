"""Data cleaning and the feature-transformation pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate students, normalise text columns and clip out-of-range values."""
    df = df.drop_duplicates(subset="student_id") if "student_id" in df else df.drop_duplicates()
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col in df:
            df[col] = df[col].astype(str).str.strip().str.title()
            df.loc[df[col].isin(["Nan", "None", ""]), col] = pd.NA
    df["branch"] = df["branch"].str.upper()
    for col in ("ssc_percentage", "hsc_percentage", "aptitude_score"):
        df[col] = df[col].clip(0, 100)
    df["cgpa"] = df["cgpa"].clip(0, 10)
    return df


def split_xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET].astype(int)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)]
    )
