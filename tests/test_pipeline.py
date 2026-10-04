import numpy as np
import pandas as pd
import pytest

from src.placement import database
from src.placement.generate_data import generate
from src.placement.predict import predict_student
from src.placement.preprocessing import clean, split_xy
from src.placement.train import CANDIDATES, make_pipeline


@pytest.fixture(scope="module")
def data():
    return generate(n_rows=600, seed=7)


def test_generated_data_shape_and_labels(data):
    assert len(data) == 600
    assert set(data["placed"].unique()) <= {0, 1}
    assert data["cgpa"].between(0, 10).all()


def test_clean_removes_duplicates_and_clips(data):
    dirty = pd.concat([data.head(5), data.head(5)])
    dirty.loc[dirty.index[0], "cgpa"] = 14
    dirty["branch"] = " cse "
    cleaned = clean(dirty)
    assert len(cleaned) == 5
    assert cleaned["cgpa"].max() <= 10
    assert (cleaned["branch"] == "CSE").all()


def test_sql_roundtrip_filters_invalid_rows(tmp_path, data):
    csv = tmp_path / "d.csv"
    db = tmp_path / "d.db"
    bad = data.copy()
    bad.loc[0, "cgpa"] = 42  # invalid, should be filtered by the SQL WHERE clause
    bad.to_csv(csv, index=False)
    assert database.load_csv_to_db(csv, db) == 600
    assert len(database.fetch_training_data(db)) == 599


def test_model_beats_baseline_and_predicts(data):
    X, y = split_xy(clean(data))
    model = make_pipeline(CANDIDATES["logistic_regression"]).fit(X[:500], y[:500])
    accuracy = (model.predict(X[500:]) == y[500:]).mean()
    baseline = max(y[500:].mean(), 1 - y[500:].mean())
    assert accuracy > baseline

    strong = {
        "gender": "Female", "branch": "CSE", "ssc_percentage": 92, "hsc_percentage": 94,
        "cgpa": 9.4, "internships": 2, "projects": 4, "certifications": 3,
        "aptitude_score": 92, "soft_skills_score": 9, "backlogs": 0, "extracurricular": "Yes",
    }
    weak = {**strong, "cgpa": 5.5, "aptitude_score": 30, "internships": 0, "projects": 0,
            "soft_skills_score": 3, "backlogs": 3}
    assert predict_student(strong, model)["probability"] > predict_student(weak, model)["probability"]


def test_unknown_branch_does_not_crash(data):
    X, y = split_xy(clean(data))
    model = make_pipeline(CANDIDATES["random_forest"]).fit(X, y)
    row = X.iloc[[0]].copy()
    row["branch"] = "AERO"
    assert model.predict(row).shape == (1,)
