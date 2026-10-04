"""Train several classifiers, compare them with cross-validation and save the best one."""
import json

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import database
from .config import CSV_PATH, MODEL_DIR, MODEL_PATH, RANDOM_STATE
from .generate_data import generate
from .preprocessing import build_preprocessor, clean, split_xy

CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_STATE),
    "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "knn": KNeighborsClassifier(n_neighbors=15),
}


def make_pipeline(model) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def compare_models(X, y) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in CANDIDATES.items():
        scores = cross_val_score(make_pipeline(model), X, y, cv=cv, scoring="f1")
        rows.append({"model": name, "cv_f1_mean": scores.mean(), "cv_f1_std": scores.std()})
    return pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)


def main() -> None:
    if not CSV_PATH.exists():
        print("No dataset found; generating a synthetic one...")
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(CSV_PATH, index=False)

    n = database.load_csv_to_db()
    print(f"Loaded {n} rows into SQLite.\n")
    print("Placement rate by branch (SQL):")
    print(database.placement_rate_by_branch().to_string(index=False), "\n")

    df = clean(database.fetch_training_data())
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("5-fold cross-validation (F1) on the training set:")
    results = compare_models(X_train, y_train)
    print(results.to_string(index=False, float_format="%.3f"), "\n")

    best_name = results.loc[0, "model"]
    best = make_pipeline(CANDIDATES[best_name]).fit(X_train, y_train)

    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]
    metrics = {
        "model": best_name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    print(f"Best model: {best_name}. Hold-out test set results:")
    print(classification_report(y_test, pred, target_names=["Not placed", "Placed"]))
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
