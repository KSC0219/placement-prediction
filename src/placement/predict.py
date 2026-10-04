"""Predict placement for a single student from the command line.

Example:
    python -m src.placement.predict --cgpa 8.2 --aptitude 78 --internships 1 --branch CSE
"""
import argparse

import joblib
import pandas as pd

from .config import MODEL_PATH


def predict_student(student: dict, model=None) -> dict:
    model = model or joblib.load(MODEL_PATH)
    row = pd.DataFrame([student])
    probability = float(model.predict_proba(row)[0, 1])
    return {"placed": probability >= 0.5, "probability": round(probability, 3)}


def main() -> None:
    p = argparse.ArgumentParser(description="Predict a student's placement outcome")
    p.add_argument("--gender", default="Male", choices=["Male", "Female"])
    p.add_argument("--branch", default="CSE")
    p.add_argument("--ssc", type=float, default=80, help="10th class percentage")
    p.add_argument("--hsc", type=float, default=80, help="12th / intermediate percentage")
    p.add_argument("--cgpa", type=float, required=True)
    p.add_argument("--internships", type=int, default=0)
    p.add_argument("--projects", type=int, default=2)
    p.add_argument("--certifications", type=int, default=1)
    p.add_argument("--aptitude", type=float, required=True, help="aptitude test score out of 100")
    p.add_argument("--soft-skills", type=float, default=6.5, help="soft skills rating out of 10")
    p.add_argument("--backlogs", type=int, default=0)
    p.add_argument("--extracurricular", default="No", choices=["Yes", "No"])
    a = p.parse_args()

    student = {
        "gender": a.gender,
        "branch": a.branch.upper(),
        "ssc_percentage": a.ssc,
        "hsc_percentage": a.hsc,
        "cgpa": a.cgpa,
        "internships": a.internships,
        "projects": a.projects,
        "certifications": a.certifications,
        "aptitude_score": a.aptitude,
        "soft_skills_score": a.soft_skills,
        "backlogs": a.backlogs,
        "extracurricular": a.extracurricular,
    }
    result = predict_student(student)
    verdict = "LIKELY TO BE PLACED" if result["placed"] else "NOT LIKELY TO BE PLACED"
    print(f"{verdict} (probability {result['probability']:.1%})")


if __name__ == "__main__":
    main()
