"""Generate a synthetic student placement dataset.

The data is synthetic: placement depends on the academic and skill features
through a logistic function plus noise, so the models have a real signal to
learn. To use a real dataset instead, place a CSV with the same columns at
data/placement_data.csv.
"""
import argparse

import numpy as np
import pandas as pd

from .config import CSV_PATH, RANDOM_STATE

BRANCHES = ["CSE", "ECE", "EEE", "MECH", "CIVIL", "IT"]


def generate(n_rows: int = 2000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ssc = np.clip(rng.normal(78, 9, n_rows), 45, 99).round(1)
    hsc = np.clip(0.6 * ssc + rng.normal(30, 7, n_rows), 45, 99).round(1)
    cgpa = np.clip(0.04 * hsc + rng.normal(4.2, 0.8, n_rows), 5.0, 10.0).round(2)
    internships = rng.poisson(0.8, n_rows).clip(0, 4)
    projects = rng.poisson(2.0, n_rows).clip(0, 6)
    certifications = rng.poisson(1.5, n_rows).clip(0, 6)
    aptitude = np.clip(rng.normal(65, 14, n_rows), 20, 100).round(0).astype(int)
    soft_skills = np.clip(rng.normal(6.5, 1.5, n_rows), 1, 10).round(1)
    backlogs = rng.choice([0, 0, 0, 0, 1, 1, 2, 3], n_rows)
    gender = rng.choice(["Male", "Female"], n_rows)
    branch = rng.choice(BRANCHES, n_rows, p=[0.25, 0.2, 0.12, 0.15, 0.1, 0.18])
    extracurricular = rng.choice(["Yes", "No"], n_rows, p=[0.45, 0.55])

    branch_effect = pd.Series(branch).map(
        {"CSE": 0.6, "IT": 0.5, "ECE": 0.2, "EEE": 0.0, "MECH": -0.3, "CIVIL": -0.4}
    ).to_numpy()

    logit = (
        -17.6
        + 1.1 * cgpa
        + 0.06 * aptitude
        + 0.45 * soft_skills
        + 0.55 * internships
        + 0.25 * projects
        + 0.15 * certifications
        + 0.015 * (ssc + hsc)
        - 0.6 * backlogs
        + 0.3 * (extracurricular == "Yes")
        + branch_effect
        + rng.normal(0, 0.8, n_rows)
    )
    placed = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame(
        {
            "student_id": np.arange(1, n_rows + 1),
            "gender": gender,
            "branch": branch,
            "ssc_percentage": ssc,
            "hsc_percentage": hsc,
            "cgpa": cgpa,
            "internships": internships,
            "projects": projects,
            "certifications": certifications,
            "aptitude_score": aptitude,
            "soft_skills_score": soft_skills,
            "backlogs": backlogs,
            "extracurricular": extracurricular,
            "placed": placed,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2000)
    args = parser.parse_args()

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Wrote {len(df)} rows to {CSV_PATH} (placement rate {df['placed'].mean():.1%})")


if __name__ == "__main__":
    main()
