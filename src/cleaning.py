import numpy as np
import pandas as pd

LATE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
CAP_Q = 0.99


def fit_cleaner(train):
    """Hitung semua angka pengisi dan batas dari data TRAIN saja."""
    has_income = train["MonthlyIncome"].notna()
    p = {
        "age_median": train.loc[train["age"] > 18, "age"].median(),
        "late_median": {c: train.loc[train[c] < 96, c].median() for c in LATE_COLS},
        "income_median": train["MonthlyIncome"].median(),
        "debtratio_median": train.loc[has_income, "DebtRatio"].median(),
        "dependents_median": train["NumberOfDependents"].median(),
        "cap_util": train["RevolvingUtilizationOfUnsecuredLines"].quantile(CAP_Q),
        "cap_debtratio": train.loc[has_income, "DebtRatio"].quantile(CAP_Q),
        "cap_income": train["MonthlyIncome"].quantile(CAP_Q),
    }
    return p


def apply_cleaner(df, p):
    """Terapkan aturan cleaning memakai angka dari fit_cleaner."""
    df = df.copy()

    # Flag dibuat sebelum nilai asli diubah
    df["income_missing"] = df["MonthlyIncome"].isna().astype(int)
    df["late_code_flag"] = (df[LATE_COLS] >= 96).any(axis=1).astype(int)

    # Kode 96/98 bukan jumlah sungguhan
    for c in LATE_COLS:
        df.loc[df[c] >= 96, c] = p["late_median"][c]

    # Usia tidak valid
    df.loc[df["age"] <= 18, "age"] = p["age_median"]

    # Income kosong -> DebtRatio berisi angka utang mutlak, bukan rasio
    df.loc[df["income_missing"] == 1, "DebtRatio"] = p["debtratio_median"]

    # Isi nilai kosong
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(p["income_median"])
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(p["dependents_median"])

    # Batasi nilai ekstrem
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=p["cap_util"])
    df["DebtRatio"] = df["DebtRatio"].clip(upper=p["cap_debtratio"])
    df["MonthlyIncome"] = df["MonthlyIncome"].clip(upper=p["cap_income"])

    return df