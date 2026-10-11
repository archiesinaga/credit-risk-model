import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.cleaning import apply_cleaner

CUTOFF = 0.158  # rasio biaya 5:1, dikunci dari validation
ACTIONS = {
    "A": "Setujui",
    "B": "Setujui",
    "C": "Setujui dengan syarat",
    "D": "Tinjau manual",
    "E": "Tolak atau minta agunan",
}

st.set_page_config(page_title="Credit Risk Early Warning", layout="wide")


@st.cache_resource
def load_models():
    return {
        "xgb": joblib.load(ROOT / "models" / "xgb.joblib"),
        "iso": joblib.load(ROOT / "models" / "iso_xgb.joblib"),
        "params": joblib.load(ROOT / "models" / "cleaning_params.joblib"),
        "grade": joblib.load(ROOT / "models" / "grade_config.joblib"),
    }


@st.cache_data
def load_scored():
    df = pd.read_csv(ROOT / "reports" / "scored_test.csv")
    df["PD (%)"] = (df["PD"].clip(upper=0.99) * 100).round(1)
    return df


m = load_models()
scored = load_scored()
labels = m["grade"]["labels"]

st.title("Credit Risk Early Warning")
st.info(
    "Demo portofolio. Data adalah data publik Kaggle (Give Me Some Credit), "
    "bukan data BRI. Metode bisa dipindahkan ke data bank, datanya tidak."
)

page = st.sidebar.radio(
    "Halaman", ["Portfolio overview", "Early warning list", "Score a borrower"]
)

# ---------------------------------------------------------------- Halaman 1
if page == "Portfolio overview":
    st.header("Portfolio overview")
    summary = (
        scored.groupby("grade")
        .agg(peminjam=("PD", "size"), default_aktual=("actual_default", "mean"))
        .reindex(labels)
    )
    summary["porsi"] = summary["peminjam"] / summary["peminjam"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah peminjam", f"{len(scored):,}")
    c2.metric("Default rate aktual", f"{scored['actual_default'].mean():.2%}")
    c3.metric("Ditandai peringatan dini", f"{(scored['PD'] >= CUTOFF).mean():.1%}")

    left, right = st.columns(2)
    left.subheader("Porsi peminjam per grade")
    left.bar_chart(summary["porsi"])
    right.subheader("Default rate aktual per grade")
    right.bar_chart(summary["default_aktual"])

    st.dataframe(
        summary.rename(
            columns={
                "peminjam": "Peminjam",
                "default_aktual": "Default aktual",
                "porsi": "Porsi",
            }
        ).style.format({"Default aktual": "{:.2%}", "Porsi": "{:.1%}"})
    )
    st.caption("Angka dihitung dari test set (22.500 peminjam) yang tidak dipakai saat training.")

# ---------------------------------------------------------------- Halaman 2
elif page == "Early warning list":
    st.header("Early warning list")
    c1, c2 = st.columns(2)
    sel = c1.multiselect("Grade", labels, default=["D", "E"])
    n = c2.slider("Jumlah baris", 10, 500, 50, step=10)

    view = (
        scored[scored["grade"].isin(sel)]
        .sort_values("PD", ascending=False)
        .head(n)
    )
    cols = [
        "PD (%)",
        "grade",
        "reasons",
        "age",
        "RevolvingUtilizationOfUnsecuredLines",
        "MonthlyIncome",
        "actual_default",
    ]
    st.dataframe(view[cols], hide_index=True)
    st.caption(
        "Kolom reasons berisi 3 faktor yang paling menaikkan risiko (SHAP). "
        "PD ditampilkan maksimum 99%. Nilai utilisasi dan income adalah nilai "
        "setelah cleaning (sudah dibatasi)."
    )

# ---------------------------------------------------------------- Halaman 3
else:
    st.header("Score a borrower")
    with st.form("score"):
        c1, c2 = st.columns(2)
        age = c1.number_input("Usia", min_value=18, max_value=100, value=40)
        income_known = c1.checkbox("Income bulanan diketahui", value=True)
        income = c1.number_input(
            "Income bulanan (satuan data Kaggle)", min_value=0, max_value=1000000, value=5000, step=500
        )
        debt = c1.number_input(
            "Cicilan utang bulanan", min_value=0, max_value=1000000, value=1500, step=100
        )
        util = c1.slider("Utilisasi kredit tanpa agunan (0 sampai 1,2)", 0.0, 1.2, 0.3, 0.01)
        late30 = c2.number_input("Telat 30-59 hari (jumlah)", min_value=0, max_value=20, value=0)
        late60 = c2.number_input("Telat 60-89 hari (jumlah)", min_value=0, max_value=20, value=0)
        late90 = c2.number_input("Telat 90+ hari (jumlah)", min_value=0, max_value=20, value=0)
        open_lines = c2.number_input("Jumlah kredit aktif", min_value=0, max_value=58, value=8)
        real_estate = c2.number_input("Jumlah kredit properti", min_value=0, max_value=54, value=1)
        dependents = c2.number_input("Jumlah tanggungan", min_value=0, max_value=10, value=0)
        submit = st.form_submit_button("Hitung skor")

    if submit:
        raw = pd.DataFrame(
            [
                {
                    "RevolvingUtilizationOfUnsecuredLines": util,
                    "age": age,
                    "NumberOfTime30-59DaysPastDueNotWorse": late30,
                    "DebtRatio": debt / max(income, 1),
                    "MonthlyIncome": income if income_known else np.nan,
                    "NumberOfOpenCreditLinesAndLoans": open_lines,
                    "NumberOfTimes90DaysLate": late90,
                    "NumberRealEstateLoansOrLines": real_estate,
                    "NumberOfTime60-89DaysPastDueNotWorse": late60,
                    "NumberOfDependents": dependents,
                }
            ]
        )
        feats = m["xgb"].get_booster().feature_names
        X = apply_cleaner(raw, m["params"])[feats]
        pd_val = float(m["iso"].predict(m["xgb"].predict_proba(X)[:, 1])[0])
        g = str(
            pd.cut([pd_val], bins=m["grade"]["bins"], labels=labels, right=False)[0]
        )

        r1, r2, r3 = st.columns(3)
        r1.metric("Probabilitas default (PD)", f"{min(pd_val, 0.99):.1%}")
        r2.metric("Grade", g)
        r3.metric("Saran tindakan", ACTIONS[g].split()[0])
        st.info(f"Saran tindakan lengkap: {ACTIONS[g]}")
        if pd_val >= CUTOFF:
            st.warning(f"PD di atas cutoff peringatan dini ({CUTOFF}).")
        else:
            st.success(f"PD di bawah cutoff peringatan dini ({CUTOFF}).")