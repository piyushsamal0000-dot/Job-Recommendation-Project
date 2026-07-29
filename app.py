"""
Job Recommendation System — Streamlit app
Internship minor project

Run with:  streamlit run app.py
Needs ai_job_dataset1.csv in the same folder (or update DATA_PATH below).
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = "ai_job_dataset1.csv"
SKILLS_COL = "required_skills"   # adjust if your CSV names it differently
SKILLS_SEP = ","

st.set_page_config(page_title="AI Job Match Finder", page_icon="🎯", layout="wide")

# ---------------------------------------------------------------
# DATA + MODEL — cached so this only runs once per session, not on
# every button click / widget interaction
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df[SKILLS_COL] = df[SKILLS_COL].fillna("")
    df["skills_list"] = df[SKILLS_COL].apply(
        lambda s: [x.strip().lower() for x in s.split(SKILLS_SEP) if x.strip()]
    )
    return df


@st.cache_resource
def train_model(df):
    mlb = MultiLabelBinarizer()
    skills_encoded = mlb.fit_transform(df["skills_list"])
    skills_df = pd.DataFrame(
        skills_encoded, columns=[f"skill_{s}" for s in mlb.classes_], index=df.index
    )

    profile_cats = [c for c in ["experience_level", "education_required",
                                 "employment_type", "company_size"] if c in df.columns]
    numeric_cols = [c for c in ["years_experience", "remote_ratio"] if c in df.columns]

    X = pd.concat([skills_df, df[profile_cats], df[numeric_cols]], axis=1)
    y = df["job_title"]

    title_counts = y.value_counts()
    valid_titles = title_counts[title_counts >= 5].index
    mask = y.isin(valid_titles)
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    skill_cols = list(skills_df.columns)
    preprocessor = ColumnTransformer([
        ("skills", "passthrough", skill_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), profile_cats),
        ("num", StandardScaler(), numeric_cols),
    ], remainder="drop")

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])
    pipe.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, pipe.predict(X_test))

    return {
        "pipe": pipe, "skill_cols": skill_cols, "profile_cats": profile_cats,
        "numeric_cols": numeric_cols, "X_columns": X.columns, "test_acc": test_acc,
        "mlb_classes": mlb.classes_,
    }


def recommend_classifier(df, bundle, user_skills, experience_level, education_required,
                          employment_type, company_size, years_experience, top_n=5):
    row = {col: 0 for col in bundle["skill_cols"]}
    for skill in user_skills:
        col = f"skill_{skill.strip().lower()}"
        if col in row:
            row[col] = 1
    row["experience_level"] = experience_level
    row["education_required"] = education_required
    row["employment_type"] = employment_type
    row["company_size"] = company_size
    if "years_experience" in bundle["numeric_cols"]:
        row["years_experience"] = years_experience
    if "remote_ratio" in bundle["numeric_cols"]:
        row["remote_ratio"] = df["remote_ratio"].median()

    user_df = pd.DataFrame([row])[bundle["X_columns"]]
    proba = bundle["pipe"].predict_proba(user_df)[0]
    classes = bundle["pipe"].named_steps["model"].classes_
    ranked = sorted(zip(classes, proba), key=lambda x: -x[1])[:top_n]
    return pd.DataFrame(ranked, columns=["Job Title", "Confidence"])


def recommend_similarity(df, user_skills, top_n=5):
    user_set = set(s.strip().lower() for s in user_skills)

    def jaccard(skills_list):
        posting_set = set(skills_list)
        if not posting_set or not user_set:
            return 0.0
        return len(user_set & posting_set) / len(user_set | posting_set)

    scores = df["skills_list"].apply(jaccard)
    top_by_title = (
        df.assign(similarity=scores).groupby("job_title")["similarity"].mean()
        .sort_values(ascending=False).head(top_n)
    )
    return top_by_title.reset_index().rename(
        columns={"job_title": "Job Title", "similarity": "Skill Overlap"}
    )


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("🎯 AI Job Match Finder")
st.caption("Tell us your skills and background — we'll tell you which AI/ML job profiles fit you best.")

df = load_data()
bundle = train_model(df)

left, right = st.columns([1, 1.3], gap="large")

with left:
    st.subheader("Your profile")

    all_skills = sorted(bundle["mlb_classes"])
    user_skills = st.multiselect(
        "Your skills",
        options=all_skills,
        default=[s for s in ["python", "sql", "machine learning"] if s in all_skills],
        help="Start typing to search. Pick as many as apply."
    )

    experience_level = st.selectbox(
        "Experience level", options=sorted(df["experience_level"].dropna().unique())
    )
    education_required = st.selectbox(
        "Education", options=sorted(df["education_required"].dropna().unique())
    )
    employment_type = st.selectbox(
        "Employment type", options=sorted(df["employment_type"].dropna().unique())
    )
    company_size = st.selectbox(
        "Preferred company size", options=sorted(df["company_size"].dropna().unique())
    )
    years_experience = 0
    if "years_experience" in bundle["numeric_cols"]:
        years_experience = st.slider(
            "Years of experience", 0,
            int(df["years_experience"].max()), int(df["years_experience"].median())
        )

    find_button = st.button("Find my job match", type="primary", use_container_width=True)

with right:
    st.subheader("Recommended job profiles")

    if not user_skills:
        st.info("Add at least one skill on the left to get recommendations.")
    elif find_button or user_skills:
        clf_results = recommend_classifier(
            df, bundle, user_skills, experience_level, education_required,
            employment_type, company_size, years_experience
        )
        fig = px.bar(
            clf_results.sort_values("Confidence"),
            x="Confidence", y="Job Title", orientation="h",
            color="Confidence", color_continuous_scale="Blues",
            text=clf_results.sort_values("Confidence")["Confidence"].map(lambda v: f"{v:.0%}"),
        )
        fig.update_layout(showlegend=False, height=320, margin=dict(l=0, r=0, t=10, b=0))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("See alternative view (skill-overlap match)"):
            sim_results = recommend_similarity(df, user_skills)
            st.dataframe(
                sim_results.style.format({"Skill Overlap": "{:.0%}"}),
                use_container_width=True, hide_index=True
            )

st.divider()
st.caption(
    f"Model: Random Forest classifier trained on {len(df):,} job postings "
    f"· Test accuracy: {bundle['test_acc']:.1%} "
    f"(fraction of held-out postings where the top prediction matched the actual job title)"
)
