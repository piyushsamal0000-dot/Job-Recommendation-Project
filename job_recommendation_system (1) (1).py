# -*- coding: utf-8 -*-
"""
Global AI Job Market 2025 — Job Recommendation System
Internship minor project (v3, pivoted from salary regression)

Given a candidate's skills + profile (experience level, education,
employment type preference, years of experience), recommend the
job title(s) they're best suited for.

Two approaches, both included:
1. Classifier-based: RandomForestClassifier trained on historical
   postings, predict_proba gives a ranked top-N list of job titles.
   This is the primary approach — it learns which skill combos map
   to which roles across the whole dataset.
2. Skill-overlap similarity: Jaccard similarity between the user's
   skill set and each posting's skill set, matched job titles ranked
   by overlap. Useful as an interpretable fallback / sanity check,
   and works even for a skill combo the classifier has never seen.

NOTE: adjust SKILLS_COL / SKILLS_SEP below if your CSV's skills
column is named or formatted differently — the script prints the
detected columns on load so you can check.
"""

import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------
# 1. LOAD + INSPECT
# ---------------------------------------------------------------
df = pd.read_csv("ai_job_dataset1.csv")
print("Columns:", list(df.columns))

SKILLS_COL = "required_skills"   # adjust if your column is named differently
SKILLS_SEP = ","                 # adjust if skills are semicolon/pipe separated

assert SKILLS_COL in df.columns, (
    f"'{SKILLS_COL}' not found. Available columns: {list(df.columns)}. "
    f"Update SKILLS_COL to match your CSV."
)

df[SKILLS_COL] = df[SKILLS_COL].fillna("")
df["skills_list"] = df[SKILLS_COL].apply(
    lambda s: [skill.strip().lower() for skill in s.split(SKILLS_SEP) if skill.strip()]
)

print("\nMost common skills in the dataset:")
all_skills = [s for skills in df["skills_list"] for s in skills]
print(Counter(all_skills).most_common(15))

print("\nJob title distribution (top 15):")
print(df["job_title"].value_counts().head(15))

# ---------------------------------------------------------------
# 2. FEATURE ENCODING
# ---------------------------------------------------------------
# Skills -> multi-hot binary matrix (one column per skill)
mlb = MultiLabelBinarizer()
skills_encoded = mlb.fit_transform(df["skills_list"])
skills_df = pd.DataFrame(skills_encoded, columns=[f"skill_{s}" for s in mlb.classes_], index=df.index)

profile_cats = ["experience_level", "education_required", "employment_type", "company_size"]
profile_cats = [c for c in profile_cats if c in df.columns]

numeric_cols = [c for c in ["years_experience", "remote_ratio"] if c in df.columns]

X = pd.concat([skills_df, df[profile_cats], df[numeric_cols]], axis=1)
y = df["job_title"]

# Drop job titles with too few examples to be learnable / stratifiable
title_counts = y.value_counts()
valid_titles = title_counts[title_counts >= 5].index
mask = y.isin(valid_titles)
X, y = X[mask], y[mask]

print(f"\nUsing {len(valid_titles)} job titles with >=5 postings each "
      f"({mask.sum()} of {len(mask)} rows retained)")

# ---------------------------------------------------------------
# 3. TRAIN/TEST SPLIT + PIPELINE
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

skill_cols = list(skills_df.columns)

preprocessor = ColumnTransformer([
    ("skills", "passthrough", skill_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), profile_cats),
    ("num", StandardScaler(), numeric_cols),
], remainder="drop")

clf_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced",
        random_state=42, n_jobs=-1
    )),
])

clf_pipe.fit(X_train, y_train)

# ---------------------------------------------------------------
# 4. EVALUATE — accuracy AND top-3 accuracy (fairer for a recommender:
#    getting the right job in your top 3 suggestions still counts)
# ---------------------------------------------------------------
y_pred = clf_pipe.predict(X_test)
print("\nTop-1 accuracy:", accuracy_score(y_test, y_pred))

proba = clf_pipe.predict_proba(X_test)
classes = clf_pipe.named_steps["model"].classes_
top3_preds = np.argsort(-proba, axis=1)[:, :3]
top3_hit = [
    y_test.iloc[i] in classes[top3_preds[i]]
    for i in range(len(y_test))
]
print("Top-3 accuracy:", np.mean(top3_hit))

print("\nClassification report (top-1):")
print(classification_report(y_test, y_pred, zero_division=0))

# ---------------------------------------------------------------
# 5. RECOMMENDATION FUNCTION — classifier-based, ranked top-N
# ---------------------------------------------------------------
def recommend_jobs_classifier(user_skills, experience_level=None, education_required=None,
                               employment_type=None, company_size=None,
                               years_experience=None, remote_ratio=None, top_n=5):
    """
    user_skills: list of strings, e.g. ["python", "sql", "machine learning"]
    other args: match the dataset's category values (leave None to use the
                most common category as a neutral default)
    """
    row = {col: 0 for col in skill_cols}
    for skill in user_skills:
        col = f"skill_{skill.strip().lower()}"
        if col in row:
            row[col] = 1
        else:
            print(f"Note: '{skill}' not seen in training data, ignored.")

    row["experience_level"] = experience_level or df["experience_level"].mode()[0]
    row["education_required"] = education_required or df["education_required"].mode()[0]
    row["employment_type"] = employment_type or df["employment_type"].mode()[0]
    row["company_size"] = company_size or df["company_size"].mode()[0]
    if "years_experience" in numeric_cols:
        row["years_experience"] = years_experience if years_experience is not None else df["years_experience"].median()
    if "remote_ratio" in numeric_cols:
        row["remote_ratio"] = remote_ratio if remote_ratio is not None else df["remote_ratio"].median()

    user_df = pd.DataFrame([row])[X.columns]
    proba = clf_pipe.predict_proba(user_df)[0]
    ranked = sorted(zip(classes, proba), key=lambda x: -x[1])[:top_n]
    return pd.DataFrame(ranked, columns=["job_title", "confidence"])


# ---------------------------------------------------------------
# 6. RECOMMENDATION FUNCTION — skill-overlap similarity (interpretable
#    fallback; doesn't need retraining, works for unseen skill combos)
# ---------------------------------------------------------------
def recommend_jobs_similarity(user_skills, top_n=5):
    user_set = set(s.strip().lower() for s in user_skills)

    def jaccard(skills_list):
        posting_set = set(skills_list)
        if not posting_set or not user_set:
            return 0.0
        return len(user_set & posting_set) / len(user_set | posting_set)

    scores = df["skills_list"].apply(jaccard)
    top_matches = df.assign(similarity=scores).sort_values("similarity", ascending=False)
    top_by_title = (
        top_matches.groupby("job_title")["similarity"].mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
    return top_by_title.reset_index().rename(columns={"similarity": "avg_skill_overlap"})


# ---------------------------------------------------------------
# 7. DEMO
# ---------------------------------------------------------------
if __name__ == "__main__":
    example_skills = ["python", "machine learning", "sql"]

    print("\n=== Classifier-based recommendation ===")
    print(recommend_jobs_classifier(example_skills, experience_level="EN", top_n=5))

    print("\n=== Skill-overlap recommendation ===")
    print(recommend_jobs_similarity(example_skills, top_n=5))
