"""
Streamlit UI for PathFinder AI — Ethical Job & Course Recommendations
This file is compatible with recommend_for_user_profile(...) which returns
a DataFrame or list-of-dicts with columns: ['title','company'/'platform','text','url','score' (optional)].
"""

import sys
import os
from pathlib import Path

# ensure project root is on sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from recommender_engine.recommend import recommend_for_user

st.set_page_config(page_title="PathFinder AI", layout="wide")
st.title("PathFinder AI — Job & Course Recommendations")
st.markdown(
    "Enter a user profile (skills, experience, gender, age). The AI finds top matches and shows results."
)

# ----------------------------
# Sidebar — User Inputs
# ----------------------------
with st.sidebar:
    st.header("Profile")
    skills = st.text_input(
        "Skills (comma-separated)",
        value="python, machine learning, data analysis"
    )
    interests = st.text_input(
        "Interests (comma-separated)",
        value="machine learning, data engineering"
    )
    experience = st.selectbox("Experience level", ["junior", "mid", "senior"])
    gender = st.selectbox("Gender", ["female", "male"])
    age = st.slider("Age", 18, 65, 30)
    topk = st.slider("Top K results", 1, 20, 10)

    # Choose Jobs or Courses
    rec_type = st.radio("Recommendation type", ["Jobs", "Courses"])

# Centered "Get Recommendations" button
st.write("")
st.write("")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run = st.button("Get Recommendations", type="primary")

def clean_html(raw_html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    if not isinstance(raw_html, str):
        return ""
    txt = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)
    return " ".join(txt.split())

# ----------------------------
# Main logic
# ----------------------------
if run:
    # Construct profile inputs (lists)
    skills_list = [s.strip() for s in skills.split(",") if s.strip()]
    interests_list = [s.strip() for s in interests.split(",") if s.strip()]

    profile = {
        "skills": skills_list,
        "interests": interests_list,
        "experience": experience,
        "gender": gender,
        "age": age
    }

    try:
        # Call recommender — ensure parameter names match your recommend function
        results = recommend_for_user(
            skills_list=skills_list,
            interests=interests_list,
            experience=experience,
            gender=gender,
            age=age,
            type_choice="job" if rec_type == "Jobs" else "course",
            top_k=topk
        )

        # Normalize to DataFrame if list-of-dicts returned
        if isinstance(results, list):
            results_df = pd.DataFrame(results)
        elif isinstance(results, pd.DataFrame):
            results_df = results.reset_index(drop=True)
        else:
            # try to convert
            results_df = pd.DataFrame(results)

        if results_df.empty:
            st.warning("No recommendations found for this profile.")
        else:
            st.subheader(f"Top {len(results_df)} {rec_type} Recommendations")

            # Loop with clean numbering
            for idx, row in results_df.reset_index(drop=True).iterrows():
                rank = idx + 1
                title = str(row.get("title", "Untitled")).strip()
                # clean title quirks
                if ". " in title and title.split(". ")[0].isdigit():
                    title = title.split(". ", 1)[1]
                title = title.split("—")[0].strip()

                company = row.get("company") or row.get("platform") or ""
                raw_text = row.get("text") or row.get("description") or ""
                text = clean_html(raw_text)
                snippet = text[:400] + ("..." if len(text) > 400 else "")

                url = row.get("url") or row.get("link") or "#"
                score = row.get("score", None)

                # Card-like output
                st.markdown(f"**{rank}. {title}**  \n*{company}*")
                st.write(snippet)
                if score is not None:
                    try:
                        st.caption(f"Score: {float(score):.3f}")
                    except Exception:
                        pass
                st.markdown(f"[🔗 View details]({url})")
                st.markdown("---")

    except FileNotFoundError as fnf:
        st.error(f"Data file not found: {fnf}")
        st.info("Make sure `items.csv` and `item_embeddings.npy` are generated in the data/ folder.")
    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    "Notes: Precompute embeddings (`items.csv` + `item_embeddings.npy`) for faster demo performance. "
    "The AI uses semantic matching between your profile and items."
)
