"""
Recommendation engine for PathFinder AI
Loads precomputed embeddings and returns top matches for Jobs or Courses.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from model_training.train_model import build_user_vector, Recommender


DATA_DIR = Path("data")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def load_items(data_type="job"):
    """Load item dataframe and embeddings for either jobs or courses."""
    if data_type == "job":
        csv_path = DATA_DIR / "jobs.csv"
        emb_path = DATA_DIR / "jobs_embeddings.npy"
    else:
        csv_path = DATA_DIR / "courses.csv"
        emb_path = DATA_DIR / "courses_embeddings.npy"

    if not csv_path.exists() or not emb_path.exists():
        raise FileNotFoundError(f"Missing required data: {csv_path.name} or {emb_path.name}")

    df = pd.read_csv(csv_path)
    embeddings = np.load(emb_path, allow_pickle=True)

    # Handle case where embeddings are stored as objects (list of floats)
    if embeddings.dtype == "object":
        embeddings = np.vstack(embeddings)

    return df, embeddings


def recommend_for_user(
    skills_list,
    interests,
    experience,
    gender,
    age,
    type_choice="job",
    top_k=10
):
    """Generate top recommendations for a given user profile."""
    # Build user vector
    user_vec = build_user_vector(skills_list, interests, experience, gender, age)

    # Load items (jobs or courses)
    df, embeddings = load_items(type_choice)

    # Initialize recommender
    rec = Recommender(df, embeddings)

    # Get top results
    top_items = rec.recommend(user_vec, top_k=top_k)
    return top_items.reset_index(drop=True)
