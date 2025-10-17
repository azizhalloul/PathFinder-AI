"""
Model training for PathFinder AI (FairHire AI)
Builds embeddings for jobs and courses using SentenceTransformer.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Load a universal embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class Recommender:
    """A lightweight class that handles item embeddings and user-item similarity."""
    def __init__(self, items_df, embeddings):
        self.items_df = items_df
        self.embeddings = embeddings

    def recommend(self, user_vec, top_k=10):
        """Return top K items based on cosine similarity."""
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity([user_vec], self.embeddings)[0]
        top_idx = np.argsort(-sims)[:top_k]
        return self.items_df.iloc[top_idx].copy()


def prepare_items(csv_path, text_col, save_prefix):
    """Build item embeddings for a dataset."""
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in {csv_path.name}")

    tqdm.pandas()
    texts = df[text_col].fillna("").astype(str).tolist()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    df[f"{save_prefix}_embedding"] = embeddings.tolist()
    df.to_csv(DATA_DIR / f"{save_prefix}.csv", index=False)
    np.save(DATA_DIR / f"{save_prefix}_embeddings.npy", embeddings)
    print(f"✅ {save_prefix} embeddings saved successfully.")


def build_user_vector(skills, interests, experience, gender, age):
    """Generate a semantic vector for a user's profile."""
    text = f"Skills: {', '.join(skills)}. Interests: {', '.join(interests)}. Experience: {experience}. Gender: {gender}. Age: {age}."
    return model.encode([text])[0]


if __name__ == "__main__":
    print("⏳ Preparing job items...")
    prepare_items(DATA_DIR / "jobs.csv", text_col="description", save_prefix="jobs")

    print("⏳ Preparing course items...")
    prepare_items(DATA_DIR / "courses.csv", text_col="short intro", save_prefix="courses")

    print("✅ All embeddings generated.")
