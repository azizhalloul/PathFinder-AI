from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("data")

def build_item_vectors(texts, model_name="all-MiniLM-L6-v2"):
    """Compute sentence-transformer embeddings for a list of texts."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    return np.array(embeddings)



def prepare_items(csv_path, save_prefix="items", model_name="all-MiniLM-L6-v2"):
    """
    Prepare items for embeddings.
    csv_path: Path to CSV file (jobs or courses)
    save_prefix: prefix for saved CSV + embeddings
    """
    df = pd.read_csv(csv_path)

    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Detect which column to use for text
    if "text" in df.columns:
        text_col = "text"
    elif "description" in df.columns:
        text_col = "description"
    elif "short intro" in df.columns:
        text_col = "short intro"
    else:
        raise ValueError(f"No suitable text column found in {csv_path.name}")

    texts = df[text_col].fillna("").tolist()

    print(f"⏳ Generating embeddings for {len(df)} items...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings)

    # Save CSV and embeddings
    df['item_vec'] = embeddings.tolist()
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    df.to_csv(data_dir / f"{save_prefix}.csv", index=False)
    np.save(data_dir / f"{save_prefix}_embeddings.npy", embeddings)
    print(f"✅ {save_prefix}.csv and {save_prefix}_embeddings.npy saved.")


