"""
scripts/generate_items.py

Reads either data/jobs.csv or data/courses.csv (or both),
normalizes columns, builds sentence-transformer embeddings,
and saves:
  - data/items.csv           (merged items table with column 'type'='job'|'course')
  - data/item_embeddings.npy (numpy array with embeddings aligned to items.csv rows)

Usage:
  # jobs only
  python scripts/generate_items.py --type jobs --max 1000

  # courses only
  python scripts/generate_items.py --type courses --max 500

  # both
  python scripts/generate_items.py --type both --max 500
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

# ---- try to import your embedding helper ----
try:
    from model_training.train_model import build_item_vectors
except Exception:
    # fallback: import inline to avoid hard dependency if train_model not present
    def build_item_vectors(texts, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
        return np.array(embeddings)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# candidate column names mapping -> canonical
JOB_COLUMN_CANDIDATES = {
    "title": ["title", "job_title", "position", "name"],
    "company": ["company", "company_name", "employer", "companyname"],
    "text": ["description", "job_description", "text", "full_description", "job_desc"],
    "url": ["url", "link", "job_link", "apply_link"]
}

COURSE_COLUMN_CANDIDATES = {
    "title": ["title", "course_title", "name"],
    "provider": ["provider", "institution", "platform", "university"],
    "text": ["description", "course_description", "overview", "text"],
    "url": ["url", "link", "course_url"]
}


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive search
    lc = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lc:
            return lc[c.lower()]
    return None


def normalize_df(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Map dataset columns to canonical names: title, company/provider, text, url
    """
    out = pd.DataFrame()
    lookup = JOB_COLUMN_CANDIDATES if kind == "job" else COURSE_COLUMN_CANDIDATES
    # find title
    tcol = find_column(df, lookup["title"])
    if tcol is None:
        raise ValueError(f"Could not find a title column for {kind} dataset. Columns: {list(df.columns)}")
    out["title"] = df[tcol].astype(str).fillna("")

    # find company/provider
    cp_col = find_column(df, lookup["company" if kind == "job" else "provider"])
    if cp_col is None:
        # leave blank but create column
        out["company"] = ""
    else:
        out["company"] = df[cp_col].astype(str).fillna("")

    # find text/description
    ttxt = find_column(df, lookup["text"])
    if ttxt is None:
        # fallback: try to combine other textual columns
        possible = [c for c in df.columns if df[c].dtype == object and c not in [tcol, cp_col]]
        if possible:
            out["text"] = df[possible[0]].astype(str).fillna("")
        else:
            out["text"] = ""
    else:
        out["text"] = df[ttxt].astype(str).fillna("")

    # find url
    ucol = find_column(df, lookup["url"])
    if ucol is None:
        out["url"] = ""
    else:
        out["url"] = df[ucol].astype(str).fillna("")

    return out


def prepare_items(jobs_path: Path, courses_path: Path, which: str, max_items: int) -> pd.DataFrame:
    frames = []
    if which in ("jobs", "both"):
        if not jobs_path.exists():
            print(f"[WARN] jobs file not found at {jobs_path}")
        else:
            print(f"Loading jobs from {jobs_path} ...")
            dfj = pd.read_csv(jobs_path)
            dfj = normalize_df(dfj, kind="job")
            dfj["type"] = "job"
            frames.append(dfj.head(max_items))

    if which in ("courses", "both"):
        if not courses_path.exists():
            print(f"[WARN] courses file not found at {courses_path}")
        else:
            print(f"Loading courses from {courses_path} ...")
            dfc = pd.read_csv(courses_path)
            dfc = normalize_df(dfc, kind="course")
            dfc["type"] = "course"
            frames.append(dfc.head(max_items))

    if not frames:
        raise FileNotFoundError("No input files found. Place jobs.csv and/or courses.csv in the data/ folder.")
    combined = pd.concat(frames, ignore_index=True)
    # create a 'text_for_embedding' column
    combined["text_for_embedding"] = (combined["title"].fillna("") + " " + combined["text"].fillna("")).str[:10000]
    return combined


def generate_embeddings_and_save(df: pd.DataFrame, out_csv: Path, out_npy: Path, model_name: str = "all-MiniLM-L6-v2"):
    texts = df["text_for_embedding"].tolist()
    print(f"Generating embeddings for {len(texts)} items using model {model_name} ...")
    embeddings = build_item_vectors(texts)  # expects numpy array (n, dim)
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D array (n_items, dim).")
    df["item_vec"] = embeddings.tolist()
    print(f"Saving {out_csv} and {out_npy} ...")
    df.to_csv(out_csv, index=False)
    np.save(out_npy, embeddings)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(prog="generate_items.py")
    parser.add_argument("--type", type=str, choices=["jobs", "courses", "both"], default="both",
                        help="which dataset to process")
    parser.add_argument("--max", type=int, default=500, help="max items per dataset (jobs/courses) to process")
    parser.add_argument("--jobs-file", type=str, default=str(DATA_DIR / "jobs.csv"))
    parser.add_argument("--courses-file", type=str, default=str(DATA_DIR / "courses.csv"))
    parser.add_argument("--out-csv", type=str, default=str(DATA_DIR / "items.csv"))
    parser.add_argument("--out-npy", type=str, default=str(DATA_DIR / "item_embeddings.npy"))
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    jobs_path = Path(args.jobs_file)
    courses_path = Path(args.courses_file)
    out_csv = Path(args.out_csv)
    out_npy = Path(args.out_npy)

    combined = prepare_items(jobs_path, courses_path, which=args.type, max_items=args.max)
    generate_embeddings_and_save(combined, out_csv, out_npy, model_name=args.model)
    print(f"✅ items saved to {out_csv}")
    print(f"✅ embeddings saved to {out_npy}")


if __name__ == "__main__":
    main()
