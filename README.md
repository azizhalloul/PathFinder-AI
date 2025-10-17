# 🧭 PathFinder AI — Ethical Job & Course Recommender

> **AI-powered, fairness-aware recommendation system** that helps users discover the best jobs and online courses based on their **skills, experience, and interests** , designed to ensure **ethical and inclusive recommendations**.

---

## 🚀 Overview

PathFinder AI leverages **Natural Language Processing (NLP)** and **Deep Learning** to semantically match user profiles with thousands of real-world **job** and **course listings**.  
The system goes beyond traditional recommenders by integrating **fairness metrics** that monitor and reduce gender bias during training.

It features a full machine-learning pipeline — from data collection and model training to an interactive **Streamlit web app** for real-time recommendations.

---

## ✨ Key Features

- 🧠 **Dual-Domain Recommender** — Suggests both *Jobs* and *Courses* from real datasets.  
- 🤝 **Fairness-Aware Training** — Includes *demographic parity metrics* to promote ethical AI decisions.  
- 💬 **Semantic Matching** — Uses *Sentence Transformers* for rich text embeddings and profile understanding.  
- ⚡ **Fast & Scalable** — Built with efficient PyTorch encoders and batched vector operations.  
- 🖥️ **Interactive Streamlit App** — Modern, intuitive UI for end users to test the model instantly.  
- 🔍 **Real-World Data** — Based on curated job and course listings from Kaggle and public sources.

---

## 🧩 Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | Streamlit |
| **Backend / ML** | PyTorch, NumPy, Pandas |
| **NLP Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Fairness Metrics** | Custom implementation (Demographic Parity Gap) |
| **Data Source** | Kaggle Job Listings + Online Course Datasets |
| **Version Control** | Git + GitHub |

---
### Demo



[![Watch the demo](https://drive.google.com/file/d/1JSPrPdEjsvBow0pSFXe29Q5LFDKAfAAZ/view?usp=drive_link)](https://drive.google.com/file/d/1JSPrPdEjsvBow0pSFXe29Q5LFDKAfAAZ/view?usp=drive_link)



## 🏗️ Project Structure

```text
AI_Fair_RecommenderSystem/
├── app_user_interface/  
│   └── streamlit_app.py         # Streamlit web interface  
│  
├── recommender_engine/  
│   └── recommend.py             # Inference logic for job/course recommendations  
│  
├── model_training/  
│   ├── train_model.py           # Training pipeline for recommender  
│   ├── embedding_utils.py       # Builds embeddings for items  
│   └── fairness_metrics.py      # Fairness evaluation functions  
│  
├── data/  
│   ├── jobs.csv                 # Real-world job dataset  
│   ├── courses.csv              # Real-world courses dataset  
│   ├── jobs_items.csv           # Processed jobs  
│   ├── courses_items.csv        # Processed courses  
│   └── item_embeddings.npy      # Precomputed embeddings  
│  
└── README.md 

