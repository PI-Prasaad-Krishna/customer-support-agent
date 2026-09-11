# 🍎 Apple Support AI Pipeline

An end-to-end Machine Learning & GenAI pipeline that classifies, routes, and automatically drafts replies for inbound customer support tweets directed at `@AppleSupport`.

This project was built to demonstrate full-stack AI Engineering—from messy data extraction and heuristic pseudo-labeling, to training lightweight ML classification models, and finally implementing a FAISS-backed RAG system for LLM drafting.

## 🚀 The Architecture

Due to strict rate limits on the free-tier Gemini API (20 requests/day), this project embraces the **"Pragmatic Engineer"** approach. Instead of relying on a massive LLM for every single task, it uses a hybrid system that is blazing fast, cost-effective, and highly scalable:

1. **Intent & Escalation Classification**: Handled entirely locally by a **TF-IDF + Logistic Regression** model. It was trained on 50,000 tweets that were pseudo-labeled using a custom expert heuristic algorithm.
2. **Context Retrieval (RAG)**: Handled locally by a **FAISS (Nearest Neighbors)** vector index. It instantly searches 50,000 historical Apple Support conversations to find how human agents resolved similar issues.
3. **Response Drafting**: Only the final response generation calls the **Gemini 2.5 Flash API**. The LLM is provided the predicted intent, escalation decision, and the top 2 historical context pairs retrieved by FAISS to draft a perfectly grounded, brand-safe reply.

## 🛠️ Setup & Installation

### 1. Clone the repository and install dependencies
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Set up your API Key
Rename `.env.example` to `.env` and paste your Gemini API key:
```env
GEMINI_API_KEY="AIzaSyYourKeyHere..."
```

### 3. Build the Models Locally
We don't track large model weights or vector indices in Git. You must build them yourself.
This script will train the ML classifiers and build the RAG index from the local dataset.
```bash
python src/build_models.py
```

## 🧪 Running the Interactive Demo (Phase 5)

Want to see the pipeline in action? Run the interactive CLI app! 
It allows you to type in mock customer tweets and watch the pipeline classify the intent, retrieve historical context, and draft a personalized reply in real-time.

```bash
python src/demo.py
```

## 📊 Evaluation (Phase 4)

To run the evaluation harness against the hand-crafted Golden Dataset:
```bash
python src/evaluate_pipelines.py
```
*Note: The evaluation script limits LLM drafting to the first 5 examples to prevent exhausting free-tier rate limits. The ML classification evaluation runs on the full dataset.*

---
*Built with Python, Scikit-Learn, FAISS, Pandas, and Google GenAI.*
