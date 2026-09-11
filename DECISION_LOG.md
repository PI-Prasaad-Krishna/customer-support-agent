# Apple Support AI Pipeline - Decision Log

This document tracks the micro-decisions made during the architecture and implementation of the AI support pipeline.

1. **Brand Selection**: Chose `@AppleSupport` because technical support naturally provides clear, distinct intents (software vs. hardware vs. billing) compared to generic retail brands.
2. **Dataset Scoping**: Limited the dataset extraction to 50,000 inbound-outbound tweet pairs. This is large enough to train robust ML models and populate a vector database, but small enough to iterate rapidly on a local machine.
3. **Target Variable Definition**: Consolidated intents into 6 broad categories (e.g., `software_bug`, `hardware_repair`) to ensure classes had enough representation for ML training, avoiding the long-tail problem of overly specific intents.
4. **Escalation Logic**: Defined "Escalation" not just by angry sentiment, but inherently tied to certain intents (e.g., billing, hardware repair) which *always* require human verification (account details/warranty checks).
5. **Hybrid Architecture (The "Pragmatic" Pivot)**: Originally planned to use Gemini API for classification, but due to strict 20 request/day limits on the free tier, pivoted to local ML models. This demonstrates real-world cost-saving architecture.
6. **Golden Set Labeling (Heuristic)**: Used an expert heuristic algorithm to pseudo-label the 250-tweet Golden Evaluation set to bypass API limits and save hours of manual data entry.
7. **ML Classifier Choice**: Selected `LogisticRegression` over Random Forests or Neural Networks. Logistic regression trains in seconds on 50k text embeddings and provides highly interpretable coefficients.
8. **Text Embedding Choice**: Used TF-IDF (`TfidfVectorizer`) instead of heavy SentenceTransformers (like BERT). TF-IDF requires no model downloads and is blazing fast for simple keyword-driven intents like tech support.
9. **RAG Vector Database**: Chose `FAISS` with NearestNeighbors over a complex database like Pinecone. FAISS runs locally in-memory, requiring no infrastructure setup for interview reviewers.
10. **LLM Drafter Prompting**: Provided the LLM with the *Predicted Intent*, *Escalation Decision*, and *2 Historical RAG Contexts*. This grounds the model strictly in Apple's historical brand voice and prevents hallucination.
11. **Evaluation Script Limitations**: Hardcoded the evaluation script to only test the LLM Drafter on a `max_calls=5` parameter. This prevents accidentally exhausting the free-tier daily quota while still proving the pipeline works.
12. **CLI Delivery**: Chose a rich, colorful Command Line Interface (`src/demo.py`) over a bloated Streamlit app to keep dependencies lightweight and ensure the focus remains on the pipeline's backend engineering.
