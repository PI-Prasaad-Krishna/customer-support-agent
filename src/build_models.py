import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
import os

print("Loading 50k historical dataset...")
df = pd.read_csv('data/apple_support_pairs.csv')

# --- 1. Heuristic Labeling for Training Data ---
print("Applying heuristic labels to create training set...")

def classify_intent(text):
    text = str(text).lower()
    if any(w in text for w in ['battery', 'drain', 'slow', 'lag', 'crash', 'freeze']):
        return 'software_performance'
    if any(w in text for w in ['bug', 'glitch', 'error', 'keyboard', 'ui', 'update']):
        return 'software_bug'
    if any(w in text for w in ['crack', 'screen', 'break', 'broken', 'button', 'water', 'shatter']):
        return 'hardware_repair'
    if any(w in text for w in ['payment', 'decline', 'app store', 'icloud', 'password', 'charge', 'refund', 'subscription']):
        return 'account_billing'
    if any(w in text for w in ['down', 'working', 'server', 'connect']):
        return 'service_outage'
    return 'general_inquiry'

def decide_escalation(text, intent):
    text = str(text).lower()
    angry_words = ['fuck', 'shit', 'terrible', 'moron', 'ridiculous', 'hate', 'pissed']
    if intent in ['account_billing', 'hardware_repair']:
        return 'escalated'
    if any(w in text for w in angry_words):
        return 'escalated'
    return 'auto_handled'

df['intent'] = df['text_inbound'].apply(classify_intent)
df['escalate_decision'] = df.apply(lambda row: decide_escalation(row['text_inbound'], row['intent']), axis=1)

# Drop rows that are in the golden set to prevent data leakage!
golden_df = pd.read_csv('data/golden_evaluation_set.csv')
golden_ids = golden_df['tweet_id_inbound'].tolist()
train_df = df[~df['tweet_id_inbound'].isin(golden_ids)].copy()

print(f"Training on {len(train_df)} examples...")

# --- 2. Train ML Models ---
print("Training Intent Classifier (TF-IDF + Logistic Regression)...")
intent_model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])
intent_model.fit(train_df['text_inbound'], train_df['intent'])

print("Training Escalation Classifier (TF-IDF + Logistic Regression)...")
escalate_model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])
escalate_model.fit(train_df['text_inbound'], train_df['escalate_decision'])

os.makedirs('models', exist_ok=True)
joblib.dump(intent_model, 'models/intent_model.pkl')
joblib.dump(escalate_model, 'models/escalate_model.pkl')

# --- 3. Build RAG Index (Nearest Neighbors on TF-IDF) ---
print("Building RAG Index...")
# We use TF-IDF for RAG retrieval as well. It's incredibly fast and requires no GPU/downloads.
rag_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_rag = rag_vectorizer.fit_transform(train_df['text_inbound'])

rag_index = NearestNeighbors(n_neighbors=3, metric='cosine')
rag_index.fit(X_rag)

joblib.dump(rag_vectorizer, 'models/rag_vectorizer.pkl')
joblib.dump(rag_index, 'models/rag_index.pkl')

# Save the reference dataframe so we can retrieve the actual AppleSupport replies
train_df[['tweet_id_inbound', 'text_inbound', 'text_outbound']].to_csv('models/rag_corpus.csv', index=False)

print("All models and RAG index built and saved successfully!")
