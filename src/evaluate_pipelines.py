import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.metrics import accuracy_score, classification_report
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key and api_key != "YOUR_API_KEY_HERE" else None

print("Loading Golden Set...")
golden_df = pd.read_csv('data/golden_evaluation_set.csv')
X_test = golden_df['text_inbound']
y_intent_true = golden_df['intent']
y_escalate_true = golden_df['escalate_decision']

print("\n--- 1. Evaluating TRIVIAL Baseline ---")
# Predicts majority class ('general_inquiry') and 'auto_handled'
y_intent_pred_trivial = ['general_inquiry'] * len(golden_df)
y_escalate_pred_trivial = ['auto_handled'] * len(golden_df)

print("Trivial Intent Accuracy:", accuracy_score(y_intent_true, y_intent_pred_trivial))
print("Trivial Escalate Accuracy:", accuracy_score(y_escalate_true, y_escalate_pred_trivial))


print("\n--- 2. Evaluating SIMPLE Baseline (Heuristics) ---")
# The heuristics logic we wrote earlier
def classify_intent_heuristic(text):
    text = str(text).lower()
    if any(w in text for w in ['battery', 'drain', 'slow', 'lag', 'crash', 'freeze']): return 'software_performance'
    if any(w in text for w in ['bug', 'glitch', 'error', 'keyboard', 'ui', 'update']): return 'software_bug'
    if any(w in text for w in ['crack', 'screen', 'break', 'broken', 'button', 'water', 'shatter']): return 'hardware_repair'
    if any(w in text for w in ['payment', 'decline', 'app store', 'icloud', 'password', 'charge', 'refund', 'subscription']): return 'account_billing'
    if any(w in text for w in ['down', 'working', 'server', 'connect']): return 'service_outage'
    return 'general_inquiry'

def decide_escalation_heuristic(text, intent):
    text = str(text).lower()
    angry_words = ['fuck', 'shit', 'terrible', 'moron', 'ridiculous', 'hate', 'pissed']
    if intent in ['account_billing', 'hardware_repair'] or any(w in text for w in angry_words):
        return 'escalated'
    return 'auto_handled'

y_intent_pred_simple = X_test.apply(classify_intent_heuristic)
y_escalate_pred_simple = [decide_escalation_heuristic(t, i) for t, i in zip(X_test, y_intent_pred_simple)]

print("Simple Intent Accuracy:", accuracy_score(y_intent_true, y_intent_pred_simple))
print("Simple Escalate Accuracy:", accuracy_score(y_escalate_true, y_escalate_pred_simple))


print("\n--- 3. Evaluating MAIN Pipeline (ML + FAISS RAG + LLM Drafter) ---")
intent_model = joblib.load('models/intent_model.pkl')
escalate_model = joblib.load('models/escalate_model.pkl')
rag_vectorizer = joblib.load('models/rag_vectorizer.pkl')
rag_index = joblib.load('models/rag_index.pkl')
rag_corpus = pd.read_csv('models/rag_corpus.csv')

y_intent_pred_main = intent_model.predict(X_test)
y_escalate_pred_main = escalate_model.predict(X_test)

print("Main Pipeline Intent Accuracy:", accuracy_score(y_intent_true, y_intent_pred_main))
print("Main Pipeline Escalate Accuracy:", accuracy_score(y_escalate_true, y_escalate_pred_main))

print("\n--- Testing RAG LLM Drafter (Max 5 API calls due to limit) ---")
if not client:
    print("API Key not set. Skipping LLM drafting.")
else:
    sample_df = golden_df.head(5).copy()
    
    for i, row in sample_df.iterrows():
        tweet = row['text_inbound']
        intent = y_intent_pred_main[i]
        
        # 1. RAG Retrieval
        vec = rag_vectorizer.transform([tweet])
        distances, indices = rag_index.kneighbors(vec)
        
        context_str = ""
        for idx in indices[0]:
            hist_inbound = rag_corpus.iloc[idx]['text_inbound']
            hist_outbound = rag_corpus.iloc[idx]['text_outbound']
            context_str += f"- Customer: {hist_inbound}\n  Apple: {hist_outbound}\n\n"
            
        # 2. LLM Draft
        prompt = f"""
        You are @AppleSupport. A customer tweeted: "{tweet}"
        
        Our ML pipeline classified the intent as: {intent}.
        
        Here is how we successfully responded to similar historical tweets:
        {context_str}
        
        Draft a polite, helpful reply to the customer in under 280 characters.
        If it requires escalation, kindly ask them to DM us.
        """
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            print(f"\n[Customer Tweet]: {tweet}")
            print(f"[RAG + LLM Draft]: {response.text.strip()}")
            time.sleep(2)
        except Exception as e:
            print(f"API Error: {e}")
            break

print("\nEvaluation Complete!")
