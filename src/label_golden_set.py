import pandas as pd
import numpy as np

print("Loading pairs...")
df = pd.read_csv('data/apple_support_pairs.csv')

print("Sampling 250 for Golden Set...")
golden_df = df.sample(n=250, random_state=42).copy()

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
        return 'escalated', f'Requires sensitive {intent} handling.'
    if any(w in text for w in angry_words):
        return 'escalated', 'Customer is highly frustrated.'
    return 'auto_handled', 'Standard troubleshooting should apply.'

print("Labeling using expert heuristics...")
golden_df['intent'] = golden_df['text_inbound'].apply(classify_intent)
golden_df['escalate_decision'] = golden_df.apply(lambda row: decide_escalation(row['text_inbound'], row['intent'])[0], axis=1)
golden_df['escalate_reason'] = golden_df.apply(lambda row: decide_escalation(row['text_inbound'], row['intent'])[1], axis=1)

golden_df.to_csv('data/golden_evaluation_set.csv', index=False)
print("Golden set pseudo-labeled and saved to data/golden_evaluation_set.csv!")
