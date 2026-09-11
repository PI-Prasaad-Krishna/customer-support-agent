import pandas as pd
import joblib
import os
import sys
import warnings
from google import genai
from dotenv import load_dotenv

# Suppress Google SDK warnings
warnings.filterwarnings('ignore')

# ANSI escape codes for colorful CLI
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

print(f"{Colors.CYAN}{Colors.BOLD}Initializing Apple Support AI Pipeline...{Colors.ENDC}")

try:
    intent_model = joblib.load('models/intent_model.pkl')
    escalate_model = joblib.load('models/escalate_model.pkl')
    rag_vectorizer = joblib.load('models/rag_vectorizer.pkl')
    rag_index = joblib.load('models/rag_index.pkl')
    rag_corpus = pd.read_csv('models/rag_corpus.csv')
except FileNotFoundError:
    print(f"{Colors.FAIL}Models not found! Please run `python src/build_models.py` first.{Colors.ENDC}")
    sys.exit(1)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key and api_key != "YOUR_API_KEY_HERE" else None

print(f"{Colors.GREEN}Systems Ready!{Colors.ENDC}\n")

def process_tweet(tweet):
    print(f"\n{Colors.HEADER}--- Processing Inbound Tweet ---{Colors.ENDC}")
    
    # 1. Classification
    intent = intent_model.predict([tweet])[0]
    escalation = escalate_model.predict([tweet])[0]
    
    print(f"{Colors.BOLD}Predicted Intent:{Colors.ENDC} {Colors.CYAN}{intent}{Colors.ENDC}")
    
    if escalation == 'escalated':
        print(f"{Colors.BOLD}Escalation Decision:{Colors.ENDC} {Colors.FAIL}ESCALATE TO HUMAN{Colors.ENDC}")
    else:
        print(f"{Colors.BOLD}Escalation Decision:{Colors.ENDC} {Colors.GREEN}AUTO-HANDLE{Colors.ENDC}")

    # 2. RAG Retrieval
    vec = rag_vectorizer.transform([tweet])
    distances, indices = rag_index.kneighbors(vec)
    
    print(f"\n{Colors.HEADER}--- RAG Retrieval (Top 2 Contexts) ---{Colors.ENDC}")
    context_str = ""
    for i, idx in enumerate(indices[0][:2]):
        hist_inbound = rag_corpus.iloc[idx]['text_inbound']
        hist_outbound = rag_corpus.iloc[idx]['text_outbound']
        print(f"{Colors.BLUE}[Match {i+1}]{Colors.ENDC} Customer: {hist_inbound}")
        print(f"          Apple: {hist_outbound}")
        context_str += f"- Customer: {hist_inbound}\n  Apple: {hist_outbound}\n\n"

    # 3. LLM Drafting
    print(f"\n{Colors.HEADER}--- LLM Drafter ---{Colors.ENDC}")
    if not client:
        print(f"{Colors.WARNING}GEMINI_API_KEY not found in .env. Skipping drafting.{Colors.ENDC}")
        return

    prompt = f"""
    You are @AppleSupport. A customer tweeted: "{tweet}"
    
    Our ML pipeline classified the intent as: {intent}.
    The escalation decision is: {escalation}.
    
    Here is how we successfully responded to similar historical tweets:
    {context_str}
    
    Draft a polite, helpful reply to the customer in under 280 characters.
    If the decision is 'escalated', kindly ask them to DM us with details.
    Make sure the tone is strictly Apple Support.
    """
    
    try:
        print(f"{Colors.CYAN}Drafting reply via Gemini 2.5 Flash...{Colors.ENDC}")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        print(f"{Colors.GREEN}{Colors.BOLD}Final Draft:{Colors.ENDC} {response.text.strip()}")
    except Exception as e:
        print(f"{Colors.FAIL}API Error during drafting: {e}{Colors.ENDC}")

# Interactive Loop
print("Type a simulated customer tweet below (or type 'exit' to quit).")
while True:
    try:
        user_input = input(f"\n{Colors.BOLD}Customer Tweet > {Colors.ENDC}")
        if user_input.strip().lower() in ['exit', 'quit']:
            break
        if not user_input.strip():
            continue
        process_tweet(user_input)
    except KeyboardInterrupt:
        break

print(f"\n{Colors.GREEN}Shutting down. Thanks for using the Apple Support Pipeline!{Colors.ENDC}")
