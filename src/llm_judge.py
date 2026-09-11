import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# The 5 tweets and drafts generated earlier
samples = [
    {
        "tweet": "@AppleSupport @115948 @118101 As I have had no progress over the phone at all. Could somebody please tell me why I am being charged £9.99 per month for a FREE Apple Music trial that that I subscribed to but doesn’t expire until the 26th January 2018?!",
        "draft": "Hi there! We want to help clear up this charge for your Apple Music trial. To investigate this account-specific concern, please DM us with details. Our team will gladly assist you further. Thanks!"
    },
    {
        "tweet": "@AppleSupport is this a scam or really Apple? See pic below that came up on my phone https://t.co/fT7cCDTgWF",
        "draft": "@CustomerHandle We can help you check that out. Here's how to identify legitimate communications from Apple and report suspicious messages: https://t.co/LNMCdqt6fD"
    },
    {
        "tweet": "i’m soo mad my phone is not working right i think it has s virus @AppleSupport",
        "draft": "@customer Oh no! We're sorry to hear your iPhone isn't working as expected. Please tell us more about what's happening. We're here to help you get this sorted out!"
    },
    {
        "tweet": "I'm about to throw my damn phone at a wall. @ATT and @115858 you guys suck",
        "draft": "We're sorry to hear this is happening and understand your frustration. We'd like to help with your iPhone. Can you tell us more about what's going on?"
    },
    {
        "tweet": "@115858 @AppleSupport have had to erase new IPhone X and begin setup 3 times now due to “unable to verify update”. Really crappy UX",
        "draft": "@115858 We're sorry to hear about the trouble with your new iPhone X setup! The \"unable to verify update\" issue can often be resolved with these steps for updating over Wi-Fi: https://t.co/Ui1yTnOCUz If it persists, please DM us so we can investigate further."
    }
]

prompt_template = """
You are an expert QA judge for customer support AI. Evaluate the following Drafted Reply to the Customer Tweet.
Score the reply out of 5 based on this rubric:
1. Brand Tone (1-5): Is it polite, empathetic, and professional?
2. Actionability (1-5): Does it offer a clear next step (like a link or DM)?
3. Contextual Relevance (1-5): Does it address the specific issue?

Provide a very brief 1-sentence justification, then the final average score out of 5.

Customer Tweet: "{tweet}"
Drafted Reply: "{draft}"

Format exactly like this:
Justification: <your reason>
Score: <number>
"""

print("Running LLM-as-Judge evaluation (5 requests)...\n")

for i, sample in enumerate(samples):
    prompt = prompt_template.format(tweet=sample["tweet"], draft=sample["draft"])
    print(f"--- Sample {i+1} ---")
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        print(response.text.strip())
        print()
        time.sleep(2) # Prevent rate limiting
    except Exception as e:
        print(f"API Error: {e}\n")
