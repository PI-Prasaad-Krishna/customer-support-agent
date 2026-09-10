import pandas as pd
import os

print("Loading dataset...")
df = pd.read_csv('data/twcs.csv')

print(f"Total tweets: {len(df)}")

# Find all tweets by AppleSupport
apple_tweets = df[df['author_id'] == 'AppleSupport']
print(f"Total AppleSupport tweets: {len(apple_tweets)}")

# Get the inbound tweets they are responding to
apple_responses = apple_tweets[apple_tweets['in_response_to_tweet_id'].notna()]
inbound_ids = apple_responses['in_response_to_tweet_id'].astype(int).tolist()

inbound_tweets = df[df['tweet_id'].isin(inbound_ids)]
print(f"Total Inbound tweets responded to by AppleSupport: {len(inbound_tweets)}")

# Merge them
merged = pd.merge(inbound_tweets, apple_responses, left_on='tweet_id', right_on='in_response_to_tweet_id', suffixes=('_inbound', '_outbound'))
print(f"Total paired conversations: {len(merged)}")

# Save a 50k subset
subset = merged.sample(n=min(50000, len(merged)), random_state=42)
os.makedirs('data', exist_ok=True)
subset.to_csv('data/apple_support_pairs.csv', index=False)
print("Saved 50k pairs to data/apple_support_pairs.csv")
