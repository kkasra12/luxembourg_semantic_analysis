import http.client
import json
import os
import pandas as pd
from api import twitter_api

# Function to fetch data from the API
def fetch_twitter_data(query, count, batch_num):
    conn = http.client.HTTPSConnection("twitter135.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': twitter_api,
        'x-rapidapi-host': "twitter135.p.rapidapi.com"
    }

    endpoint = f"/Search/?q={query}&count={count}&type=Top&safe_search=true"
    conn.request("GET", endpoint, headers=headers)
    res = conn.getresponse()
    data = res.read()

    conn.close()
    return json.loads(data), f"twitter_search_results_batch_{batch_num}.json"

# Function to extract tweet texts from a JSON file
def extract_tweet_texts(file_path):
    tweet_texts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        entries = data.get("data", {}).get("search_by_raw_query", {}).get("search_timeline", {}).get("timeline", {}).get("instructions", [])
        for instruction in entries:
            if instruction.get("type") == "TimelineAddEntries":
                for entry in instruction.get("entries", []):
                    content = entry.get("content", {})
                    if content.get("entryType") == "TimelineTimelineItem":
                        tweet_result = content.get("itemContent", {}).get("tweet_results", {}).get("result", {})
                        legacy = tweet_result.get("legacy", {})
                        full_text = legacy.get("full_text")
                        if full_text:
                            tweet_texts.append({"Tweet Text": full_text})
    return tweet_texts

# Parameters
query = "Luxembourg"
count = 20  # Number of results per request
num_batches = 7  # Number of JSON files to create
output_dir = "twitter_batches"  # Directory to store individual JSON files
csv_file = "tweets.csv"  # Output CSV file

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Fetch and save multiple JSON files
for batch_num in range(1, num_batches + 1):
    print(f"Fetching batch {batch_num}")
    data, filename = fetch_twitter_data(query, count, batch_num)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w", encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    print(f"Saved batch {batch_num} to {file_path}")

# Iterate over all JSON files in the directory and extract texts
all_tweet_texts = []

for filename in os.listdir(output_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(output_dir, filename)
        print(f"Processing file: {file_path}")
        tweet_texts = extract_tweet_texts(file_path)
        all_tweet_texts.extend(tweet_texts)

# Convert to DataFrame and save to CSV
df = pd.DataFrame(all_tweet_texts)
if os.path.exists(csv_file):
    # Append to existing CSV file
    df.to_csv(csv_file, mode='a', index=False, header=False, quotechar='"', encoding='utf-8')
else:
    # Create new CSV file
    df.to_csv(csv_file, index=False, quotechar='"', encoding='utf-8')

print(f"All tweets have been processed and saved to {csv_file}.")


print(df)



