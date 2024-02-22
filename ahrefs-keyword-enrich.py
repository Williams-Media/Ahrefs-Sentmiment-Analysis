import os
import pandas as pd
import time
from google.cloud import language_v1
from tqdm import tqdm

# Set the environment variable for Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './YOUR-GOOGLE-API-KEY.json'

# Initialize the Google Natural Language client
client = language_v1.LanguageServiceClient()

# Function to analyze sentiment of a single text
def analyze_sentiment(text):
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment
    return sentiment.score, sentiment.magnitude

# Function to determine emotional leaning
def emotional_leaning(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Function to determine strength of emotion
def strength_of_emotion(magnitude):
    if magnitude < 0.3:
        return 'Low'
    elif magnitude < 0.6:
        return 'Moderate'
    else:
        return 'High'

# Load the CSV file
df = pd.read_csv('./hrefs-Keyword-Data.csv', encoding='utf-16', sep='\t')

# Define batch size and delay between batches
batch_size = 10  # Adjust based on your needs
delay = 30  # Delay in seconds for 2 requests per minute

# Process keywords in batches with a progress bar and save progress incrementally
for i in tqdm(range(0, len(df), batch_size), desc="Analyzing Sentiments"):
    batch = df['Keyword'][i:i+batch_size].tolist()
    batch_sentiments = [analyze_sentiment(keyword) for keyword in batch]
    sentiments = [(analyze_sentiment(keyword), emotional_leaning(score), strength_of_emotion(magnitude)) for keyword, (score, magnitude) in zip(batch, batch_sentiments)]
    
    for j, (sentiment, leaning, strength) in enumerate(sentiments):
        df.loc[i+j, 'Sentiment Score'] = sentiment[0]
        df.loc[i+j, 'Sentiment Magnitude'] = sentiment[1]
        df.loc[i+j, 'Emotional Leaning'] = leaning
        df.loc[i+j, 'Strength of Emotion'] = strength

    # Save the updated DataFrame to the CSV file incrementally
    df.to_csv('./hrefs-Keyword-Data.csv', index=False, encoding='utf-16', sep='\t')

    # Introduce a delay between batches
    time.sleep(delay)

# Final save (optional, as data is already being saved in each iteration)
df.to_csv('./Ahrefs-Keyword-Data.csv', index=False, encoding='utf-16', sep='\t')

