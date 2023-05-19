import re
import pandas as pd
import re
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('../data/tweets.csv')

# Preprocess tweets
def preprocess_tweet_text(tweet_text):
    tweet_text = re.sub(r'http\S+', '', tweet_text)
    tweet_text = re.sub(r'@\w+', '', tweet_text)
    tweet_text = tweet_text.encode('ascii', 'ignore').decode('ascii')
    tweet_text = re.sub(r'[^\w\s]', '', tweet_text)
    tweet_text = tweet_text.lower()
    return tweet_text

df['text'] = df['text'].apply(preprocess_tweet_text)

# Tokenize and encode tweets
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encoded_tweets = []
for tweet_text in df['text']:
    encoded_tweet = tokenizer.encode_plus(
        tweet_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    encoded_tweets.append(encoded_tweet)

# Load BERT model
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Generate embeddings for each tweet
embeddings = []
for encoded_tweet in encoded_tweets:
    input_ids = encoded_tweet['input_ids']
    attention_mask = encoded_tweet['attention_mask']

    # Get the BERT model outputs
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    hidden_states = outputs[0]

    # Use the last four hidden states to generate the embeddings
    embedding = torch.mean(hidden_states[-4:], dim=1)
    embeddings.append(embedding.numpy())

# Convert embeddings to numpy array
embeddings = np.array(embeddings)
embeddings = embeddings.reshape(len(embeddings), -1)

# Cluster the data
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(embeddings)

clusters = kmeans.predict(embeddings)
df['cluster'] = clusters
df.to_csv('tweets_clustered.csv', index=False)


# Get predictions based on user input
user_input = input("Enter a tweet: ")
encoded_input = tokenizer.encode_plus(
    user_input,
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)


input_embedding = torch.mean(model(encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'])[0][-4:], dim=1).detach().numpy().reshape(1, -1)
prediction = kmeans.predict(input_embedding)
print("Prediction:", prediction[0])
