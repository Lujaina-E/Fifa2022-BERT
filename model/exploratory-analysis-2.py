import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


# Load the clustered tweets data
df = pd.read_csv('../data/tweets_clustered.csv')

# Perform sentiment analysis on each tweet
df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Identify top keywords for each cluster
clusters = df['cluster'].unique()
vectorizer = CountVectorizer(stop_words='english')
for cluster in clusters:
    cluster_df = df[df['cluster'] == cluster]
    cluster_texts = cluster_df['text'].tolist()
    vectorized_texts = vectorizer.fit_transform(cluster_texts)
    feature_names = vectorizer.get_feature_names_out()
    cluster_keywords = [feature_names[idx] for idx in vectorized_texts.sum(axis=0).argsort()[::-1][:10]]
    print(f"Cluster {cluster} Keywords: {cluster_keywords}")

# Visualize sentiment distribution within each cluster
for cluster in clusters:
    cluster_df = df[df['cluster'] == cluster]
    plt.hist(cluster_df['sentiment'], bins=20, range=(-1, 1), alpha=0.5, label=f'Cluster {cluster}')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Count')
    plt.title(f'Sentiment Distribution - Cluster {cluster}')
    plt.legend()
    plt.show()