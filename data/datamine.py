import os
import pandas as pd
from datetime import date

today = date.today()
end_date = today
print(end_date)

search_term = "World Cup Palestine Islam" # Lionel Messi World Cup palestine Morocco LGBTQ
from_date = '2022-7-30'

# scraping the tweets in link form into results-tweets
os.system(f"snscrape --since {from_date} twitter-search '{search_term} until:{end_date}' > result-tweets.txt")
if os.stat("result-tweets.txt").st_size == 0:
  counter = 0
else:
  df = pd.read_csv('result-tweets.txt', names=['link'])
  counter = df.size

print('Number Of Tweets : '+ str(counter))

# extracting the tweet text from the link into extracted_tweets
max_results = 296
dataframe = pd.DataFrame(columns = ['data'])

extracted_tweets = "snscrape --format '{content!r}'"+ f" --max-results {max_results} --since {from_date} twitter-search '{search_term} until:{end_date}' > extracted-tweets_MUSLIM.txt"
os.system(extracted_tweets)
if os.stat("extracted-tweets_MUSLIM.txt").st_size == 0:
  print('No Tweets found')
else:
  df = pd.read_csv('extracted-tweets_MUSLIM.txt', names=['content'])
  for row in df['content'].iteritems():
    dataframe.append(row)