# %% Imports
import requests 
import pandas as pd
# %% Read in Data
df = pd.read_csv('../data/ask_subs.csv')
# %% Grab Data
sub1 = 'AskReddit'
sub_list = [sub1,]
tdf = pd.DataFrame()
n = 10000
for sub in sub_list:
    N=0
    last = ''
    while N < n:
        url = f'https://api.pushshift.io/reddit/search/submission/?subreddit={sub}&size=1000&before={last}'
        response = requests.get(url)
        result = response.json()
        for s in result['data']:
            N += 1
        last = int(s['created_utc'])
        results_df = pd.DataFrame(result['data'])
        keep = ['subreddit', 'title', 'selftext', 'created_utc', 'score', 'num_comments']
        tdf = pd.concat([tdf, results_df[keep]])

# %%
tdf['subreddit'].value_counts()

# %%
tdf.to_csv('../data/ask_reddit_test.csv', index=False)


# %%
