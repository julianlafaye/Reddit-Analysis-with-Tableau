# %% Imports
import requests
import pandas as pd
# %% Read in Data
df = pd.read_csv('../data/ask_subs.csv')
# %% Grab Data
sub1 = 'askscience'
sub2 = 'AskHistorians'
sub3 = 'Advice'
sub_list = [sub1, sub2, sub3]
tdf = pd.DataFrame()

for sub in sub_list:
    url = f'https://api.pushshift.io/reddit/search/submission/?subreddit={sub}&size=1000'
    response = requests.get(url)
    result = response.json()
    results_df = pd.DataFrame(result['data'])
    keep = ['subreddit', 'title', 'num_comments', 'created_utc', 'score']
    tdf = pd.concat([tdf, results_df[keep]])

# %%
tdf['subreddit'].value_counts()
# %%
tdf['subreddit'] = tdf['subreddit'].map({'askscience' : 'science',
                                          'AskHistorians' : 'history',
                                          'Advice' : 'lifestyle',})
# %%
tdf.to_csv('../data/test.csv', index=False)


# %%
