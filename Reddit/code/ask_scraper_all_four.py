# %% Scraper
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 

sub1 = 'askscience'
sub2 = 'AskHistorians'
sub3 = 'AskWomen'
sub4 = 'AskMen'
n =100000
sub_list = [sub1]
sub_list2 = [sub2]
n2 = 50000
sub_list3 = [sub3, sub4]

s_df = pd.DataFrame()
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
        s_df = pd.concat([s_df, results_df[keep]])
s_df.to_csv('../data/eda/science_edata.csv')

h_df = pd.DataFrame()
for sub in sub_list2:
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
        h_df = pd.concat([h_df, results_df[keep]])
h_df.to_csv('../data/eda/history_edata.csv')
    
mw_df = pd.DataFrame()
for sub in sub_list3:
    N=0
    last = ''
    while N < n2:
        url = f'https://api.pushshift.io/reddit/search/submission/?subreddit={sub}&size=1000&before={last}'
        response = requests.get(url)
        result = response.json()
        for s in result['data']:
            N += 1
        last = int(s['created_utc'])
        results_df = pd.DataFrame(result['data'])
        keep = ['subreddit', 'title', 'selftext', 'created_utc', 'score', 'num_comments']
        mw_df = pd.concat([mw_df, results_df[keep]])
mw_df.to_csv('../data/eda/lifestyle_edata.csv')

df = pd.concat([mw_df, h_df, s_df])
df['subreddit'] = df['subreddit'].map({'askscience' : 'science',
                                          'AskHistorians' : 'history',
                                          'AskWomen' : 'lifestyle',
                                          'AskMen' : 'lifestyle'})
df.to_csv('../data/all_data.csv', index=False)
