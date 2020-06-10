# %%
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer 

# %% Read in Data
history = pd.read_csv('../data/eda/history_edata.csv')
# %% History Word Counts
cvec_history = CountVectorizer(stop_words='english', ngram_range=(1,1))
history_words = pd.DataFrame(cvec_history.fit_transform(history['title']).todense(),
                              columns=cvec_history.get_feature_names())
history_words_counts = pd.DataFrame(history_words.sum().sort_values(ascending=False),
                                    columns= ['count'])
history_words_counts.reset_index(inplace=True)
history_words_df = history_words_counts.rename({'index':'word'}, axis=1,)
history_words_df['subreddit'] = 'history'
history_words_df.head(30)
