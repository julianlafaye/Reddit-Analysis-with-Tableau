# %%
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer 

# %% Read in Data
science = pd.read_csv('../data/eda/science_edata.csv')

# %% Science Word Counts
cvec_science = CountVectorizer(stop_words='english', ngram_range=(1,1))
science_words = pd.DataFrame(cvec_science.fit_transform(science['title']).todense(), 
                             columns=cvec_science.get_feature_names())
# %%
science_words_counts = pd.DataFrame(science_words)
# %% 
science_words_counts.reset_index(inplace=True)
science_words_df = science_words_counts.rename({'index':'word'}, axis=1)
science_words_df['subreddit'] = 'science'
science_words_df.sort_values(by='count')

# %%
