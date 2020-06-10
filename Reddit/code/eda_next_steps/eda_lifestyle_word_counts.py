# %%
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer 

# %% Read in Data

# %% women Word Counts
cvec_women = CountVectorizer(stop_words='english', ngram_range=(1,1))
women_words = pd.DataFrame(cvec_women.fit_transform(df_askwomen['title']).todense(), 
                             columns=cvec_women.get_feature_names())
women_words_counts = pd.DataFrame(women_words.sum().sort_values(ascending=False),
                                    columns= ['count'])
women_words_counts.reset_index(inplace=True)
women_words_df = women_words_counts.rename({'index':'word'}, axis=1)
women_words_df['subreddit'] = 'women'
women_words_df.head(30)

# %% men Word Counts
cvec_men = CountVectorizer(stop_words='english', ngram_range=(1,1))
men_words = pd.DataFrame(cvec_men.fit_transform(df_askmen['title']).todense(),
                              columns=cvec_men.get_feature_names())
men_words_counts = pd.DataFrame(men_words.sum().sort_values(ascending=False),
                                    columns= ['count'])
men_words_counts.reset_index(inplace=True)
men_words_df = men_words_counts.rename({'index':'word'}, axis=1,)
men_words_df['subreddit'] = 'men'
men_words_df.head(30)

# %%
words_df = pd.concat([men_words_df, women_words_df])

# %% Word counts for both subreddit  
words500 = words_df[words_df['count'] >= 1000]
shared500 = pd.DataFrame(words500['word'].value_counts() == 2)
shared500.reset_index(inplace=True)
shared500 = shared500.rename({'word':'count', 'index':'word'}, axis=1)
shared500 = shared500[shared500['count'] == True]
shared_list = list(shared500.word)
top_shared = words_df[words_df['word'].isin(shared_list)]
plt.figure(figsize=[12,12])
sns.barplot(x= 'count', y= 'word', hue= 'subreddit', data= top_shared);

# %%
words_df.to_csv('../men_women_word_counts.csv', index=False)

# %%
