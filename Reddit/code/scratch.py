# %% Science Word Counts
cvec_science = CountVectorizer(stop_words='english', ngram_range=(1,1))
science_words = pd.DataFrame(cvec_science.fit_transform(science['title']).todense(), 
                             columns=cvec_science.get_feature_names())
science_words_counts = pd.DataFrame(science_words.sum().sort_values(ascending=False),
                                    columns= ['count'])
science_words_counts.reset_index(inplace=True)
science_words_df = science_words_counts.rename({'index':'word'}, axis=1)
science_words_df['subreddit'] = 'science'
science_words_df.head(30)

# %%
words_df = pd.concat([history_words_df, science_words_df, lifestyle_words_df])

# %% Word counts for both subreddit  
words500 = words_df[words_df['count'] >= 500]
shared500 = pd.DataFrame(words500['word'].value_counts() == 2)
shared500.reset_index(inplace=True)
shared500 = shared500.rename({'word':'count', 'index':'word'}, axis=1)
shared500 = shared500[shared500['count'] == True]
shared_list = list(shared500.word)
top_shared = words_df[words_df['word'].isin(shared_list)]
plt.figure(figsize=[12,12])
sns.barplot(x= 'count', y= 'word', hue= 'subreddit', data= top_shared);

# %%
words_df[words_df['word'] == 'use']

# %%
words_df.to_csv('../history_science_word_counts.csv', index=False)
