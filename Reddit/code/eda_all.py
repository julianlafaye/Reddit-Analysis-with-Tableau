# %%
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer 

# %% Read in Data
history = pd.read_csv('../data/eda/history_edata.csv')
science = pd.read_csv('../data/eda/science_edata.csv')
lifestyle = pd.read_csv('../data/eda/lifestyle_edata.csv')

# %% Describe data
history.describe()
science.describe()
lifestyle.describe()
# history[history['score']>history['score'].mean()].describe()
# science[science['score']>science['score'].mean()].describe()

# %%
df['title_len'] = df['title'].map(len)
science['title_len'] = science['title'].map(len)
history['title_len'] = history['title'].map(len)
lifestyle['title_len'] = lifestyle['title'].map(len)

# %%
sns.distplot(history['title_len'])
plt.title('Ask History Title Length')

# %%
sns.distplot(science['title_len'])
plt.title('Ask Science Title Length')

# %%
sns.distplot(lifestyle['title_len'])
plt.title('AskMen/Women Title Length')
# %%
sum(df['title_len'] < 5)
# %%
df[df['title_len'] <= 30]