# %% Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
# %% Read in Data
df = pd.read_csv('../data/all_data.csv')

# %% Clean SelfText
df['selftext'] = df['selftext'].replace(np.nan, '', regex=True)
df['selftext'] = df['selftext'].replace('[removed]', '', regex=True)
df['X'] = df['title'] + df['selftext']
df['X'].head()

# %% Set X & y
X = df['title']
y = df['subreddit']

# %% Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    stratify=y,
                                                    random_state=42)
y_test.value_counts(normalize=True)
y_test.shape
# %% SVC 
pipe = Pipeline(steps=[('cvec', CountVectorizer()),
                        ('SGD', SGDClassifier())])

pipe.fit(X_train, y_train)
print(pipe.score(X_train, y_train))
print(pipe.score(X_test, y_test))


# %%
pipe_params = {
    'cvec__ngram_range': [(1,1), (1,2), (1,3), (1,4)],
    'SGD__loss': ['hinge', 'log']
}
gs = GridSearchCV(pipe,
                  pipe_params,
                  cv= 5,
                  n_jobs=6
                  )

gs.fit(X_train, y_train)
# %%
print(gs.best_score_)
print(gs.score(X_test, y_test))
gs.best_params_
# %% Export predictions
predict = pd.DataFrame()
predict['title'] = df['title']
predict['prediction'] = pipe.predict(X)
predict['actual'] = y
predict['comments'] = df['num_comments']
predict['score'] = df['score']
# predict['text'] = df['selftext']
predict['title_len'] = df['title'].map(len)
predict.to_csv('../data/predictions/question_SGD_predictions.csv', index=False)
predict.head()

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

# %% Set X & y
X = df['X']
y = df['subreddit']

# %% Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    stratify=y,
                                                    random_state=42)
y_test.value_counts(normalize=True)
y_test.shape
# %% SVC 
pipe = Pipeline(steps=[('cvec', CountVectorizer()),
                        ('SGD', SGDClassifier(n_jobs=4))])

pipe.fit(X_train, y_train)
print(pipe.score(X_train, y_train))
print(pipe.score(X_test, y_test))

# %%
pipe_params = {
    'cvec__ngram_range': [(1,1), (1,2), (1,3), (1,4)],
    'SGD__loss': ['hinge', 'log']
}
gs = GridSearchCV(pipe,
                  pipe_params,
                  cv= 5,
                  n_jobs=6
                  )

gs.fit(X_train, y_train)
# %%
print(gs.best_score_)
print(gs.score(X_test, y_test))
gs.best_params_
# %%
# %% Export predictions
predict = pd.DataFrame()
predict['title'] = df['title']
predict['prediction'] = pipe.predict(X)
predict['actual'] = y
predict['comments'] = df['num_comments']
predict['score'] = df['score']
# predict['text'] = df['selftext']
predict['title_len'] = df['title'].map(len)
predictions = predict[predict['prediction'] != predict['actual']]
predict.to_csv('../data/predictions/question_SGD_predictions_self_text.csv', index=False)
predict.head()
# %%
