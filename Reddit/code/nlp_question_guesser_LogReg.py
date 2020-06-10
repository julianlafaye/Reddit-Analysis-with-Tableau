# %% Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
# %% Read in Data
ask_lifestyle = pd.read_csv('../data/men_women.csv')
ask_science_history = pd.read_csv('../data/history_science.csv')

# %% Concat

ask_lifestyle['subreddit']='lifestyle'
ask_science_history['subreddit'] = ask_science_history['subreddit'].map({'askscience' : 'science', 'AskHistorians' : 'history'})

df = pd.concat([ask_lifestyle, ask_science_history])
# df.to_csv('../data/ask_subs.csv')
# %%
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
                                                    test_size=0.25,
                                                    stratify=y,
                                                    random_state=42)
y_test.value_counts(normalize=True)
# %%
pipe_params = {
    'cvec__ngram_range': [(1,1), (1,2), (1,3)]
}
gs = GridSearchCV(pipe,
                  pipe_params,
                  cv= 5,
                  n_jobs=-1
                  )

gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.score(X_test, y_test))
# %% CVec LogReg
pipe = Pipeline(steps=[('cvec', CountVectorizer(ngram_range=(1,3))),
                        ('LogReg', LogisticRegression(n_jobs=4))])
pipe.fit(X_train, y_train)
print(pipe.score(X_train, y_train))
print(pipe.score(X_test, y_test))

# %% Other Models
# # %% TVec LogReg
# pipe = Pipeline(steps=[('tvec', TfidfVectorizer()),
#                         ('LogReg', LogisticRegression(n_jobs=-1))])
# pipe.fit(X_train, y_train)
# print(pipe.score(X_train, y_train))
# print(pipe.score(X_test, y_test))
# print(pipe.score(X_trial, y_trial))

# # %% DecisionTreeClassifier
# pipe = Pipeline(steps=[('cvec', CountVectorizer()),
#                         ('tree', DecisionTreeClassifier())])
# pipe.fit(X_train, y_train)
# print(pipe.score(X_train, y_train))
# print(pipe.score(X_test, y_test))
# print(pipe.score(X_trial, y_trial))

# # %% KNN
# pipe = Pipeline(steps=[('cvec', CountVectorizer()),
#                         ('knn', KNeighborsClassifier(n_jobs=-1))])
# pipe.fit(X_train, y_train)
# print(pipe.score(X_train, y_train))
# print(pipe.score(X_test, y_test))

# # %% KNN
# pipe = Pipeline(steps=[('tvec', TfidfVectorizer()),
#                         ('knn', KNeighborsClassifier(n_jobs=-1))])
# pipe.fit(X_train, y_train)
# print(pipe.score(X_train, y_train))
# print(pipe.score(X_test, y_test))

# # %% Multinomial Naive Bayes
# pipe = Pipeline(steps=[('cvec', CountVectorizer()),
#                         ('nb', MultinomialNB())])
# pipe.fit(X_train, y_train)
# print(pipe.score(X_train, y_train))
# print(pipe.score(X_test, y_test))

# # %% Gaussian Naive Bayes
# pipe = Pipeline(steps=[('cvec', CountVectorizer()),
#                        ('to_dense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
#                        ('gnb', GaussianNB())])
# pipe.fit(X_train, y_train)
# print(pipe.score(X_train, y_train))
# print(pipe.score(X_test, y_test))

# %%
predict = pd.DataFrame(X)
predict['title'] = df['title']
predict['prediction'] = pipe.predict(X)
predict['pred_prob_history'] = pipe.predict_proba(X)[:,0]
predict['pred_prob_lifestyle'] = pipe.predict_proba(X)[:,1]
predict['pred_prob_science'] = pipe.predict_proba(X)[:,2]
predict['actual'] = y
predict['comments'] = df['num_comments']
predict['score'] = df['score']
predict['url'] = df['url']
# predict['text'] = df['selftext']
predict['title_len'] = df['title'].map(len)
predictions = predict[predict['prediction'] != predict['actual']]
# %%
predict.to_csv('../data/predictions/question_LogReg_predictions.csv', index=False)
predict.head()
# %%