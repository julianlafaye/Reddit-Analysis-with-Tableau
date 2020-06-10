
### The Data Science Process
**Problem Statement**
The aim of this repo is to gather Data from several Question-based subreddits and build a model to predict where a question came from/ the type of question being asked. I wanted to focus on inference given that I knew it would not be the highest performing model.


**Data Collection**

I gathered data from a total of 5 subreddits(askscience, AskHistorians, Advice, AskMen, AskWomen) using reddit pushshift api. I divided the data from the subreddits into three categories (Lifestyle, Science, History).

**Data Cleaning and EDA**
Reddit data was very clean to start so I had no problems jumping right into EDA. However most of my EDA was post model building. The link to my tableau workbooks can be found [here](https://public.tableau.com/profile/julian.lafaye#!/vizhome/QuestionSubredditModel1/Dashboard1?publish=yes) and [here](https://public.tableau.com/profile/julian.lafaye#!/vizhome/QuestionData2/TitleLengthCharts)

**Preprocessing and Modeling**

I tried I number of models but none out performed my first, Logistic Regression with sklearn CountVectorizer

**Conclusion and Recommendations**

For my next steps I would love to run a word vectorizer 