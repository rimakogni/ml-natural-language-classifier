from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

RANDOM_SEED = 2042

df_movies = pd.read_json("./json_data/movie_data.json")

# print(df_movies.head())

X, y = df_movies['review'].to_list(), df_movies['pos_or_neg'].to_list()

# print(X[:2])
# print(y[:10])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
    )
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.fit_transform(X_test)
X_train
X_test

gnb = GaussianNB()

model = gnb.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f1_score(y_test, y_pred, average="macro"))
print(precision_score(y_test, y_pred, average="macro"))
print(recall_score(y_test, y_pred, average="macro"))