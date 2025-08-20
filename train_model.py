from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
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
X_test = vectorizer.transform(X_test)
# X_train.toarray()
# X_test.toarray()

gnb = LogisticRegression()

model = gnb.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average="macro"))
print(precision_score(y_test, y_pred, average="macro"))
print(recall_score(y_test, y_pred, average="macro"))
print(confusion_matrix(y_test, y_pred))

print(model.predict(vectorizer.transform(['this movie is good'])))

# Example of using pickle to save a model.
import pickle

# Save
with open('multiNB_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
# from sklearn.pipeline import make_pipeline
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.naive_bayes import MultinomialNB

# pipe = make_pipeline(
#     TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"),
#     MultinomialNB()
# )

# pipe.fit(X_train, y_train)     # fits TF-IDF + trains the classifier
# y_pred = pipe.predict(X_test)  # transforms X_test with the same TF-IDF, then predicts

# # import pickle
# with open("trained_model.pkl", "wb") as f:
#     pickle.dump(pipe, f)

# # import pickle
# with open("trained_model.pkl", "rb") as f:
#     pipe = pickle.load(f)

# pipe.predict(["This movie was fantastic!"])  # one-liner inference
# # Load
# with open('your_model.pkl', 'rb') as f:
#     model = pickle.load(f)