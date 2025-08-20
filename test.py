from MovieReviewClassifier import MovieReviewClassifier
# import pickle

movie_review_classifier = MovieReviewClassifier("./multiNB_model.pkl")

review_classification = movie_review_classifier.classify_review("This was the worst two hours of my life!")

print(review_classification) # "Negative"