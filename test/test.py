from src.logic.MovieReviewClassifier import MovieReviewClassifier
# import pickle

movie_review_classifier = MovieReviewClassifier("./multiNB_model.pkl")
text = "My friend sent me a review of a movie. They said 'it really sucked'. I can't tell if they liked it or not!"
text_vect = movie_review_classifier.vectorizer.transform([text])
values = movie_review_classifier.model.predict_proba(text_vect)[0]
#average = sum(values[0]) / len(values[0]) 
# review_classification = movie_review_classifier.classify_review("This was the worst two hours of my life!")
print (values)
# print(review_classification) # "Negative"