import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
    )

class MovieReviewClassifier:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open('./vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        
      
    def classify_review(self, review):
        print(review)
        predicted = self.model.predict(self.vectorizer.transform([review]))
        if predicted == [0]:
            return 'Negative'
        elif predicted == [1]:
            return 'Positive'
        else:
            return 'Wrong input'
   