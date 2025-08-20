from MovieReviewClassifier import MovieReviewClassifier
classifier = MovieReviewClassifier('./multiNB_model.pkl')
# while exit == False:
sentence = input('Enter a sentence to classify: ')
print(f'[Result] {classifier.classify_review(sentence)}')