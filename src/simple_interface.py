from src.logic.MovieReviewClassifier import MovieReviewClassifier

classifier = MovieReviewClassifier('./multiNB_model.pkl')

sentence = input('Enter a sentence to classify: ')
print(f'[Result] {classifier.classify_review(sentence)}')
exit = False
while exit == False:

    if sentence == 'exit' :
        exit = True
    else:
        sentence = input('Enter a sentence to classify: ')
        print(f'[Result] {classifier.classify_review(sentence)}')
    