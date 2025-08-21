from src.logic.MovieReviewClassifier import MovieReviewClassifier
from src.chatbot_interface import Chatbot_Interface
# classifier = MovieReviewClassifier('./multiNB_model.pkl')

# sentence = input('Enter a sentence to classify: ')
# print(f'[Result] {classifier.classify_review(sentence)}')
# exit = False
# while exit == False:

#     if sentence == 'exit' :
#         exit = True
#     else:
#         sentence = input('Enter a sentence to classify: ')
#         print(f'[Result] {classifier.classify_review(sentence)}')


bot = Chatbot_Interface(classifier_model_path="./multiNB_model.pkl")

user_message = "My friend sent me a review of a movie. They said 'it really sucked'. I can't tell if they liked it or not!"

reply= bot.generate_reply(user_message)
# print(proba_row)
print("==== User message ====")
print(user_message)
print("\n==== Bot reply ====")
print(reply)
#print(f' {confidence} confidence')