import os
import json
# print(os.listdir('./data/aclImdb/test/neg')[0])

def extract(path, pos_or_neg):
    reviews = os.listdir(path)
    data = []
    for review in reviews:

        movie_id, rating = review.split('_')
        rating = rating[:-4]    
        with open(f'{path}/{review}', 'r') as f: 
            
            review_data = {'id': movie_id, 'rating': rating, 'review': f.read(), 'pos_or_neg': pos_or_neg}
            data.append(review_data)
    return data


def get_all_review_data():
    whole_data = []
    
    test_neg = extract('./aclImdb/test/neg', 0)
    test_pos = extract('./aclImdb/test/pos', 1)
    train_neg = extract('./aclImdb/train/neg', 0)
    train_pos = extract('./aclImdb/train/pos', 1)

    for item in test_neg:
        whole_data.append(item)
    for item in test_pos:
        whole_data.append(item)
    for item in train_neg:
        whole_data.append(item)
    for item in train_pos:
        whole_data.append(item)
    with open('./json_data/movie_data.json', 'w') as f:
        json.dump(whole_data, f, indent=4)

get_all_review_data()