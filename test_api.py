import requests

res = requests.post('http://localhost:5000/api/predict', json={ "data":"This project is to predict sentiment analysis using NLTK's Multionomial Naive Bayes baseline model." })
if res.ok:
  print(res.json())

# custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."
# custom_tokens = remove_noise(word_tokenize(custom_tweet))
# print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))
