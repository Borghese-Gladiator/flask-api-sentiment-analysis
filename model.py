# NAIVE BAYES CLASSIFIER
## Preprocess text (load, tokenize, normalize - remove noise/lemmatize)
## Train naive bayes on processed text

# NLTK - nlp library
import nltk
# download data
nltk.download('twitter_samples')                # load data
nltk.download('punkt')                          # tokenization
nltk.download('stopwords')                      # normalize - removing noise
nltk.download('wordnet')                        # normalize - WordNet Lemmatizer
nltk.download('averaged_perceptron_tagger')     # implement part-of-speech tagging - pos_tag function
# load twitter_samples data
from nltk.corpus import twitter_samples, stopwords
from nltk.tokenize import word_tokenize
from nltk import classify, NaiveBayesClassifier
import random
# write pickle to file
import os
import pickle
# custom text preprocessing utils
from utils import remove_noise 

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

## LOADING DATA
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

text = twitter_samples.strings('tweets.20150430-223406.json')
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

## TOKENIZATION - split phrases into words using punkt tokenization
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

## NORMALIZATION - lemmatize verbs && remove noise from data
stop_words = stopwords.words('english')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]

## TRAINING
dataset = positive_dataset + negative_dataset
random.shuffle(dataset)

X = dataset[:7000] # train_data
y = dataset[7000:] # test_data

classifier = NaiveBayesClassifier.train(X)
print("Accuracy is:", classify.accuracy(classifier, y))

## CLASSIFICATION
print(classifier.show_most_informative_features(10))

custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."

custom_tokens = remove_noise(word_tokenize(custom_tweet))

print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))


## SAVE MODEL
pickle.dump(classifier, open(os.path.join('lib', 'models', 'classifier.pkl'),'wb+'))