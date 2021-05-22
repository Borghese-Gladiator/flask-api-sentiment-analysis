# Multinomial Naive Bayes - baseline solution for sentiment analysis task
## find probabilities of classes assigned to texts by using joint probabilites of words and classes

## LOAD DATA
import pandas as pd
import numpy as np
# Read the data from CSV files
n = ['id', 'date','name','text','typr','rep','rtw','faw','stcount','foll','frien','listcount']
data_positive = pd.read_csv('positive.csv', sep=';',error_bad_lines=False, names=n, usecols=['text'])
data_negative = pd.read_csv('negative.csv', sep=';',error_bad_lines=False, names=n, usecols=['text'])
# Create balanced dataset
sample_size = min(data_positive.shape[0], data_negative.shape[0])
raw_data = np.concatenate((data_positive['text'].values[:sample_size], 
                           data_negative['text'].values[:sample_size]), axis=0) 
labels = [1]*sample_size + [0]*sample_size

## PREPROCESS
import re
def preprocess_text(text):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', text)
    text = re.sub('@[^\s]+','USER', text)
    text = text.lower().replace("ё", "е")
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +',' ', text)
    return text.strip()
data = [preprocess_text(t) for t in raw_data]

## TRAINING
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
}

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)

from sklearn.metrics import classification_report
clf = GridSearchCV(text_clf, tuned_parameters, cv=10, scoring=score)
clf.fit(x_train, y_train)

print(classification_report(y_test, clf.predict(x_test), digits=4))