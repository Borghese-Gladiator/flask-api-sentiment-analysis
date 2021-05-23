# Core libraries for web server and decoding requests
from flask import Flask, request, jsonify
import json
# Load model object from pickle file
import pickle
import os
# Utils
from utils import remove_noise 
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load the model
model = pickle.load(open(os.path.join('lib', 'models', 'classifier.pkl'),'rb'))

@app.route('/', methods=['GET'])
def indexPage():
    output = {
        "msg": "This project is to predict sentiment analysis using NLTK's Multionomial Naive Bayes baseline model."
    }
    return jsonify(output)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True) # Get the data from the POST request.
    stringData = data["data"] ## data is expected to be a string - eg: custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."
    custom_tokens = remove_noise(word_tokenize(stringData))
    prediction = model.classify(dict([token, True] for token in custom_tokens))

    return jsonify(prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)