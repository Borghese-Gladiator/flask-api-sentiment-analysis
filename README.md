# Flask REST API for Senttiment Analysis
REST API to expose my Multinomial Naive Bayes sentiment analysis classifier written with NLTK

- see it live here: [https://flask-api-sentiment-analysis.herokuapp.com/](https://flask-api-sentiment-analysis.herokuapp.com/)
- / - returns default description
- /api/predict - POST method to return "Positive" or "Negative" based on given sentence 

## Methodology
- Created naive bayes sentiment analysis classifier with NLTK
- The NLTK Naive Bayes is of the Multinomial variety (typical with classification) as opposed to SKLearn which uses the Gaussian Naive Bayes typically used with continuous data - [https://stackoverflow.com/questions/55154381/difference-between-nltk-and-scikit-naive-bayes](https://stackoverflow.com/questions/55154381/difference-between-nltk-and-scikit-naive-bayes)

## Resources
- Downloaded CSV data using curl from: [https://raw.githubusercontent.com/vineetdhanawat/twitter-sentiment-analysis/master/datasets/Sentiment%20Analysis%20Dataset.csv](https://raw.githubusercontent.com/vineetdhanawat/twitter-sentiment-analysis/master/datasets/Sentiment%20Analysis%20Dataset.csv)
- Wrote sentiment analysis classifier - referenced my previous GitHub repo and the correpsonding article [https://github.com/Borghese-Gladiator/notebook-compilation](https://github.com/Borghese-Gladiator/notebook-compilation) && [https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk](https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk)
- Wrote up app to deploy with help from Heroku docs & [https://towardsdatascience.com/deploy-a-machine-learning-model-using-flask-da580f84e60c](https://towardsdatascience.com/deploy-a-machine-learning-model-using-flask-da580f84e60c)
