import pickle
import os
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

vect = pickle.load(open("main/model/vect.pk", "rb"))
model = pickle.load(open("main/model/model.pk", "rb"))


def data_processing(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub('<br />', '', text)
    text = re.sub(r'https\S+|www\S+|http\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if w not in stop_words]
    return ' '.join(filtered_text)


def stemming(data):
    ps = PorterStemmer()
    return ' '.join([ps.stem(word) for word in data.split()])


def predict(review):
    d = {0: 'negative', 1: 'positive'}
    review = data_processing(review)
    review = stemming(review)
    review = [review]
    x = vect.transform(review)
    rating = 1 + round(model.predict_proba(x)[0][1] * 9)
    sentiment = d[model.predict(x)[0]]
    return sentiment, rating



