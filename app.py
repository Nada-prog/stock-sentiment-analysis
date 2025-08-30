from flask import Flask, request, render_template
from textblob import TextBlob
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"http\S+|[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = ""
    if request.method == "POST":
        text = request.form["text"]
        cleaned = clean_text(text)
        sentiment = get_sentiment(cleaned)
    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
