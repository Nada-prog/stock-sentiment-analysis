import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# =========================
# Download NLTK resources
# =========================
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# =========================
# Load the dataset
# =========================
df = pd.read_csv("stock_data.csv")
print("Preview of data:")
print(df.head())

# =========================
# Clean text function
# =========================
def clean_text(text):
    if pd.isna(text):       # Handle missing values
        return ""
    text = re.sub(r"http\S+|[^a-zA-Z\s]", "", text)  # Remove URLs & non-letters
    tokens = word_tokenize(text.lower())             # Tokenize & lowercase
    tokens = [w for w in tokens if w not in stop_words]  # Remove stopwords
    return " ".join(tokens)

# =========================
# Apply cleaning
# =========================
df['clean_text'] = df['Text'].apply(clean_text)

# =========================
# Features & labels
# =========================
X = df['clean_text']
y = df['Sentiment']

# =========================
# Train/test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# TF-IDF vectorizer
# =========================
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# =========================
# Train model
# =========================
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# =========================
# Evaluate
# =========================
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# =========================
# Save model & vectorizer
# =========================
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved successfully!")
