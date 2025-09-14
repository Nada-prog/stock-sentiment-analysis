# 📈 Stock Sentiment Analysis

## 📌 Project Overview
This project is a **Stock Sentiment Analysis Web Application** that predicts whether stock-related news or tweets have a **Positive, Negative, or Neutral sentiment**.  
It uses **Natural Language Processing (NLP)** with a Machine Learning model and provides a simple **Flask web interface** for interaction.

---

## 🚀 Features
- Preprocess and clean stock-related text data.
- Train a Machine Learning model using TF-IDF and classifiers.
- Save and load trained models (`.pkl` files).
- Flask web app for real-time sentiment prediction.
- User-friendly HTML interface.

---

## 📂 Project Structure
├── templates/ # HTML templates for Flask app
├── venv/ # Virtual environment
├── app.py # Flask web app
├── train.py # Model training script
├── stock_data.csv # Dataset
├── sentiment_model.pkl # Saved trained model
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
├── requirements.txt # Project dependencies  


---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nada-prog/stock-sentiment-analysis.git
   cd stock-sentiment-analysis
Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows


Install dependencies:

pip install -r requirements.txt

📊 Usage
🔹 Train the Model

Run the following command to train the sentiment analysis model:

python train.py

🔹 Run the Flask App
python app.py


Then open your browser and go to:

http://127.0.0.1:5000/


You can now enter stock-related text and get a sentiment prediction.

📷 Screenshots

Add screenshots or GIFs of your app running here.

📈 Example Predictions

"Stock prices are expected to rise significantly this quarter." → Positive

"The company reported huge losses." → Negative

"The stock market is open today." → Neutral

📦 Dependencies

Python 3.x

Flask

Scikit-learn

Pandas

NLTK

(See requirements.txt for full list)

🙌 Acknowledgments

Dataset: Custom stock sentiment dataset (stock_data.csv)

Libraries: Scikit-learn, Flask, Pandas, NLTK

👩‍💻 Author

Nada Ragab
📌 AI & Data Science Enthusiast | Passionate about NLP & ML
