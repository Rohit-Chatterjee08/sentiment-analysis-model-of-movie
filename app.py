import os
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir())
print("Files in 'model/' directory:", os.listdir("model"))

import gradio as gr
import joblib
import re
import string

# Load your trained model and vectorizer
model = joblib.load("model/imdb_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

# Prediction function
def predict_sentiment(review):
    review = preprocess_text(review)
    vectorized = vectorizer.transform([review])
    prediction = model.predict(vectorized)[0]
    sentiment = "✅ Positive" if prediction == 1 else "❌ Negative"
    return sentiment

# Gradio interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="Enter IMDB movie review here..."),
    outputs="text",
    title="🎬 IMDB Movie Review Sentiment Classifier",
    description="Paste a movie review to predict its sentiment (Positive or Negative). Model trained on IMDB dataset."
)

iface.launch()

