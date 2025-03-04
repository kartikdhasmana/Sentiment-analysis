import streamlit as st
import joblib  # or use pickle if needed
import re
import string

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")  # Make sure this file exists
vectorizer = joblib.load("count_vectorizer.pkl")  # If using TF-IDF or CountVectorizer

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove whitespace
    return text

# Streamlit UI
st.title("📝 Sentiment Analysis App")
st.write("This app predicts whether a given text is **Positive** or **Negative**.")

# User Input
user_input = st.text_area("Enter a sentence for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess input
        processed_text = preprocess_text(user_input)
        
        # Transform text using vectorizer
        input_vector = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(input_vector)[0]
        
        # Display Result
        sentiment = "😊 Positive" if prediction == 0 else "😞 Negative"
        st.subheader(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter text to analyze.")
