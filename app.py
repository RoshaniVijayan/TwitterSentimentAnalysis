import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    return text

# Function to predict sentiment
def predict_sentiment(model, vectorizer, text):
    text = preprocess_text(text)
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]

def main():
    # Title of the app
    st.title('Twitter Sentiment Analysis')

    # Text input for user to enter tweet
    tweet = st.text_input('Enter your tweet')

    # Button to predict sentiment
    if st.button('Predict'):
        # Load the model and vectorizer
        model_path = "/content/TwitterSentimentAnalysis/twitter_sentiment.pkl"  # Update with the correct file path
        model = load_model(model_path)
        vectorizer = TfidfVectorizer()  # You may need to load the vectorizer used during training

        # Make prediction
        prediction = predict_sentiment(model, vectorizer, tweet)
        
        # Display prediction
        st.write('Predicted Sentiment:', prediction)

if __name__ == '__main__':
    main()
