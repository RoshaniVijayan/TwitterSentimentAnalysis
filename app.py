import streamlit as st
import re
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load model and vectorizer
@st.cache(allow_output_mutation=True)
def load_model():
    with open('/content/TwitterSentimentAnalysis/twitter_sentiment.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache(allow_output_mutation=True)
def load_vectorizer():
    with open('/content/TwitterSentimentAnalysis/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

# Function to preprocess text
def preprocess_text(text):
    # Define stopwords
    stopwords = set(nlp.Defaults.stop_words)
    
    # Necessary Functions for data cleaning

    # Function to remove emails from text
    def remove_emails(x):
        return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', "", x)

    # Function to remove URLs from text
    def remove_urls(x):
        return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', x)

    # Function to remove HTML tags from text
    def remove_html_tags(x):
        return BeautifulSoup(x, 'lxml').get_text().strip()

    # Function to remove special characters from text
    def remove_special_chars(x):
        x = re.sub(r'[^\w ]+', "", x)
        x = ' '.join(x.split())
        return x

    # Function to remove 'RT' (retweet) from text
    def remove_rt(x):
        return re.sub(r'\brt\b', '', x).strip()

    # Data cleaning
    text = text.lower()
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_special_chars(text)
    text = remove_rt(text)
    
    return text

def predict_sentiment(model, vectorizer, text):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Transform the processed text
    text_vectorized = vectorizer.transform([processed_text])
    
    # Predict sentiment
    prediction = model.predict(text_vectorized)
    
    return prediction[0]

def main():
    # Load model and vectorizer
    model = load_model()
    vectorizer = load_vectorizer()

    # Title
    st.title('Twitter Sentiment Analysis')

    # Text input for user to input tweet
    tweet = st.text_input('Enter your tweet')

    # Predict button
    if st.button('Predict'):
        # Predict sentiment
        prediction = predict_sentiment(model, vectorizer, tweet)
        
        # Display prediction
        st.write('Predicted Sentiment:', prediction)

if __name__ == '__main__':
    main()
