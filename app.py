
# Import necessary libraries
import streamlit as st
import pickle
import time
import requests
from io import BytesIO

# Function to download model file from Google Drive
def download_model_file_from_google_drive(file_id, file_name):
    url = "https://drive.google.com/uc?id=" + file_id
    response = requests.get(url)
    file = open(file_name, "wb")
    file.write(response.content)
    file.close()

# Define the app title
st.title('Twitter Sentiment Analysis')

# Upload the sentiment analysis model file
file_id = "1a3vNrSW5WVUgJzDoBMwjETerNZfsapC8"  # Google Drive file ID
model_file_name = "twitter_sentiment.pkl"

download_model_file_from_google_drive(file_id, model_file_name)

# Load the model
with open(model_file_name, 'rb') as model_file:
    model = pickle.load(model_file)

    # Add a text input field for the user to input the tweet
    tweet = st.text_input('Enter your tweet')

    # Add a button to submit the tweet for prediction
    submit = st.button('Predict')

    # When the user submits the tweet
    if submit:
        # Measure prediction time
        start = time.time()
        # Predict the sentiment of the tweet
        prediction = model.predict([tweet])
        end = time.time()
        # Display prediction time
        st.write('Prediction time taken: ', round(end-start, 2), 'seconds')
        # Display the predicted sentiment
        st.write('Predicted Sentiment:', prediction[0])
