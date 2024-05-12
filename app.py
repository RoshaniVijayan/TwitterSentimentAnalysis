
# Import necessary libraries
import streamlit as st
import pickle
import time
import requests

# Define function to download model file from Google Drive
def download_model_file_from_google_drive(file_id, file_name):
    url = "https://drive.google.com/uc?id=" + file_id
    response = requests.get(url)
    with open(file_name, "wb") as file:
        file.write(response.content)

# Define the app title
st.title('Twitter Sentiment Analysis')

# Define Google Drive file ID and model file name
file_id = "1a3vNrSW5WVUgJzDoBMwjETerNZfsapC8"  # Google Drive file ID
model_file_name = "twitter_sentiment.pkl"

# Download model file from Google Drive
try:
    download_model_file_from_google_drive(file_id, model_file_name)
    st.success("Model file downloaded successfully.")
except Exception as e:
    st.error(f"Error downloading model file: {e}")

    # Exit the app if downloading model fails
    raise SystemExit("Downloading model file failed. Please check the file ID.")

# Load the model
try:
    with open(model_file_name, 'rb') as model_file:
        model = pickle.load(model_file)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

    # Exit the app if model loading fails
    raise SystemExit("Model loading failed. Please check the model file.")

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
