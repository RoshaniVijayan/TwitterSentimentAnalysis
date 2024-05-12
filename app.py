# Import necessary libraries
import streamlit as st
import pickle
import time

# Define the app title
st.title('Twitter Sentiment Analysis')

# Define the model file name
model_file_name = "twitter_sentiment.pkl"

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
