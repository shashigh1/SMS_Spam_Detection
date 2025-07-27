# app.py
import streamlit as st
import pickle
from preprocessing import transform_text

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed = transform_text(input_sms)
    vector_input = vectorizer.transform([transformed]).toarray()
    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("Spam Message")
    else:
        st.success("Not Spam")
