import streamlit as st
import pickle
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Point to local nltk_data folder for Streamlit Cloud
nltk.data.path.append('./nltk_data')

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# Initialize stemmer
ps = PorterStemmer()

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    
    y = []
    for word in text:
        if word.isalnum():
            y.append(word)
    
    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(ps.stem(word))
    
    return " ".join(y)

# Streamlit UI
st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess
    transformed = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed])

    # Predict
    result = model.predict(vector_input)[0]

    # Output
    if result == 1:
        st.error("Spam 😈")
    else:
        st.success("Not Spam 😊")
