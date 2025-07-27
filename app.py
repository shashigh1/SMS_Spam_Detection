import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# ✅ Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()

# ✅ Preprocessing function
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

# ✅ Streamlit UI
st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. Preprocess
    transformed = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("🚨 Spam")
    else:
        st.header("✅ Not Spam")
