# 🚨 SMS Spam Detection Web App

An end-to-end machine learning project to detect **SMS spam messages** using **Natural Language Processing (NLP)** and **Support Vector Machine (SVM)**, deployed using **Streamlit**.  
It uses advanced text preprocessing techniques and TF-IDF vectorization to make real-time predictions.

[🔴 Try the Live Demo](https://sms-spam-detection-skk.streamlit.app/)  
[📂 View the Repository](https://github.com/shashigh1/SMS_Spam_Detection)

---

## 📌 Table of Contents

- [🔍 Overview](#-overview)
- [🧠 Model Pipeline](#-model-pipeline)
- [💡 Sample Predictions](#-sample-predictions)
- [📈 Results & Evaluation](#-results--evaluation)
- [🛠️ Tech Stack](#️-tech-stack)
- [👨‍💻 Author](#-author)

---

## 🔍 Overview

This project classifies incoming text messages as either **Spam** or **Not Spam (Ham)**. It combines classical machine learning with modern web app deployment to provide a fast, accurate, and user-friendly SMS classification tool.

---

## 🧠 Model Pipeline

1. **Data Cleaning**
   - Removing symbols, punctuations, and special characters
2. **Tokenization** using `nltk.word_tokenize`
3. **Stopword Removal**
4. **Stemming** using `PorterStemmer`
5. **Vectorization** using **TF-IDF**
6. **Model Training** using **SVM Classifier**
7. **Deployment** using **Streamlit**

💡 Sample Predictions
Input Message	Prediction
"Congratulations! You've won a lottery worth $10,000. Claim now!"	Spam
"Meeting rescheduled to 3 PM. Please confirm your availability."	Not Spam
"Free entry to the iPhone contest! Click here to register."	Spam
"Reminder: Your electricity bill is due tomorrow. Kindly make the payment."	Not Spam

📈 Results & Evaluation
Model Used: Support Vector Machine (SVM)

Accuracy: ~98%

Vectorization: TF-IDF

Preprocessing: NLTK tokenization, stemming, stopword removal

🛠️ Tech Stack
Category	Tools Used
Language	Python 3.11
ML Algorithm	Support Vector Machine (SVM)
NLP	NLTK
Vectorization	TF-IDF (scikit-learn)
Deployment	Streamlit

👨‍💻 Author
Shashikant Kumar
🎓 M.Tech, CSE (Data Science), NIT Patna
📧 shashikantkumar744@gmail.com
🔗 LinkedIn
💻 GitHub
