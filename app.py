import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# load model
model = tf.keras.models.load_model("sentiment_cnn.keras")

# load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 80

# preprocessing (same as training)
pattern = re.compile(r"(?:\@|https?\://)\S+|[^\w\s#]")
lemm = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = pattern.sub("", text)
    tokens = text.split()
    tokens = [lemm.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

st.title("Twitter Sentiment Analyzer")

text = st.text_area("Enter tweet")

if st.button("Predict"):
    clean = preprocess(text)
    seq = tokenizer.texts_to_sequences([clean])
    pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)
    pred = model.predict(pad)[0][0]

    if pred > 0.5:
        st.success("Positive")
    else:
        st.error("Negative ")