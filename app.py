import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from textblob import TextBlob
from scipy.sparse import hstack, csr_matrix

# Load model and preprocessors
@st.cache_resource
def load_model():
    model = joblib.load('best_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    cat_columns = joblib.load('category_columns.pkl')
    return model, tfidf, cat_columns

model, tfidf, cat_columns = load_model()

# Simple text cleaning — NO NLTK AT ALL
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_features(text, rating, category):
    clean = clean_text(text)
    
    # Basic features (no NLTK, no lemmatizer)
    length = len(text)
    word_count = len(text.split())
    avg_word_len = np.mean([len(w) for w in text.split()]) if word_count > 0 else 0
    sentiment = TextBlob(text).sentiment.polarity
    rating_dev = abs(rating - 4.0)
    
    num_vals = [rating, length, word_count, avg_word_len, sentiment, rating_dev]
    num_array = np.array(num_vals).reshape(1, -1)
    
    # Category one-hot
    cat_dummy = pd.get_dummies(pd.DataFrame({'category': [category]}), prefix='cat')
    cat_dummy = cat_dummy.reindex(columns=cat_columns, fill_value=0)
    
    num_cat = np.hstack([num_array, cat_dummy.values])
    num_cat_sparse = csr_matrix(num_cat)
    
    text_vec = tfidf.transform([clean])
    final = hstack([text_vec, num_cat_sparse])
    return final

# UI
st.set_page_config(page_title="Fake Review Detector", layout="centered")
st.title("Fake Review Detector")
st.markdown("### Detect fake reviews using AI (Live Demo)")

col1, col2 = st.columns([3,1])
with col1:
    review_text = st.text_area("Enter Review Text", height=180, placeholder="I love this product so much!!! Best ever...")
with col2:
    rating = st.slider("Rating", 1.0, 5.0, 4.0, 0.5)
    category = st.selectbox("Category", [
        'Home_and_Kitchen_5', 'Clothing_Shoes_and_Jewelry_5', 'Electronics_5',
        'Sports_and_Outdoors_5', 'Toys_and_Games_5', 'Books_5'
    ])

if st.button("Check if Fake", type="primary"):
    if not review_text.strip():
        st.error("Please write a review")
    else:
        with st.spinner("Analyzing..."):
            X = create_features(review_text, rating, category)
            proba = model.predict_proba(X)[0][1]
            pred = "FAKE" if proba > 0.5 else "GENUINE"
        
        if pred == "FAKE":
            st.error(f"FAKE REVIEW DETECTED!")
        else:
            st.success(f"GENUINE REVIEW")
        
        st.progress(proba)
        st.write(f"**Fake Probability: {proba:.1%}**")
        st.write(f"**Prediction: {pred}**")
