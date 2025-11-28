import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from textblob import TextBlob
from scipy.sparse import hstack, csr_matrix

# MUST BE FIRST LINE
st.set_page_config(page_title="Fake Review Detector", layout="centered")

# Load everything with cache
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    cat_cols = joblib.load('category_columns.pkl')
    return model, tfidf, cat_cols

model, tfidf, cat_columns = load_artifacts()

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Create features — FORCED float64 everywhere
def create_features(text, rating, category):
    clean = clean_text(text)
    
    # Numerical features — ALL float64
    length = float(len(text))
    word_count = float(len(text.split()))
    avg_word_len = float(np.mean([len(w) for w in text.split()]) if word_count > 0 else 0)
    sentiment = float(TextBlob(text).sentiment.polarity)
    rating_dev = float(abs(rating - 4.0))
    
    num_features = np.array([[rating, length, word_count, avg_word_len, sentiment, rating_dev]], dtype=np.float64)
    
    # Category one-hot — forced float64
    cat_dummy = pd.get_dummies(pd.DataFrame({'category': [category]}), prefix='cat')
    cat_dummy = cat_dummy.reindex(columns=cat_columns, fill_value=0).astype(np.float64)
    
    # Combine numerical + categorical
    num_cat_dense = np.hstack([num_features, cat_dummy.values])
    num_cat_sparse = csr_matrix(num_cat_dense)
    
    # Text TF-IDF
    text_vec = tfidf.transform([clean])
    
    # Final matrix
    final_features = hstack([text_vec, num_cat_sparse])
    return final_features

# UI
st.title("Fake Review Detector")
st.markdown("### Instantly detect fake vs genuine product reviews")

col1, col2 = st.columns([3,1])
with col1:
    review_text = st.text_area("Enter Review", height=180, placeholder="This is the best product ever!!!")
with col2:
    rating = st.slider("Rating", 1.0, 5.0, 4.0, 0.5)
    category = st.selectbox("Category", [
        'Home_and_Kitchen_5', 'Clothing_Shoes_and_Jewelry_5', 'Electronics_5',
        'Sports_and_Outdoors_5', 'Toys_and_Games_5', 'Books_5', 'Pet_Supplies_5'
    ])

if st.button("Analyze Review", type="primary"):
    if not review_text.strip():
        st.error("Please enter a review")
    else:
        with st.spinner("Analyzing..."):
            X = create_features(review_text, rating, category)
            proba = model.predict_proba(X)[0][1]
            pred = "FAKE" if proba >= 0.5 else "GENUINE"
        
        st.markdown("### Result")
        if pred == "FAKE":
            st.error("**FAKE REVIEW DETECTED**")
        else:
            st.success("**GENUINE REVIEW**")
        
        st.progress(proba)
        st.write(f"**Fake Probability: {proba:.1%}**")
