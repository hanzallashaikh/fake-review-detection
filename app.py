import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from textblob import TextBlob
from scipy.sparse import hstack, csr_matrix

# Move this to the VERY TOP – fixes the StreamlitAPIException
st.set_page_config(page_title="Fake Review Detector", layout="centered")

# Load model and preprocessors
@st.cache_resource
def load_model():
    model = joblib.load('best_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    cat_columns = joblib.load('category_columns.pkl')
    return model, tfidf, cat_columns

model, tfidf, cat_columns = load_model()

# Simple text cleaning — NO NLTK
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_features(text, rating, category):
    clean = clean_text(text)
    
    # Basic features
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
st.title("Fake Review Detector")
st.markdown("Detect whether an online review is **genuine** or **fake** using AI")

col1, col2 = st.columns([3,1])
with col1:
    review_text = st.text_area("Review Text", height=180, placeholder="Paste the review here...")
with col2:
    rating = st.slider("Rating", 1.0, 5.0, 3.0, 0.5)
    category = st.selectbox("Product Category", [
        'Home_and_Kitchen_5', 'Sports_and_Outdoors_5', 'Electronics_5',
        'Movies_and_TV_5', 'Tools_and_Home_Improvement_5', 'Pet_Supplies_5',
        'Kindle_Store_5', 'Books_5', 'Toys_and_Games_5', 'Clothing_Shoes_and_Jewelry_5'
    ])

if st.button("Analyze Review", type="primary"):
    if not review_text.strip():
        st.error("Please enter a review")
    else:
        with st.spinner("Analyzing..."):
            features = create_features(review_text, rating, category)
            proba = model.predict_proba(features)[0][1]
            pred = int(proba >= 0.5)
        
        st.markdown("### Result")
        if pred == 1:
            st.error(f"**FAKE REVIEW DETECTED**")
        else:
            st.success(f"**GENUINE REVIEW**")
        
        st.progress(proba)
        st.write(f"Fake probability: **{proba:.1%}**")
