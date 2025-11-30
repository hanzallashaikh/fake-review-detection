import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from textblob import TextBlob
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="Fake Review Detector", page_icon="magnifying glass", layout="centered")

@st.cache_resource
def load_artifacts():
    return (joblib.load('best_model.pkl'),
            joblib.load('tfidf_vectorizer.pkl'),
            joblib.load('category_columns.pkl'))

model, tfidf, cat_columns = load_artifacts()

def clean_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z\s]', '', text.lower())).strip()

def create_features(text, rating, category):
    clean = clean_text(text)
    length = float(len(text))
    word_count = float(len(text.split()))
    avg_word_len = float(np.mean([len(w) for w in text.split()] or [0]))
    sentiment = float(TextBlob(text).sentiment.polarity)
    rating_dev = float(abs(rating - 4.0))

    num_features = np.array([[rating, length, word_count, avg_word_len, sentiment, rating_dev]], dtype=np.float64)
    cat_dummy = pd.get_dummies(pd.DataFrame({'category': [category]}), prefix='cat')
    cat_dummy = cat_dummy.reindex(columns=cat_columns, fill_value=0).astype(np.float64)
    num_cat_sparse = csr_matrix(np.hstack([num_features, cat_dummy.values]))
    text_vec = tfidf.transform([clean])
    return hstack([text_vec, num_cat_sparse])

# ——— BEAUTIFUL UI ———
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #667eea, #764ba2); padding: 2rem; border-radius: 20px; min-height: 100vh;}
    .title {font-size: 4.5rem; font-weight: 900; text-align: center;
            background: linear-gradient(90deg, #ff9a9e, #fad0c4);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .subtitle {text-align: center; color: white; font-size: 1.5rem;}
    .stButton>button {background: white; color: #667eea; font-weight: bold; border-radius: 15px; height: 60px;}
    .result {padding: 2rem; border-radius: 20px; text-align: center; font-size: 2.8rem; font-weight: bold;}
    .fake {background: #ffe0e0; border: 6px solid #ff5252; color: #c62828;}
    .genuine {background: #e8f5e8; border: 6px solid #4caf50; color: #2e7d32;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">Fake Review Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered • 95%+ Accuracy • Instant Results</p>', unsafe_allow_html=True)

col1, col2 = st.columns([2,1])
with col1:
    review = st.text_area("Enter Review Text", height=200, placeholder="Paste any review here...", key="review_input")
with col2:
    rating = st.slider("Rating", 1.0, 5.0, 4.0, 0.5)
    category = st.selectbox("Category", [
        'Home_and_Kitchen_5','Electronics_5','Books_5','Clothing_Shoes_and_Jewelry_5',
        'Toys_and_Games_5','Sports_and_Outdoors_5','Pet_Supplies_5','Movies_and_TV_5',
        'Tools_and_Home_Improvement_5','Kindle_Store_5'])

st.markdown("#### Quick Test")
c1, c2 = st.columns(2)
with c1:
    if st.button("Genuine Example"):
        st.session_state.review_input = "Great product! Fast delivery and works perfectly."
        st.rerun()
with c2:
    if st.button("Fake Example"):
        st.session_state.review_input = "amazing best perfect love love love wow"
        st.rerun()

if st.button("ANALYZE REVIEW", type="primary"):
    if not review.strip():
        st.error("Please enter a review!")
    else:
        with st.spinner("Analyzing..."):
            X = create_features(review, rating, category)
            proba = float(model.predict_proba(X)[0][1])
            pred = "FAKE" if proba >= 0.5 else "GENUINE"

        st.markdown("### RESULT")
        if pred == "FAKE":
            st.markdown('<div class="result fake">FAKE REVIEW DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result genuine">GENUINE REVIEW</div>', unsafe_allow_html=True)
        st.progress(proba)
        st.markdown(f"**Fake Probability: {proba:.1%}**")
