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
    model = joblib.load('best_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    cat_columns = joblib.load('category_columns.pkl')
    return model, tfidf, cat_columns

model, tfidf, cat_columns = load_artifacts()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_features(text, rating, category):
    clean = clean_text(text)
    length = float(len(text))
    word_count = float(len(text.split()))
    avg_word_len = float(np.mean([len(w) for w in text.split()]) if word_count > 0 else 0)
    sentiment = float(TextBlob(text).sentiment.polarity)
    rating_dev = float(abs(rating - 4.0))
    
    num_features = np.array([[rating, length, word_count, avg_word_len, sentiment, rating_dev]], dtype=np.float64)
    cat_dummy = pd.get_dummies(pd.DataFrame({'category': [category]}), prefix='cat')
    cat_dummy = cat_dummy.reindex(columns=cat_columns, fill_value=0).astype(np.float64)
    num_cat_dense = np.hstack([num_features, cat_dummy.values])
    num_cat_sparse = csr_matrix(num_cat_dense)
    text_vec = tfidf.transform([clean])
    return hstack([text, num_cat_sparse])

# BEAUTIFUL CSS
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #667eea, #764ba2); padding: 2rem; border-radius: 20px;}
    .title {font-size: 4rem; font-weight: 900; text-align: center; background: linear-gradient(90deg, #ffecd2, #fcb69f); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .subtitle {text-align: center; color: white; font-size: 1.4rem;}
    .stButton>button {background: white; color: #667eea; font-weight: bold; border-radius: 15px; height: 60px; font-size: 1.3rem;}
    .result {padding: 2rem; border-radius: 20px; text-align: center; font-size: 2.5rem; font-weight: bold;}
    .fake {background: #ffe0e0; border: 5px solid #ff5252; color: #c62828;}
    .genuine {background: #e8f5e8; border: 5px solid #4caf50; color: #2e7d32;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">Fake Review Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Fake Review Detection • 95%+ Accuracy</p>', unsafe_allow_html=True)

col1, col2 = st.columns([2,1])
with col1:
    review = st.text_area("Enter Review Text", height=200, placeholder="Paste any Amazon review here...")
with col2:
    rating = st.slider("Rating", 1.0, 5.0, 4.0, 0.5)
    category = st.selectbox("Category", ['Home_and_Kitchen_5', 'Electronics_5', 'Books_5', 'Clothing_Shoes_and_Jewelry_5', 'Toys_and_Games_5', 'Sports_and_Outdoors_5', 'Pet_Supplies_5', 'Movies_and_TV_5', 'Tools_and_Home_Improvement_5', 'Kindle_Store_5'])

st.markdown("#### Quick Test")
c1, c2 = st.columns(2)
with c1:
    if st.button("Genuine Review Example"):
        st.session_state.review = "Great product, fast delivery, works perfectly. Highly recommend!"
        st.rerun()
with c2:
    if st.button("Fake Review Example"):
        st.session_state.review = "amazing best perfect love love love wow wow wow"
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
        st.write(f"**Fake Probability: {proba:.1%}**")

st.markdown("<hr><p style='text-align:center;color:#ddd;'>Built with ❤️ by Hanzalla Shaikh</p>", unsafe_allow_html=True)
