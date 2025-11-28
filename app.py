import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from scipy.sparse import hstack, csr_matrix

# Load everything
model = joblib.load('best_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
num_features = joblib.load('numerical_features.pkl')
cat_columns = joblib.load('category_columns.pkl')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

def create_features(text, rating, category):
    clean = preprocess_text(text)
    
    # Numerical features
    length = len(text)
    word_count = len(text.split())
    avg_word_len = np.mean([len(w) for w in text.split()]) if word_count > 0 else 0
    sentiment = TextBlob(text).sentiment.polarity
    rating_dev = abs(rating - 4.0)  # approximate mean
    
    num_vals = [rating, length, word_count, avg_word_len, sentiment, rating_dev]
    num_array = np.array(num_vals).reshape(1, -1)
    
    # Category one-hot
    cat_df = pd.DataFrame({'category': [category]})
    cat_dummy = pd.get_dummies(cat_df, prefix='cat')
    cat_dummy = cat_dummy.reindex(columns=cat_columns, fill_value=0)
    
    # Combine num + cat
    num_cat = np.hstack([num_array, cat_dummy.values])
    num_cat_sparse = csr_matrix(num_cat)
    
    # Text vector + final
    text_vec = tfidf.transform([clean])
    final_features = hstack([text_vec, num_cat_sparse])
    return final_features

# Streamlit UI
st.set_page_config(page_title="Fake Review Detector", layout="centered")
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
        st.error("Please enter a review text")
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
