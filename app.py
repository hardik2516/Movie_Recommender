import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data_from_gdrive():
    """
    Load datasetCleaned.csv directly from Google Drive public URL.
    """
    # Paste your Google Drive shareable link here:
    drive_link = "https://drive.google.com/file/d/1dLEpQi4U0UOzEV0lenoy8iMRjeUydSJh/view?usp=sharing"
    
    # Extract the FILE_ID from the link
    file_id = drive_link.split('/d/')[1].split('/')[0]
    
    # Construct direct download URL
    csv_url = f'https://drive.google.com/uc?id={file_id}'
    
    try:
        df = pd.read_csv(csv_url)
    except Exception as e:
        st.error(f"Error loading dataset from Google Drive: {e}")
        return pd.DataFrame()
    
    # Parse 'cast' column for names
    def parse_names(text):
        try:
            items = ast.literal_eval(str(text))
            return " ".join(d['name'] for d in items if isinstance(d, dict) and 'name' in d)
        except:
            return ""

    df['cast_clean'] = df['cast'].astype(str).apply(parse_names)
    df = df[~df['title'].astype(str).str.startswith('#')]
    df['combined_features'] = df['cast_clean'].fillna('')
    return df


def compute_similarity_matrix(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(df['combined_features'])
    return cosine_similarity(matrix, matrix)


def recommend(title, df, cosine_sim, num=5):
    if df.empty or cosine_sim is None:
        return []
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: float(x[1]), reverse=True)
    sim_scores = sim_scores[1:num+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()


# Streamlit app UI
st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬", layout="wide")

st.markdown('<h1 style="text-align:center; color:#FF4B4B;">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888;">Find movies similar to your favorite ones (content-based filtering)</p>', unsafe_allow_html=True)
st.markdown("---")

# Load data from Google Drive
df = load_data_from_gdrive()

if df.empty:
    st.error("⚠️ Could not load data from Google Drive. Please check your link and file sharing permissions.")
else:
    cosine_sim = compute_similarity_matrix(df)

    movies = df['title'].dropna().astype(str).sort_values().tolist()
    selected_movie = st.selectbox("🎥 Choose a movie you like:", movies)
    num_recommendations = st.slider("🔢 Number of recommendations:", 1, 10, 5)

    if st.button("✨ Show Recommendations"):
        recs = recommend(selected_movie, df, cosine_sim, num_recommendations)
        if recs:
            st.subheader(f"✅ Because you like **{selected_movie}**, you might also like:")
            for rec in recs:
                st.markdown(f'🎬 {rec}')
        else:
            st.warning(f"Could not find recommendations for '{selected_movie}'.")

st.markdown("---")
st.caption("🚀 Built with Streamlit • Content-based filtering demo")
