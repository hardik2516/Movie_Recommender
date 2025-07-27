import streamlit as st
from recommender import load_data_from_gdrive, compute_similarity_matrix, recommend

# --- Streamlit UI ---
st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬", layout="wide")

st.markdown('<h1 style="text-align:center; color:#FF4B4B;">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888;">Find movies similar to your favorite ones using cast, genre, and plot!</p>', unsafe_allow_html=True)
st.markdown("---")

# Load and prepare data
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
st.caption("🚀 Built with Streamlit • Enhanced with genres, cast, and plot-based filtering.")
