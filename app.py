import streamlit as st
from recommender import load_data_from_gdrive, compute_similarity_matrix, recommend

# --- Page Config ---
st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬", layout="wide")

# --- Header ---
st.markdown('<h1 style="text-align:center; color:#FF4B4B;">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888;">Find similar movies based on cast similarity!</p>', unsafe_allow_html=True)
st.markdown("---")

# --- Load Dataset ---
df = load_data_from_gdrive()

if df.empty:
    st.error("⚠️ Could not load data. Please check your file or column names.")
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
            st.warning(f"Sorry, no recommendations found for **{selected_movie}**.")

# --- Footer ---
st.markdown("---")
st.caption("🚀 Built with Streamlit • Based on cast similarity only (limited dataset)")
