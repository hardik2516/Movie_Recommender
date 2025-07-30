import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data_from_gdrive():
    """
    Load datasetCleaned.csv from Google Drive or local upload.
    Only uses 'cast_clean' for similarity as the dataset lacks genres, keywords, and overview.
    """
    try:
        df = pd.read_csv("datasetCleaned.csv")  # or provide full local path if needed
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    if 'cast_clean' not in df.columns or 'title' not in df.columns:
        print("Missing required columns: 'cast_clean' or 'title'")
        return pd.DataFrame()

    # Use only available cleaned feature
    df['combined_features'] = df['cast_clean'].fillna('')
    df = df[~df['title'].astype(str).str.startswith('#')]
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
