import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data_from_gdrive():
    """
    Load dataset from Google Drive (public share link).
    Combines cast, genres, keywords, and overview for hybrid similarity.
    """
    drive_link = "https://drive.google.com/file/d/1dLEpQi4U0UOzEV0lenoy8iMRjeUydSJh/view?usp=sharing"
    file_id = drive_link.split('/d/')[1].split('/')[0]
    csv_url = f'https://drive.google.com/uc?id={file_id}'

    try:
        df = pd.read_csv(csv_url)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    # Parse dictionary-like columns
    def parse_names(text):
        try:
            items = ast.literal_eval(str(text))
            return " ".join(d['name'] for d in items if isinstance(d, dict) and 'name' in d)
        except:
            return ""

    df['cast_clean'] = df['cast'].apply(parse_names)
    df['genres_clean'] = df['genres'].apply(parse_names) if 'genres' in df.columns else ''
    df['keywords_clean'] = df['keywords'].apply(parse_names) if 'keywords' in df.columns else ''
    df['overview_clean'] = df['overview'].astype(str)

    df['combined_features'] = (
        df['cast_clean'].fillna('') + " " +
        df['genres_clean'].fillna('') + " " +
        df['keywords_clean'].fillna('') + " " +
        df['overview_clean'].fillna('')
    )

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
