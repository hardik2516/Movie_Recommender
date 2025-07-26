import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data_from_gdrive():
    """
    Load datasetCleaned.csv directly from a public Google Drive link.
    """
    # Paste your Google Drive shareable link for datasetCleaned.csv here:
    drive_link = "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
    # Replace FILE_ID with your actual Google Drive file ID

    # Extract FILE_ID from the link
    try:
        file_id = drive_link.split('/d/')[1].split('/')[0]
        csv_url = f'https://drive.google.com/uc?id={file_id}'
        df = pd.read_csv(csv_url)
    except Exception as e:
        print(f"Error loading dataset from Google Drive: {e}")
        return pd.DataFrame()

    # Parse 'cast' column (stringified list of dicts)
    def parse_names(text):
        try:
            items = ast.literal_eval(str(text))
            return " ".join(d['name'] for d in items if isinstance(d, dict) and 'name' in d)
        except Exception as e:
            print(f"Error parsing cast: {e}")
            return ""

    df['cast_clean'] = df['cast'].astype(str).apply(parse_names)
    df = df[~df['title'].astype(str).str.startswith('#')]
    df['combined_features'] = df['cast_clean'].fillna('')
    return df

def compute_similarity_matrix(df):
    """
    Build cosine similarity matrix using TF-IDF vectorization.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(df['combined_features'])
    return cosine_similarity(matrix, matrix)

def recommend(title, df, cosine_sim, num=5):
    """
    Recommend movies similar to the given title.
    """
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
