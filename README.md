# 🎬 Movie Recommendation System

Welcome to this **Content-Based Movie Recommendation App**! This project leverages advanced NLP techniques to deliver tailored movie recommendations based on cast similarity. It’s built with clean, modular code, modern best practices, and is easily deployable—making it a fantastic demonstration of both **software engineering** and **data science** skills.

## 🚀 Features

- **Instant Movie Recommendations** based on cast similarity using TF-IDF and cosine similarity.
- **Intuitive Streamlit Web App**: Clean, modern, and responsive user interface.
- **Automatic Data Loading**: Supports loading from a public Google Drive CSV or local repository file—no file path headaches!
- **Fully Cloud-Ready**: Deploy on Streamlit Cloud with a single click; all dependencies managed via `requirements.txt`.
- **Robust Data Pipeline**: Safely parses complex stringified lists from CSV, cleans data, and excludes corrupted rows automatically.
- **Customizable UI**: Highlighted recommendations, adjustable suggestion count, stylish indicators, and error feedback.

## ✨ Live Demo

Experience the app instantly!  
> **[Click here to try the Movie Recommendation System (Streamlit Cloud)](https://movierecommender-ofusgivuqmzwrfaihrnvse.streamlit.app/)**

## 🛠️ Tech Stack

- **Python 3.10+**
- [Streamlit](https://streamlit.io) – for interactive web UI
- [scikit-learn](https://scikit-learn.org/) – NLP & machine learning
- [pandas](https://pandas.pydata.org/) – robust data handling
- Google Drive / GitHub – for dataset hosting

## 📂 Project Structure

```
movie_recommender/
│
├── app.py                  # Main Streamlit application
├── recommender.py          # Data loading, similarity, recommendation logic
├── requirements.txt        # Dependency list for cloud/local setup
└── datasetCleaned.csv      # (Optional; auto-loaded or from Google Drive)
```

## ⚡ How it Works

1. **Data Source**:  
   Loads `datasetCleaned.csv` directly from Google Drive (public share link) or optionally from local directory.
2. **Preprocessing**:  
   - Parses the `cast` column (stringified JSON) into clean names.
   - Removes comments/bad data.
3. **Feature Engineering**:  
   - Builds a text feature matrix from cast names.
   - Uses TF-IDF vectorization.
4. **Similarity Calculation**:  
   - Computes cosine similarity for all movies.
5. **Interactive Recommendation**:  
   - User selects a movie and number of suggestions.
   - Recommends most similar movies based on cast overlap.

## 🚧 Quick Start (Local)

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/movie_recommender.git
   cd movie_recommender
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App:**

   ```bash
   streamlit run app.py
   ```

   (App will auto-load the dataset from Google Drive—customize the share link in code as needed.)

## ☁️ Deploy on Streamlit Cloud

1. **Push your code and `requirements.txt` to GitHub.**
2. **Connect your repo to Streamlit Cloud:**  
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click New App, select your repo & main file, and deploy—no setup required!
   - Make sure your Google Drive CSV is set to "Anyone with the link" (Viewer).

## 🎯 Why This Project Stands Out (for Recruitment)

- **Full-Stack Data Product:**  
  Showcases end-to-end skills: data wrangling, ML system design, API consumption, and modern Python web app shipment.
- **Clean Engineering:**  
  Modular, well-commented code, separate logic/UI, robust error and edge-case handling.
- **Cloud-Native Ready:**  
  Zero-configuration deployment capability, full reproducibility with `requirements.txt`, and external dataset integration.
- **Business-Realistic Use Case:**  
  Focuses on movie similarity—a classic content recommendation scenario in industry.
- **Scalable and Extensible:**  
  Architecture supports easy extensions: genre-based filtering, image integration, or collaborative filtering models.

## 📢 Recruiters: Key Takeaways

- Demonstrates solid Python, ML, pandas, and cloud deployment skills.
- Proves ability to ingest, clean, and enrich real-world, messy data.
- Highlights modern approaches to reproducibility and app sharing.
- Offers a snappy, attractive UI and strong UX—immediately accessible for non-technical evaluators.

## 💡 Customization Ideas

- Swap out the Google Drive CSV URL to test with new datasets.
- Integrate more features (e.g., genre, overview summaries).
- Connect external APIs (OMDb, TMDb) for richer recommendations.
- Enhance visuals with cover images or movie posters.

## 📝 Example requirements.txt

```
streamlit
pandas
scikit-learn
```

## 👤 About the Creator

Built with a passion for data science, Python, and modern web tools.  
Let’s connect on [LinkedIn](www.linkedin.com/in/hardik-gautam-53646132a)! ⭐ If you enjoy the project, leave a star on GitHub!

> **Ready for your next ML/Product/Data role! This project highlights both my technical depth and my commitment to user-centered design and modern engineering best practices.**

**Feel free to copy-paste and adapt this README for your repo. Recruiters will see both your technical skill and your product sense at a glance!**
