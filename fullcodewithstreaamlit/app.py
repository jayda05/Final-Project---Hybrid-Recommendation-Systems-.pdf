# ⚠️ DO NOT add any !wget, !pip, or shell commands here
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import warnings
warnings.filterwarnings('ignore')

# Use full dataset paths here
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    return movies.dropna(), ratings.dropna()

movies, ratings = load_data()

# Content-based filtering setup
@st.cache_resource
def setup_content():
    tfidf = TfidfVectorizer(stop_words='english')
    movies['genres'] = movies['genres'].str.replace('|', ' ')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim

cosine_sim = setup_content()

# Collaborative filtering setup
@st.cache_resource
def setup_collaborative():
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    svd = SVD()
    svd.fit(trainset)
    predictions = svd.test(testset)
    return svd, predictions, testset

svd, predictions, testset = setup_collaborative()

def get_content_recommendations(movie_id, top_n=10):
    idx = movies.index[movies['movieId'] == movie_id].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    indices = [i[0] for i in sim_scores]
    return movies.iloc[indices][['movieId', 'title', 'genres']]

def get_collab_recommendations(user_id, top_n=10):
    all_movie_ids = movies['movieId'].unique()
    rated = ratings[ratings['userId'] == user_id]['movieId']
    to_predict = np.setdiff1d(all_movie_ids, rated)
    testset = [[user_id, mid, 4.] for mid in to_predict]
    preds = svd.test(testset)
    top_preds = sorted(preds, key=lambda x: x.est, reverse=True)[:top_n]
    top_ids = [pred.iid for pred in top_preds]
    return movies[movies['movieId'].isin(top_ids)][['movieId', 'title', 'genres']]

def hybrid_recommend(user_id, movie_title, content_w=0.5, collab_w=0.5):
    try:
        movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]
        content = get_content_recommendations(movie_id)
        content['score'] = content_w

        collab = get_collab_recommendations(user_id)
        collab['score'] = collab_w

        hybrid = pd.concat([content, collab])
        hybrid = hybrid.groupby(['movieId', 'title', 'genres']).agg({'score': 'sum'}).reset_index()
        return hybrid.sort_values('score', ascending=False).head(10)
    except Exception as e:
        st.error(f"Recommendation error: {e}")
        return None

def compute_top_n_metrics(user_id, top_n=10):
    test_df = pd.DataFrame([(pred.uid, pred.iid, pred.r_ui, pred.est) for pred in predictions],
                           columns=['userId', 'movieId', 'actual', 'predicted'])
    user_df = test_df[test_df['userId'] == user_id]

    if user_df.empty:
        return None

    actual_liked = user_df[user_df['actual'] >= 4]['movieId'].values
    predicted_top = user_df.sort_values('predicted', ascending=False)['movieId'].head(top_n).values

    true_positives = len(np.intersect1d(actual_liked, predicted_top))
    precision = true_positives / top_n if top_n else 0
    recall = true_positives / len(actual_liked) if len(actual_liked) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'actual_count': len(actual_liked),
        'recommended_count': len(predicted_top)
    }

# Streamlit UI
st.set_page_config(page_title="Hybrid Recommender", layout="wide")
st.title("🎬 Hybrid Movie Recommendation System")

app_mode = st.sidebar.radio("Choose mode", ["Home", "Content-Based", "Collaborative", "Hybrid", "Evaluation"])

if app_mode == "Home":
    st.markdown("## MovieLens Dataset Overview")
    st.write(f"Total Movies: {len(movies)}")
    st.write(f"Total Ratings: {len(ratings)}")
    st.write(f"Total Users: {ratings['userId'].nunique()}")
    st.dataframe(movies.head(5))
elif app_mode == "Content-Based":
    movie = st.selectbox("Choose a movie", sorted(movies['title'].unique()))
    if st.button("Recommend Similar Movies"):
        mid = movies[movies['title'] == movie]['movieId'].values[0]
        recs = get_content_recommendations(mid)
        st.write(recs)
elif app_mode == "Collaborative":
    uid = st.number_input("Enter User ID", min_value=1, max_value=610, value=1)
    if st.button("Recommend Movies"):
        st.write(get_collab_recommendations(uid))
elif app_mode == "Hybrid":
    uid = st.number_input("User ID", min_value=1, max_value=610, value=1)
    movie = st.selectbox("Your favorite movie", sorted(movies['title'].unique()))
    cw = st.slider("Content Weight", 0.0, 1.0, 0.5)
    mw = st.slider("Collaborative Weight", 0.0, 1.0, 0.5)
    if st.button("Get Hybrid Recommendations"):
        result = hybrid_recommend(uid, movie, cw, mw)
        if result is not None:
            st.write(result)
elif app_mode == "Evaluation":
    st.subheader("Model Performance")
    st.write(f"RMSE: {accuracy.rmse(predictions):.4f}")
    st.write(f"MAE: {accuracy.mae(predictions):.4f}")

    uid = st.number_input("Evaluate for User ID:", min_value=1, max_value=610, value=1)
    if st.button("Evaluate Top-N Precision/Recall"):
        metrics = compute_top_n_metrics(uid)
        if metrics:
            st.write(f"Precision@10: {metrics['precision']:.2f}")
            st.write(f"Recall@10: {metrics['recall']:.2f}")
            st.write(f"F1 Score: {metrics['f1_score']:.2f}")
            st.write(f"Actual liked movies: {metrics['actual_count']}")
            st.write(f"Movies recommended: {metrics['recommended_count']}")
        else:
            st.warning("This user has no valid ratings in the test set.")
