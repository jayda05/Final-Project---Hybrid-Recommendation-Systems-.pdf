import pandas as pd

def hybrid_recommend(movies, content_sim, svd, user_id, movie_title, content_w=0.5, collab_w=0.5):
    try:
        movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]
        content = get_content_recommendations(movies, content_sim, movie_id)
        content['score'] = content_w

        collab = get_collab_recommendations(movies, ratings, svd, user_id)
        collab['score'] = collab_w

        hybrid = pd.concat([content, collab])
        return hybrid.groupby(['movieId', 'title', 'genres']).agg({'score': 'sum'}).reset_index()
    except Exception as e:
        raise Exception(f"Recommendation error: {e}")
