

from surprise import Dataset, Reader, SVD
import numpy as np

def setup_collaborative_model(ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD()
    svd.fit(trainset)
    return svd

def get_collab_recommendations(movies, ratings, svd, user_id, top_n=10):
    all_movie_ids = movies['movieId'].unique()
    rated = ratings[ratings['userId'] == user_id]['movieId']
    to_predict = np.setdiff1d(all_movie_ids, rated)
    testset = [[user_id, mid, 4.] for mid in to_predict]
    preds = svd.test(testset)
    top_preds = sorted(preds, key=lambda x: x.est, reverse=True)[:top_n]
    top_ids = [pred.iid for pred in top_preds]
    return movies[movies['movieId'].isin(top_ids)][['movieId', 'title', 'genres']]
