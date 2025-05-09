import pandas as pd

def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    movies['genres'] = movies['genres'].str.replace('|', ' ')
    return movies.dropna(), ratings.dropna()
