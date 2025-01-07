from DataReader import file_reader
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

reader = file_reader()
data_ratings, data_movies, data_users = reader.run()

movies_subset = data_movies[['MovieID', 'Title']]
important_things = pd.merge(data_ratings, data_movies, on='MovieID', how='inner')
MatrixDataset = important_things.groupby(by=['UserID','Title'], as_index=False).agg({"Rating":"mean"})

user_to_movie_df = MatrixDataset.pivot(
    index='UserID',
     columns='Title',
      values='Rating').fillna(0)

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
user_to_movie_sparse_df = csr_matrix(user_to_movie_df.values)
knn_model.fit(user_to_movie_sparse_df)


distances, indices = knn_model.kneighbors(user_to_movie_sparse_df[8], n_neighbors=5)

movie_counts = data_ratings['MovieID'].value_counts()
movie_counts = movie_counts.drop_duplicates()

top_10_movies = movie_counts.head(10)

for movie_id, count in top_10_movies.items():
    movie_name = data_movies.loc[data_movies['MovieID'] == movie_id, 'Title'].values
    movie_name = movie_name[0] if len(movie_name) > 0 else "Nieznany film"
    print(f"Film {movie_name} został oceniony {count} razy.")

