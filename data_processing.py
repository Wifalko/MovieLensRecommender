from DataReader import file_reader
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import json

def get_recommendations_for_user(user_id = 9999, neighbours_amount = 4):
    with open('user_ratings.json', 'r', encoding='utf-8') as file:
        user_input_data = json.load(file)

    user_data = pd.DataFrame(user_input_data)
    user_data['Timestamp'] = pd.to_datetime(user_data['Timestamp'])

    reader = file_reader()
    data_ratings, data_movies, data_users = reader.run()

    movies_subset = data_movies[['MovieID', 'Title', 'Genres']]
    key_values = pd.merge(data_ratings, data_movies, on='MovieID', how='inner')
    key_value = pd.concat([key_values, user_data], ignore_index=True)
    MatrixDataset = key_value.groupby(by=['UserID','Title'], as_index=False).agg({"Rating":"mean"})

    user_to_movie_df = MatrixDataset.pivot(
        index='UserID',
        columns='Title',
        values='Rating').fillna(0)
    user_to_movie_sparse_df = csr_matrix(user_to_movie_df.values)

    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(user_to_movie_sparse_df)

    user_row = user_to_movie_df.loc[user_id].values.reshape(1, -1)
    distances, indices = knn_model.kneighbors(user_row, n_neighbors=neighbours_amount)

    neighbor_indices = indices.flatten()[1:]
    neighbor_user_ids = user_to_movie_df.index[neighbor_indices]

    average_ratings = data_ratings.groupby('MovieID').agg({'Rating': 'mean'}).reset_index()
    average_ratings = pd.merge(average_ratings, movies_subset, on='MovieID', how='inner')

    recommendations_by_user = []
    
    for neighbor_id in neighbor_user_ids:
        neighbor_ratings = user_to_movie_df.loc[neighbor_id]
        recommended_titles = neighbor_ratings[neighbor_ratings > 0].index.tolist()
        
        user_recommendations = []
        for title in recommended_titles:
            movie_info = movies_subset[movies_subset['Title'] == title].iloc[0]
            rating = average_ratings[average_ratings['Title'] == title]['Rating'].values[0]
            neighbor_rating = neighbor_ratings[title]
            
            user_recommendations.append({
                'Title': title,
                'Genres': movie_info['Genres'],
                'AverageRating': float(rating),
                'UserRating': float(neighbor_rating)
            })
        
        # Sortuj rekomendacje tego użytkownika po jego ocenach
        user_recommendations = sorted(user_recommendations, 
                                   key=lambda x: (x['UserRating'], x['AverageRating']), 
                                   reverse=True)
        
        # Dodaj informacje o użytkowniku
        user_summary = {
            'UserId': int(neighbor_id),
            'SimilarityScore': float(1 - distances.flatten()[1:][list(neighbor_user_ids).index(neighbor_id)]),
            'Recommendations': user_recommendations
        }
        
        recommendations_by_user.append(user_summary)
    
    return recommendations_by_user

if __name__ == "__main__":
    user_recommendations = get_recommendations_for_user()