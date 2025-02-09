import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from DataReader import file_reader
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import json
import os

reader = file_reader()
data_ratings, data_movies, data_users = reader.run()

class KNNRecommender:
    """
    Implementation of a recommendation system based on k-nearest neighbors algorithm.
    Uses similarity between users to generate movie recommendations.
    """
    def __init__(self, data_ratings: pd.DataFrame, data_movies: pd.DataFrame):
        """
        Initialize the KNN recommender.
        
        Args:
            data_ratings (pd.DataFrame): DataFrame containing movie ratings by users
                required columns: UserID, MovieID, Rating
            data_movies (pd.DataFrame): DataFrame containing movie information
                required columns: MovieID, Title, Genres
        """
        self.data_ratings = data_ratings
        self.data_movies = data_movies
        self.movies_subset = data_movies[['MovieID', 'Title', 'Genres']]
        self.user_to_movie_df = None
        self.knn_model = None

    def prepare_data(self, user_data: pd.DataFrame) -> None:
        """
        Prepares data for KNN model training by creating a user-movie matrix.
        
        Args:
            user_data (pd.DataFrame): DataFrame containing additional user ratings
                must have the same columns as data_ratings
                
        Returns:
            None - results are stored in self.user_to_movie_df
        """
        key_values = pd.merge(self.data_ratings, self.data_movies, on='MovieID', how='inner')
        key_value = pd.concat([key_values, user_data], ignore_index=True)
        matrix_dataset = key_value.groupby(by=['UserID','Title'], as_index=False).agg({"Rating":"mean"})

        self.user_to_movie_df = matrix_dataset.pivot(
            index='UserID',
            columns='Title',
            values='Rating'
        ).fillna(0)

    def train(self, metric: str = 'cosine', algorithm: str = 'brute') -> None:
        """
        Trains the KNN model on the prepared data.
        
        Args:
            metric (str): Distance metric used by KNN (default: 'cosine')
            algorithm (str): Algorithm used for finding neighbors (default: 'brute')
                
        Returns:
            None - trained model is stored in self.knn_model
        """
        user_to_movie_sparse_df = csr_matrix(self.user_to_movie_df.values)
        self.knn_model = NearestNeighbors(metric=metric, algorithm=algorithm)
        self.knn_model.fit(user_to_movie_sparse_df)

    def get_recommendations(self, user_id: int, neighbours_amount: int = 4):
        """
        Generates recommendations for a specific user.
        
        Args:
            user_id (int): ID of the user to generate recommendations for
            neighbours_amount (int): Number of neighbors to find (default: 4)
            
        Returns:
            list: List of recommendations grouped by similar users, each containing:
                - UserId: ID of the similar user
                - SimilarityScore: Similarity score with the target user
                - Recommendations: List of recommended movies with ratings and genres
        """
        user_row = self.user_to_movie_df.loc[user_id].values.reshape(1, -1)
        distances, indices = self.knn_model.kneighbors(user_row, n_neighbors=neighbours_amount)

        neighbor_indices = indices.flatten()[1:]
        neighbor_user_ids = self.user_to_movie_df.index[neighbor_indices]

        average_ratings = self.data_ratings.groupby('MovieID').agg({'Rating': 'mean'}).reset_index()
        average_ratings = pd.merge(average_ratings, self.movies_subset, on='MovieID', how='inner')

        recommendations_by_user = []
        
        for neighbor_id in neighbor_user_ids:
            neighbor_ratings = self.user_to_movie_df.loc[neighbor_id]
            recommended_titles = neighbor_ratings[neighbor_ratings > 0].index.tolist()
            
            user_recommendations = []
            for title in recommended_titles:
                movie_info = self.movies_subset[self.movies_subset['Title'] == title].iloc[0]
                rating = average_ratings[average_ratings['Title'] == title]['Rating'].values[0]
                neighbor_rating = neighbor_ratings[title]
                
                user_recommendations.append({
                    'Title': title,
                    'Genres': movie_info['Genres'],
                    'AverageRating': float(rating),
                    'UserRating': float(neighbor_rating)
                })

            user_recommendations = sorted(
                user_recommendations, 
                key=lambda x: (x['UserRating'], x['AverageRating']), 
                reverse=True
            )
            
            user_summary = {
                'UserId': int(neighbor_id),
                'SimilarityScore': float(1 - distances.flatten()[1:][list(neighbor_user_ids).index(neighbor_id)]),
                'Recommendations': user_recommendations
            }
            
            recommendations_by_user.append(user_summary)
        
        return recommendations_by_user
    

class NCF(nn.Module):
    def __init__(self, data_ratings, data_movies, latent_dim=16, layers=[32, 16, 8]):
        super(NCF, self).__init__()
        
        self.data_ratings = data_ratings
        self.data_movies = data_movies
        
        self.user_embedding = None
        self.movie_embedding = None
        self.fc_layers = None
        self.output_layer = None
        
        self.latent_dim = latent_dim
        self.layers = layers
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _init_model(self):
        self.user_embedding = nn.Embedding(len(self.user_ids), self.latent_dim)
        self.movie_embedding = nn.Embedding(len(self.movie_ids), self.latent_dim)

        self.fc_layers = nn.ModuleList()
        input_dim = self.latent_dim * 2
        
        for layer_size in self.layers:
            self.fc_layers.append(nn.Linear(input_dim, layer_size))
            input_dim = layer_size
            
        self.output_layer = nn.Linear(input_dim, 1)
        
        self.to(self.device)
        
    def forward(self, user_indices, movie_indices):
        user_features = self.user_embedding(user_indices)
        movie_features = self.movie_embedding(movie_indices)
        x = torch.cat([user_features, movie_features], dim=-1)
        
        for layer in self.fc_layers:
            x = nn.ReLU()(layer(x))
            
        return torch.sigmoid(self.output_layer(x))
    
    def prepare_data(self, user_data):
        """
        Prepare data for training, always including user_data
        """
        self.all_ratings = pd.concat([self.data_ratings, user_data], ignore_index=True)
        
        self.user_ids = self.all_ratings['UserID'].unique()
        self.movie_ids = self.data_movies['MovieID'].unique()
        
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        
        users = self.all_ratings['UserID'].map(self.user_to_idx).values
        movies = self.all_ratings['MovieID'].map(self.movie_to_idx).values
        ratings = self.all_ratings['Rating'].values / 5.0 

        self._init_model()
        
        return train_test_split(users, movies, ratings, test_size=0.2, random_state=42)
    
    def train_model(self, user_data, epochs=10, batch_size=64):
        """
        Train the model, requiring user_data
        """
        X_train_user, X_test_user, X_train_movie, X_test_movie, y_train, y_test = self.prepare_data(user_data)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters())
        
        n_samples = len(X_train_user)
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(0, n_samples, batch_size):
                batch_users = torch.LongTensor(X_train_user[i:i+batch_size]).to(self.device)
                batch_movies = torch.LongTensor(X_train_movie[i:i+batch_size]).to(self.device)
                batch_ratings = torch.FloatTensor(y_train[i:i+batch_size]).to(self.device)
                
                optimizer.zero_grad()
                predictions = self(batch_users, batch_movies).squeeze()
                loss = criterion(predictions, batch_ratings)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (n_samples // batch_size)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def get_recommendations(self, user_id, n_recommendations=30):
        """
        Get recommendations for a specific user
        """
        self.eval()
        
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        user_tensor = torch.LongTensor([user_idx] * len(self.movie_ids)).to(self.device)
        movie_tensor = torch.LongTensor(range(len(self.movie_ids))).to(self.device)
        
        with torch.no_grad():
            predictions = self(user_tensor, movie_tensor).cpu().numpy()

        rated_movies = set(self.all_ratings[self.all_ratings['UserID'] == user_id]['MovieID'].values)
        available_movies = [(idx, pred) for idx, pred in enumerate(predictions.flatten()) 
                          if self.movie_ids[idx] not in rated_movies]
        
        available_movies.sort(key=lambda x: x[1], reverse=True)
        top_predictions = available_movies[:n_recommendations]
        
        user_summary = {
            'UserId': user_id,
            'SimilarityScore': 1.0,
            'Recommendations': []
        }
        
        for idx, pred in top_predictions:
            movie_id = self.movie_ids[idx]
            movie_info = self.data_movies[self.data_movies['MovieID'] == movie_id].iloc[0]
            avg_rating = self.data_ratings[self.data_ratings['MovieID'] == movie_id]['Rating'].mean()
            
            user_summary['Recommendations'].append({
                'Title': movie_info['Title'],
                'Genres': movie_info['Genres'],
                'UserRating': float(pred * 5),
                'AverageRating': float(avg_rating)
            })
        
        return [user_summary]


class MovieRecommender:
    """
    High-level movie recommender class that combines NCF model functionality
    with data preprocessing and recommendation generation.
    """
    def __init__(self, ratings_df, movies_df):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        
        self.user_ids = ratings_df['UserID'].unique()
        self.movie_ids = ratings_df['MovieID'].unique()
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        
        self.model = NCF(
            num_users=len(self.user_ids),
            num_items=len(self.movie_ids),
            latent_dim=16,
            layers=[32, 16, 8]
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def prepare_data(self):
        """
        Prepare data for model training
        
        Returns:
            tuple: Train-test split of processed data
        """
        users = self.ratings_df['UserID'].map(self.user_to_idx).values
        movies = self.ratings_df['MovieID'].map(self.movie_to_idx).values
        ratings = self.ratings_df['Rating'].values / 5.0  
        
        return train_test_split(
            users, movies, ratings,
            test_size=0.2,
            random_state=42
        )
        
    def train(self, epochs=10, batch_size=64):
        """
        Train the recommendation model
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Size of training batches
        """
        X_train_user, X_test_user, X_train_movie, X_test_movie, y_train, y_test = self.prepare_data()
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())
        
        n_samples = len(X_train_user)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for i in range(0, n_samples, batch_size):
                batch_users = torch.LongTensor(X_train_user[i:i+batch_size]).to(self.device)
                batch_movies = torch.LongTensor(X_train_movie[i:i+batch_size]).to(self.device)
                batch_ratings = torch.FloatTensor(y_train[i:i+batch_size]).to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_users, batch_movies).squeeze()
                loss = criterion(predictions, batch_ratings)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_samples:.4f}")
    
    def get_recommendations_for_user(self, user_id, n_recommendations=10):
        """
        Generates movie recommendations for a specific user using the trained model.
        
        Args:
            user_id (int): ID of the user to generate recommendations for
            n_recommendations (int): Number of movies to recommend (default: 10)
            
        Returns:
            list: List of dictionaries containing recommended movies with:
                - MovieID (int): ID of the recommended movie
                - Title (str): Title of the movie
                - Genres (str): Genres of the movie
                - PredictedRating (float): Predicted rating for this user
                
        Returns empty list if user_id is not found in the training data.
        """
        self.model.eval()
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        user_tensor = torch.LongTensor([user_idx] * len(self.movie_ids)).to(self.device)
        movie_tensor = torch.LongTensor(list(range(len(self.movie_ids)))).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(user_tensor, movie_tensor).cpu().numpy()
        
        movie_indices = np.argsort(predictions.flatten())[-n_recommendations:][::-1]
        recommendations = []
        
        for idx in movie_indices:
            movie_id = self.movie_ids[idx]
            movie_info = self.movies_df[self.movies_df['MovieID'] == movie_id].iloc[0]
            recommendations.append({
                'MovieID': int(movie_id),
                'Title': movie_info['Title'],
                'Genres': movie_info['Genres'],
                'PredictedRating': float(predictions[idx])
            })
        
        return recommendations

def save_model(model, path='recommender_model.pth'):
    torch.save(model.state_dict(), path)

def load_model(model, path='recommender_model.pth'):
    model.load_state_dict(torch.load(path))
    return model