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
    Klasa implementująca system rekomendacji oparty na algorytmie k-najbliższych sąsiadów.
    """
    def __init__(self, data_ratings: pd.DataFrame, data_movies: pd.DataFrame):
        """
        Inicjalizacja rekomendatora KNN.
        
        Args:
            data_ratings: DataFrame z ocenami filmów
            data_movies: DataFrame z informacjami o filmach
        """
        self.data_ratings = data_ratings
        self.data_movies = data_movies
        self.movies_subset = data_movies[['MovieID', 'Title', 'Genres']]
        self.user_to_movie_df = None
        self.knn_model = None

    def prepare_data(self, user_data: pd.DataFrame) -> None:
        """
        Przygotowuje dane do treningu modelu KNN.
        
        Args:
            user_data: DataFrame z ocenami użytkownika
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
        Trenuje model KNN na przygotowanych danych.
        
        Args:
            metric: metryka odległości używana przez KNN
            algorithm: algorytm używany do znajdowania sąsiadów
        """
        user_to_movie_sparse_df = csr_matrix(self.user_to_movie_df.values)
        self.knn_model = NearestNeighbors(metric=metric, algorithm=algorithm)
        self.knn_model.fit(user_to_movie_sparse_df)

    def get_recommendations(self, user_id: int, neighbours_amount: int = 4):
        """
        Generuje rekomendacje dla określonego użytkownika.
        
        Args:
            user_id: ID użytkownika
            neighbours_amount: liczba sąsiadów do znalezienia
            
        Returns:
            Lista rekomendacji pogrupowanych według podobnych użytkowników
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
        """
        Initialize the NCF model with data
        
        Args:
            data_ratings: DataFrame with ratings data
            data_movies: DataFrame with movies data
            latent_dim: dimension of the embedding layers
            layers: list of layer sizes for the neural network
        """
        self.data_ratings = data_ratings
        self.data_movies = data_movies
        self.recommender = None
        
        # Get unique users and movies
        self.user_ids = data_ratings['UserID'].unique()
        self.movie_ids = data_movies['MovieID'].unique()
        
        # Initialize the neural network
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(len(self.user_ids), latent_dim)
        self.movie_embedding = nn.Embedding(len(self.movie_ids), latent_dim)
        
        # Create the fully connected layers
        self.fc_layers = nn.ModuleList()
        input_dim = latent_dim * 2
        for layer_size in layers:
            self.fc_layers.append(nn.Linear(input_dim, layer_size))
            input_dim = layer_size
        self.output_layer = nn.Linear(input_dim, 1)
        
        # Create mappings for user and movie IDs
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, user_indices, movie_indices):
        user_features = self.user_embedding(user_indices)
        movie_features = self.movie_embedding(movie_indices)
        x = torch.cat([user_features, movie_features], dim=-1)
        for layer in self.fc_layers:
            x = nn.ReLU()(layer(x))
        return torch.sigmoid(self.output_layer(x))
        
    def prepare_data(self, user_data):
        """Prepare data for the NCF model by combining historical and user data"""
        self.all_ratings = pd.concat([self.data_ratings, user_data], ignore_index=True)
        
        # Update user and movie mappings
        self.user_ids = self.all_ratings['UserID'].unique()
        self.movie_ids = self.data_movies['MovieID'].unique()
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        
        # Prepare training data
        users = self.all_ratings['UserID'].map(self.user_to_idx).values
        movies = self.all_ratings['MovieID'].map(self.movie_to_idx).values
        ratings = self.all_ratings['Rating'].values / 5.0
        
        return train_test_split(users, movies, ratings, test_size=0.2, random_state=42)
        
    def train(self, epochs=10, batch_size=64):
        """Train the NCF model"""
        X_train_user, X_test_user, X_train_movie, X_test_movie, y_train, y_test = self.prepare_data()
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters())
        
        n_samples = len(X_train_user)
        
        self.train()  # Set model to training mode
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
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_samples:.4f}")
    
    def get_recommendations(self, user_id, n_recommendations=4):
        """Get movie recommendations for a user"""
        self.eval()  # Set model to evaluation mode
        
        if user_id not in self.user_to_idx:
            return []
            
        user_idx = self.user_to_idx[user_id]
        user_tensor = torch.LongTensor([user_idx] * len(self.movie_ids)).to(self.device)
        movie_tensor = torch.LongTensor(range(len(self.movie_ids))).to(self.device)
        
        with torch.no_grad():
            predictions = self(user_tensor, movie_tensor).cpu().numpy()
        
        # Get top N recommendations
        movie_indices = np.argsort(predictions.flatten())[-n_recommendations:][::-1]
        
        # Format recommendations
        user_summary = {
            'UserId': user_id,
            'SimilarityScore': 1.0,
            'Recommendations': []
        }
        
        for idx in movie_indices:
            movie_id = self.movie_ids[idx]
            movie_info = self.data_movies[self.data_movies['MovieID'] == movie_id].iloc[0]
            avg_rating = self.data_ratings[self.data_ratings['MovieID'] == movie_id]['Rating'].mean()
            
            user_summary['Recommendations'].append({
                'Title': movie_info['Title'],
                'Genres': movie_info['Genres'],
                'UserRating': float(predictions[idx] * 5),  # Convert back to 5-star scale
                'AverageRating': float(avg_rating)
            })
            
        return [user_summary]
        
    def save_model(self, path='ncf_model.pth'):
        """Save the trained model"""
        torch.save(self.state_dict(), path)
        
    def load_model(self, path='ncf_model.pth'):
        """Load a previously trained model"""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            return True
        return False


class MovieRecommender:
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
        users = self.ratings_df['UserID'].map(self.user_to_idx).values
        movies = self.ratings_df['MovieID'].map(self.movie_to_idx).values
        ratings = self.ratings_df['Rating'].values / 5.0  
        
        return train_test_split(
            users, movies, ratings,
            test_size=0.2,
            random_state=42
        )
        
    def train(self, epochs=10, batch_size=64):
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