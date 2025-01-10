from DataReader import file_reader
import pandas as pd
import json
from models import KNNRecommender, NCF  # zakładając że mamy oba modele w models.py

def load_data():
    """Ładuje wszystkie potrzebne dane"""
    with open('user_ratings.json', 'r', encoding='utf-8') as file:
        user_input_data = json.load(file)

    user_data = pd.DataFrame(user_input_data)
    user_data['Timestamp'] = pd.to_datetime(user_data['Timestamp'])
 
    reader = file_reader()
    data_ratings, data_movies, data_users = reader.run()
    
    return user_data, data_ratings, data_movies, data_users

def get_recommendations_knn(user_id=9999, neighbours_amount=4):
    user_data, data_ratings, data_movies, _ = load_data()

    recommender = KNNRecommender(data_ratings, data_movies)
    recommender.prepare_data(user_data)
    recommender.train()
    
    return recommender.get_recommendations(user_id, neighbours_amount)

def get_recommendations_ncf(user_id=9999, n_recommendations=4):
    user_data, data_ratings, data_movies, _ = load_data()

    recommender = NCF(data_ratings, data_movies)
    recommender.prepare_data(user_data)
    
    try:
        recommender.load_model()
    except:
        recommender.train(epochs=10)
        recommender.save_model()
    
    return recommender.get_recommendations(user_id, n_recommendations)

def get_recommendations(model_type='knn', user_id=9999, n_recommendations=4):
    if model_type.lower() == 'knn':
        return get_recommendations_knn(user_id, n_recommendations)
    elif model_type.lower() == 'ncf':
        return get_recommendations_ncf(user_id, n_recommendations)
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}. Dostępne opcje: 'knn', 'ncf'")

if __name__ == "__main__":
    knn_recommendations = get_recommendations('knn')
    ncf_recommendations = get_recommendations('ncf')