from flask import Flask, render_template, request, redirect, url_for, jsonify
from DataReader import file_reader
from datetime import datetime
import pandas as pd
import numpy as np
import os
import json
from data_processing import get_recommendations


reader = file_reader()
data_ratings, data_movies, data_users = reader.run()

movies = data_movies
user_ratings = pd.DataFrame(columns=["UserID", "MovieID", "Rating", "Timestamp", "Title", "Genres"])

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_suggestions')
def search_suggestions():
    query = request.args.get('query', '').lower()
    if len(query) < 2:
        return jsonify([])

    matching_movies = movies[
        movies['Title'].str.lower().str.contains(query, na=False)
    ]

    suggestions = []
    for _, movie in matching_movies.head(10).iterrows():
        suggestions.append({
            'MovieID': int(movie['MovieID']),
            'Title': movie['Title'],
            'Genres': movie['Genres']
        })
    
    return jsonify(suggestions)

@app.route('/get_movie/<int:movie_id>')
def get_movie(movie_id):
    movie = movies[movies['MovieID'] == movie_id]
    if len(movie) == 0:
        return jsonify({'error': 'Film nie został znaleziony'}), 404

    avg_rating = data_ratings[data_ratings['MovieID'] == movie_id]['Rating'].mean()
    
    movie_data = {
        'MovieID': int(movie.iloc[0]['MovieID']),
        'Title': movie.iloc[0]['Title'],
        'Genres': movie.iloc[0]['Genres'],
        'AverageRating': float(avg_rating) if not np.isnan(avg_rating) else None
    }
    
    return jsonify(movie_data)

@app.route('/rate', methods=['POST'])
def rate_movie():
    data = request.json
    movie_id = data.get('movieId')
    rating = data.get('rating')
    
    if not movie_id or not rating:
        return jsonify({'error': 'Brak wymaganych pól'}), 400
    
    movie = movies[movies['MovieID'] == movie_id].iloc[0]
    
    new_rating = {
        "UserID": 9999,  
        "MovieID": movie_id,
        "Rating": float(rating),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Title": movie["Title"],
        "Genres": movie["Genres"]
    }
    
    global user_ratings
    user_ratings = pd.concat([user_ratings, pd.DataFrame([new_rating])], ignore_index=True)
    save_ratings_to_file()  
    
    return jsonify({'success': True})



@app.route('/get_saved_ratings')
def get_saved_ratings():
    try:
        if os.path.exists('user_ratings.json'):
            with open('user_ratings.json', 'r', encoding='utf-8') as file:
                ratings = json.load(file)
            return jsonify(ratings)
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/delete_rating/<int:movie_id>', methods=['DELETE'])
def delete_rating(movie_id):
    try:
        if os.path.exists('user_ratings.json'):
            with open('user_ratings.json', 'r', encoding='utf-8') as file:
                ratings = json.load(file)
            
            # Filtruj oceny, usuwając ocenę dla danego filmu
            ratings = [rating for rating in ratings if rating['MovieID'] != movie_id]
            
            # Zapisz zaktualizowaną listę ocen
            with open('user_ratings.json', 'w', encoding='utf-8') as file:
                json.dump(ratings, file, indent=2)
                
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Plik z ocenami nie istnieje'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/recommendations')
def recommendations():
    # Możesz dodać parametr w URL, który model chcesz użyć
    model_type = request.args.get('model', 'knn')  # domyślnie użyj KNN
    recommended_movies = get_recommendations(model_type)
    return render_template('recommendations.html', recommended_movies=recommended_movies)

# Albo możesz stworzyć osobne endpointy dla każdego modelu
@app.route('/recommendations/knn')
def knn_recommendations():
    recommended_movies = get_recommendations('knn')
    return render_template('recommendations.html', recommended_movies=recommended_movies)

@app.route('/recommendations/ncf')
def ncf_recommendations():
    recommended_movies = get_recommendations('ncf')
    return render_template('recommendations.html', recommended_movies=recommended_movies)

RATINGS_FILE = 'user_ratings.json'

def save_ratings_to_file():
    if not user_ratings.empty:
        user_ratings.to_json(RATINGS_FILE, orient='records', date_format='iso')

def load_ratings_from_file():
    global user_ratings
    if os.path.exists(RATINGS_FILE):
        user_ratings = pd.read_json(RATINGS_FILE)


if __name__ == '__main__':
    load_ratings_from_file()
    app.run(debug=True)