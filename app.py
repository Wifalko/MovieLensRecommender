from flask import Flask, render_template, request, redirect, url_for, jsonify
from DataReader import file_reader
from datetime import datetime
import pandas as pd
import numpy as np

reader = file_reader()
data_ratings, data_movies, data_users = reader.run()

movies = data_movies
user_ratings = pd.DataFrame(columns=["UserID", "MovieID", "Rating", "Timestamp", "Title"])

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_suggestions')
def search_suggestions():
    query = request.args.get('query', '').lower()
    if len(query) < 2:
        return jsonify([])
    
    # Filtrowanie filmów zawierających zapytanie w tytule
    matching_movies = movies[
        movies['Title'].str.lower().str.contains(query, na=False)
    ]
    
    # Przygotowanie wyników
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
    
    # Oblicz średnią ocenę dla filmu
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
        "UserID": 1,  # Przykładowy ID użytkownika
        "MovieID": movie_id,
        "Rating": float(rating),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Title": movie["Title"],
    }
    
    global user_ratings
    user_ratings = pd.concat([user_ratings, pd.DataFrame([new_rating])], ignore_index=True)
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)