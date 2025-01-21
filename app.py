from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from DataReader import file_reader
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from models import NCF

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

USERS_FILE = 'users.json'
USER_RATINGS_DIR = 'user_ratings'

reader = file_reader()
data_ratings, data_movies, data_users = reader.run()
movies = data_movies

def load_users():
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def get_user_ratings_file(username):
    return f'{USER_RATINGS_DIR}/{username}_ratings.json'

def load_user_ratings(username):
    try:
        with open(get_user_ratings_file(username), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_user_ratings(username, ratings):
    os.makedirs(USER_RATINGS_DIR, exist_ok=True)
    with open(get_user_ratings_file(username), 'w') as f:
        json.dump(ratings, f, indent=2)

def get_user_movie_matrix(user_ratings_df):
    all_ratings = pd.concat([
        data_ratings[['UserID', 'MovieID', 'Rating']],
        user_ratings_df[['UserID', 'MovieID', 'Rating']]
    ])
    
    user_movie_matrix = all_ratings.pivot(
        index='UserID', 
        columns='MovieID', 
        values='Rating'
    ).fillna(0)
    
    return user_movie_matrix

def get_similar_users(user_ratings_df, n_similar=5):
    user_movie_matrix = get_user_movie_matrix(user_ratings_df)
    
    user_similarity = cosine_similarity(user_movie_matrix)
    
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index
    )
    
    current_user_id = user_ratings_df['UserID'].iloc[0]
    
    similar_users = user_similarity_df[current_user_id].sort_values(ascending=False)[1:n_similar+1]
    
    return similar_users

def get_recommendations(model_type, user_ratings_df, username, n_recommendations=5):
    if len(user_ratings_df) == 0:
        return []
        
    similar_users = get_similar_users(user_ratings_df)
    
    recommendations = []
    
    for user_id, similarity_score in similar_users.items():
        user_movies = data_ratings[
            (data_ratings['UserID'] == user_id) & 
            (data_ratings['Rating'] >= 4.0)
        ]
        
        rated_movies = set(user_ratings_df['MovieID'].values)
        user_movies = user_movies[~user_movies['MovieID'].isin(rated_movies)]
        
        top_movies = user_movies.nlargest(n_recommendations, 'Rating')
        
        user_recs = {
            'UserId': int(user_id),
            'SimilarityScore': float(similarity_score),
            'Recommendations': []
        }
        
        for _, movie in top_movies.iterrows():
            movie_info = movies[movies['MovieID'] == movie['MovieID']].iloc[0]
            avg_rating = data_ratings[data_ratings['MovieID'] == movie['MovieID']]['Rating'].mean()
            
            user_recs['Recommendations'].append({
                'MovieID': int(movie['MovieID']),
                'Title': movie_info['Title'],
                'Genres': movie_info['Genres'],
                'UserRating': float(movie['Rating']),
                'AverageRating': float(avg_rating)
            })
            
        recommendations.append(user_recs)
    
    return recommendations

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('select_user'))
    return render_template('index.html')

@app.route('/select_user')
def select_user():
    users = load_users()
    return render_template('select_user.html', users=users)

@app.route('/create_user', methods=['POST'])
def create_user():
    data = request.json
    username = data.get('username')
    
    if not username:
        return jsonify({'error': 'Username is required'}), 400
        
    users = load_users()
    
    if username in users:
        return jsonify({'error': 'Username already exists'}), 400
    
    users[username] = {
        'name': username,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    save_users(users)
    session['username'] = username
    
    return jsonify({
        'success': True,
        'username': username
    })

@app.route('/switch_user', methods=['POST'])
def switch_user():
    data = request.json
    username = data.get('username')
    
    if not username:
        return jsonify({'error': 'Username is required'}), 400
        
    users = load_users()
    
    if username not in users:
        return jsonify({'error': 'User does not exist'}), 404
    
    session['username'] = username
    return jsonify({
        'success': True,
        'username': username
    })

@app.route('/get_current_user')
def get_current_user():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'No user selected'}), 404
    
    users = load_users()
    if username not in users:
        return jsonify({'error': 'Invalid user'}), 404
        
    return jsonify({
        'username': username,
        'created_at': users[username]['created_at']
    })

@app.route('/get_users')
def get_users():
    users = load_users()
    return jsonify(users)

@app.route('/rate', methods=['POST'])
def rate_movie():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'No user selected'}), 401
    
    data = request.json
    movie_id = data.get('movieId')
    rating = data.get('rating')
    
    if not movie_id or not rating:
        return jsonify({'error': 'Missing required fields'}), 400
    
    movie = movies[movies['MovieID'] == movie_id].iloc[0]
    
    ratings = load_user_ratings(username)
    
    rating_exists = False
    for r in ratings:
        if r['MovieID'] == movie_id:
            r['Rating'] = float(rating)
            rating_exists = True
            break
    
    if not rating_exists:
        ratings.append({
            "UserID": 9999,  # Default UserID for new ratings
            "MovieID": movie_id,
            "Rating": float(rating),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Title": movie["Title"],
            "Genres": movie["Genres"]
        })
    
    save_user_ratings(username, ratings)
    return jsonify({'success': True})

@app.route('/get_saved_ratings')
def get_saved_ratings():
    username = session.get('username')
    if not username:
        return jsonify([])
    return jsonify(load_user_ratings(username))

@app.route('/delete_rating/<int:movie_id>', methods=['DELETE'])
def delete_rating(movie_id):
    username = session.get('username')
    if not username:
        return jsonify({'error': 'No user selected'}), 401

    try:
        ratings = load_user_ratings(username)
        ratings = [rating for rating in ratings if rating['MovieID'] != movie_id]
        save_user_ratings(username, ratings)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/recommendations')
def recommendations():
    if 'username' not in session:
        return redirect(url_for('select_user'))

    username = session.get('username')
    model_type = request.args.get('model', 'knn')

    user_ratings = load_user_ratings(username)
    if not user_ratings:
        return render_template(
            'recommendations.html',
            recommended_movies=[],
            no_ratings=True,
            model_type=model_type
        )

    user_ratings_df = pd.DataFrame(user_ratings)

    if model_type.lower() == 'ncf':
        recommender = NCF(data_ratings, data_movies)
        recommender.train_model(user_ratings_df, epochs=10)
        recommended_movies = recommender.get_recommendations(9999) or []
    else:
        recommended_movies = get_recommendations(model_type, user_ratings_df, username) or []

    return render_template(
        'recommendations.html',
        recommended_movies=recommended_movies,
        no_ratings=False,
        model_type=model_type,
    )

@app.route('/recommendations/loading')
def recommendations_loading():
    if 'username' not in session:
        return redirect(url_for('select_user'))
    
    model_type = request.args.get('model', 'knn')
    if model_type != 'ncf':
        return redirect(url_for('recommendations', model=model_type))
    
    return render_template('loading.html', model_type=model_type)

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
        return jsonify({'error': 'Movie not found'}), 404

    avg_rating = data_ratings[data_ratings['MovieID'] == movie_id]['Rating'].mean()
    
    movie_data = {
        'MovieID': int(movie.iloc[0]['MovieID']),
        'Title': movie.iloc[0]['Title'],
        'Genres': movie.iloc[0]['Genres'],
        'AverageRating': float(avg_rating) if not np.isnan(avg_rating) else None
    }
    
    return jsonify(movie_data)

@app.route('/similar_users')
def similar_users():
    if 'username' not in session:
        return redirect(url_for('select_user'))

    username = session.get('username')
    user_ratings = load_user_ratings(username)
    
    if not user_ratings:
        return render_template(
            'similar_users.html',
            recommended_movies=[],
            no_ratings=True
        )

    user_ratings_df = pd.DataFrame(user_ratings)
    recommended_movies = get_recommendations('knn', user_ratings_df, username) or []

    return render_template(
        'similar_users.html',
        recommended_movies=recommended_movies,
        no_ratings=False
    )

@app.route('/recommended_movies')
def recommended_movies():
    if 'username' not in session:
        return redirect(url_for('select_user'))

    username = session.get('username')
    user_ratings = load_user_ratings(username)
    
    if not user_ratings:
        return render_template(
            'recommended_movies.html',
            recommended_movies=[],
            no_ratings=True,
            loading=False
        )

    user_ratings_df = pd.DataFrame(user_ratings)
    recommender = NCF(data_ratings, data_movies)
    recommender.train_model(user_ratings_df, epochs=10)
    recommended_movies = recommender.get_recommendations(9999) or []

    return render_template(
        'recommended_movies.html',
        recommended_movies=recommended_movies,
        no_ratings=False,
        loading=True
    )

@app.route('/recommended_movies/loading')
def recommended_movies_loading():
    if 'username' not in session:
        return redirect(url_for('select_user'))
    return render_template('loading.html')

if __name__ == '__main__':
    app.run(debug=True)