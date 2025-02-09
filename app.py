from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
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
    
USER_RATINGS_FOLDER = "user_ratings"

def get_user_ratings_file(username):
    """Zwraca ścieżkę do pliku ocen użytkownika."""
    return os.path.join(USER_RATINGS_FOLDER, f"{username}_ratings.json")

def load_user_ratings(username):
    """Wczytuje oceny użytkownika z jego pliku JSON."""
    file_path = get_user_ratings_file(username)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []
    return []

def save_user_ratings(username, ratings):
    """Zapisuje oceny użytkownika do jego pliku JSON."""
    file_path = get_user_ratings_file(username)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(ratings, file, indent=4)

def get_user_id(username, start_id=9999):
    """Zwraca przypisane UserID dla użytkownika lub nadaje nowe."""
    ratings = load_user_ratings(username)
    
    if ratings:
        return ratings[0]["UserID"]  

    existing_user_ids = set()
    
    for filename in os.listdir(USER_RATINGS_FOLDER):
        if filename.endswith("_ratings.json"):
            file_path = os.path.join(USER_RATINGS_FOLDER, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    if data:
                        existing_user_ids.add(data[0]["UserID"])  
                except json.JSONDecodeError:
                    print(f"Błąd odczytu pliku: {file_path}")

    user_id = start_id
    while user_id in existing_user_ids:
        user_id += 1

    return user_id

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

def load_users():
    try:
        with open('users.json', 'r') as f:
            return json.load(f)  
    except FileNotFoundError:
        return {}

def get_all_users():
    users = load_users()
    return sorted([user_data['name'] for user_data in users.values()])

def generate_unique_user_id():
    users = load_users()
    if not users:
        return 1
    existing_ids = [int(users[user].get('userId', 0)) for user in users]
    return max(existing_ids) + 1


@app.route('/')
def home():
    return render_template('home.html')  

@app.route('/index')
def index():
    if 'username' not in session:
        return redirect(url_for('select_user'))
    username = session['username']
    return render_template('index.html', username=username)


@app.route('/create_user', methods=['POST'])
def create_user():
    try:
        data = request.json
        username = data.get('username')
        
        if not username:
            return jsonify({'error': 'Username is required'}), 400
            
        users = load_users()
        
        if username in users:
            return jsonify({'error': 'Username already exists'}), 400
        
        new_user_id = generate_unique_user_id()
        users[username] = {
            'name': username,
            'userId': new_user_id,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        save_users(users)
        session['username'] = username
        session['user_id'] = new_user_id
        
        return jsonify({
            'success': True,
            'userId': new_user_id,
            'username': username
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/select_user', methods=['GET', 'POST'])
def select_user():
    if request.method == 'GET':
        users = load_users()
        return render_template('select_user.html', users=users)
    
    username = request.form.get('username')
    if not username:
        flash('Username is required')
        return redirect(url_for('select_user'))
    
    users = load_users()
    if username not in users:
        flash('User does not exist')
        return redirect(url_for('select_user'))
    
    session['username'] = username
    session['user_id'] = users[username]['userId']
    return redirect(url_for('index'))

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
    
    user_id = get_user_id(username)  

    rating_exists = False
    for r in ratings:
        if r['MovieID'] == movie_id:
            r['Rating'] = float(rating)
            rating_exists = True
            break
    
    if not rating_exists:
        ratings.append({
            "UserID": user_id,  
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
    user_id = get_user_id(username)  
    user_ratings = load_user_ratings(username)
    if not user_ratings:
        return render_template(
            'recommendations.html',
            recommended_movies=[],
            no_ratings=True,
            model_type=model_type,
            all_users=get_all_users(),
            current_user=username
        )

    user_ratings_df = pd.DataFrame(user_ratings)

    if model_type.lower() == 'ncf':
        recommender = NCF(data_ratings, data_movies)
        recommender.train_model(user_ratings_df, epochs=4)
        recommended_movies = recommender.get_recommendations(user_id) or []
    else:
        recommended_movies = get_recommendations(model_type, user_ratings_df, username) or []

    return render_template(
        'recommendations.html',
        recommended_movies=recommended_movies,
        no_ratings=False,
        model_type=model_type,
        all_users=get_all_users(),
        current_user=username,
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
    user_id = get_user_id(username)
    if not user_ratings:
        return render_template(
            'recommended_movies.html',
            recommended_movies=[],
            no_ratings=True,
            loading=False
        )

    user_ratings_df = pd.DataFrame(user_ratings)
    recommender = NCF(data_ratings, data_movies)
    recommender.train_model(user_ratings_df, epochs=3)
    recommended_movies = recommender.get_recommendations(user_id) or []

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


@app.route('/dual_recommendations')
def dual_recommendations():
    if 'username' not in session:
        return redirect(url_for('select_user'))
    
    current_user = session['username']
    return render_template(
        'dual_recommendations.html',
        current_user=current_user,
        all_users=get_all_users(),
        recommended_movies=None,
        second_user=None
    )

@app.route('/get_dual_recommendations')
def get_dual_recommendations():
    if 'username' not in session:
        return redirect(url_for('select_user'))
    
    current_user = session['username']
    second_user = request.args.get('second_user')
    current_user_id = get_user_id(current_user)
    second_user_id = get_user_id(second_user)

    if not second_user:
        return redirect(url_for('dual_recommendations'))
    
    try:
        user1_ratings = load_user_ratings(current_user)
        user2_ratings = load_user_ratings(second_user)
        
        if not user1_ratings or not user2_ratings:
            flash('Nie znaleziono ocen dla jednego lub obu użytkowników', 'error')
            return redirect(url_for('dual_recommendations'))
        
        user1_df = pd.DataFrame(user1_ratings)
        user2_df = pd.DataFrame(user2_ratings)

        merged_df = pd.merge(user1_df, user2_df, on='MovieID', how='outer', suffixes=('_user1', '_user2'))
        merged_df['Rating'] = merged_df[['Rating_user1', 'Rating_user2']].mean(axis=1, skipna=True)
        merged_df['Title'] = merged_df['Title_user1'].combine_first(merged_df['Title_user2'])
        merged_df['Genres'] = merged_df['Genres_user1'].combine_first(merged_df['Genres_user2'])
                                                                      
        merged_df['UserID'] = 0
        merged_df['Timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

        result_df = merged_df[['UserID', 'MovieID', 'Rating', 'Timestamp', 'Title', 'Genres']]
        
        recommender = NCF(data_ratings, data_movies)
        recommender.train_model(result_df, epochs=4)
        recommended_movies = recommender.get_recommendations(0)
        
        if recommended_movies is None or len(recommended_movies) == 0:
            flash('Nie udało się wygenerować rekomendacji', 'error')
            return redirect(url_for('dual_recommendations'))
            
        return render_template(
            'dual_recommendations.html',
            current_user=current_user,
            all_users=get_all_users(),
            recommended_movies=recommended_movies,
            second_user=second_user
        )
    except Exception as e:
        flash(f'Wystąpił błąd podczas generowania rekomendacji: {str(e)}', 'error')
        return redirect(url_for('dual_recommendations'))
    
    
if __name__ == '__main__':
    app.run(debug=True)