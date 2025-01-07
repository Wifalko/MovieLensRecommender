import pandas as pd

folder_path = 'Data/ml-1m/'
movies = 'movies.dat'
users = 'users.dat'
ratings = 'ratings.dat'
name_movie = folder_path + movies
name_user = folder_path + users
name_rating = folder_path + ratings

class file_reader:
    def __init__(self, name_ratings=name_rating, name_movies=name_movie, name_users=name_user):
        self.name_ratings = name_ratings
        self.name_movies = name_movies
        self.name_users = name_users

    def get_ratings(self):
        with open(self.name_ratings, 'r') as file:
            data = file.read()

        lines = data.splitlines()
        split_data = [line.split('::') for line in lines]

        df = pd.DataFrame(split_data, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])

        df['UserID'] = df['UserID'].astype(int)
        df['MovieID'] = df['MovieID'].astype(int)
        df['Rating'] = df['Rating'].astype(float)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        return df

    def get_movies(self):
        with open(self.name_movies, 'r') as file:
            data = file.read()
        lines = data.splitlines()
        split_data = [line.split('::') for line in lines]

        df = pd.DataFrame(split_data, columns=['MovieID', 'Title', 'Genres'])

        df['MovieID'] = df['MovieID'].astype(int)
        df['Title'] = df['Title'].astype(str)
        df['Genres'] = df['Genres'].astype(str)
        return df

    def get_users(self):
        with open(self.name_users, 'r') as file:
            data = file.read()
        lines = data.splitlines()
        split_data = [line.split('::') for line in lines]

        df = pd.DataFrame(split_data, columns=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])

        df['UserID'] = df['UserID'].astype(int)
        df['Gender'] = df['Gender'].astype(str)
        df['Age'] = df['Age'].astype(int)
        df['Occupation'] = df['Occupation'].astype(int)
        return df

    def run(self):
        return self.get_ratings(), self.get_movies(), self.get_users()
    

