from DataReader import file_reader

folder_path = 'Data/ml-1m/'
movies = 'movies.dat'
users = 'users.dat'
ratings = 'ratings.dat'
name_movies = folder_path + movies
name_users = folder_path + users
name_ratings = folder_path + ratings

reader = file_reader(name_ratings, name_movies, name_users)
data_ratings, data_movies, data_users = reader.run()

print('elo')
