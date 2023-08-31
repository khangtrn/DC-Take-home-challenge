from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
model = load_model('RecommendSystem')
# Read movies data
movies_df = pd.read_csv('movies.csv')
# Read ratings data
ratings_df = pd.read_csv('ratings.csv')
movie = ratings_df['movieId'].unique()
user_id = np.array(int(input('User ID from 1 to 610: ')))
moive_seen_idx = ratings_df[(ratings_df['userId']==user_id)].index
moive_seen = []
movies_to_predict = []
for idx in moive_seen_idx:
  moive_seen.append(ratings_df['movieId'][idx])
for movie_id in movie:
  if movie_id not in np.array(moive_seen):
    movies_to_predict.append(movie_id)

user_id_to_predict = np.array([user_id] * len(movies_to_predict))

# Prepare the movie IDs for prediction
movie_ids_to_predict = np.array(movies_to_predict)

# Predict using the model
predicted_ratings = model.predict([user_id_to_predict, movie_ids_to_predict])

# Combine movie IDs with their predicted ratings
movie_predictions = list(zip(movies_to_predict, predicted_ratings))

# Sort predictions based on predicted ratings
sorted_predictions = sorted(movie_predictions, key=lambda x: x[1], reverse=True)

print(f"Top recommendations for user #{user_id}:")
for movie_id, predicted_rating in sorted_predictions[:10]:  # Print top 10 recommendations
    print(f"Movie: {movies_df.loc[movie_id-1].title}")
