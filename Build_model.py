import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input, dot, concatenate, Flatten
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Read movies data
movies_df = pd.read_csv('movies.csv')

# Read ratings data
ratings_df = pd.read_csv('ratings.csv')

# Train and test
X = ratings_df.iloc[:,:2]
Y = ratings_df.iloc[:,2]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 66)
# the number of embedding dimensions: embedding size = min(50, number of categories/2)
embedding_dimension = 50
# no of users and movies
n_users = ratings_df['userId'].nunique()
n_movies = ratings_df['movieId'].nunique()
# User Embeddings
user_input = Input(shape=(1,))
user_embeddings = Embedding(input_dim = n_users + 1, output_dim=embedding_dimension, input_length=1,
                              name='User_Embedding')(user_input)
user_vector = Flatten(name='User_Vector')(user_embeddings)
# Movie Embeddings
movie_input = Input(shape=(1,))
movie_embeddings = Embedding(input_dim = n_movies + 1, output_dim=embedding_dimension, input_length=1,
                               name='Movie_Embedding') (movie_input)
movie_vector = Flatten(name='Movie_Vector') (movie_embeddings)
# Dot Product
merged_vectors = dot([user_vector, movie_vector], axes=1)
model = Model([user_input, movie_input], merged_vectors)
# Compile
model.compile(loss='mean_squared_error', optimizer = 'Adam')
# Train
batch_size = 128
epochs = 10
# x=[x_train['userId'], x_train['movieId']]
# print(x_train)
history = model.fit(x=[x_train['userId'], x_train['movieId']], y=y_train, batch_size= batch_size, epochs=epochs,
                    verbose= 2, validation_data=([x_test['userId'], x_test['movieId']], y_test))
# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)
model.save('RecommendSystem')
# Visualize loss history
plt.figure(figsize = (8,4))
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
