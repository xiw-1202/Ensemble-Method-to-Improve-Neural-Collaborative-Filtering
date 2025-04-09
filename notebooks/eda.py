#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis for MovieLens Dataset
# 
# This notebook performs exploratory data analysis on the MovieLens dataset for the Neural Collaborative Filtering project.

# ## Setup

# In[1]:


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.abspath('..'))

# Set Seaborn style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


# ## Load Raw Data

# In[2]:


# Path to raw data
RAW_DATA_DIR = os.path.join('..', 'data', 'raw', 'ml-1m')

# Load ratings data
ratings = pd.read_csv(
    os.path.join(RAW_DATA_DIR, 'ratings.dat'), 
    sep='::', 
    engine='python',
    names=['userId', 'movieId', 'rating', 'timestamp'],
    encoding='ISO-8859-1'
)

# Load users data
users = pd.read_csv(
    os.path.join(RAW_DATA_DIR, 'users.dat'), 
    sep='::', 
    engine='python',
    names=['userId', 'gender', 'age', 'occupation', 'zipcode'],
    encoding='ISO-8859-1'
)

# Load movies data
movies = pd.read_csv(
    os.path.join(RAW_DATA_DIR, 'movies.dat'), 
    sep='::', 
    engine='python',
    names=['movieId', 'title', 'genres'],
    encoding='ISO-8859-1'
)

# Display data samples
print("Ratings sample:")
print(ratings.head())
print("\nUsers sample:")
print(users.head())
print("\nMovies sample:")
print(movies.head())


# ## Basic Dataset Statistics

# In[3]:


# Basic statistics
print(f"Number of ratings: {len(ratings)}")
print(f"Number of users: {len(ratings['userId'].unique())}")
print(f"Number of movies: {len(ratings['movieId'].unique())}")

# Rating statistics
print("\nRating statistics:")
print(ratings['rating'].describe())

# Distribution of ratings
plt.figure(figsize=(10, 6))
ratings['rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


# ## Analyze User Activity

# In[4]:


# Number of ratings per user
user_ratings_count = ratings.groupby('userId').size()

# Plot distribution of ratings per user
plt.figure(figsize=(12, 6))
sns.histplot(user_ratings_count, bins=50, kde=True)
plt.title('Distribution of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.xlim(0, 1000)  # Focus on the main part of the distribution
plt.show()

# Most active users
print("Top 10 most active users:")
most_active_users = user_ratings_count.sort_values(ascending=False).head(10)
print(most_active_users)

# Merge with user information
most_active_users_info = users[users['userId'].isin(most_active_users.index)]
print("\nDetails of most active users:")
print(most_active_users_info)


# ## Analyze Movie Popularity

# In[5]:


# Number of ratings per movie
movie_ratings_count = ratings.groupby('movieId').size()

# Plot distribution of ratings per movie
plt.figure(figsize=(12, 6))
sns.histplot(movie_ratings_count, bins=50, kde=True)
plt.title('Distribution of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Movies')
plt.xlim(0, 1000)  # Focus on the main part of the distribution
plt.show()

# Most popular movies
print("Top 10 most popular movies:")
most_popular_movies = movie_ratings_count.sort_values(ascending=False).head(10)
most_popular_movies_info = movies[movies['movieId'].isin(most_popular_movies.index)]
most_popular_movies_df = pd.DataFrame({
    'movieId': most_popular_movies.index,
    'title': [most_popular_movies_info[most_popular_movies_info['movieId'] == movie_id]['title'].values[0] 
              for movie_id in most_popular_movies.index],
    'ratings_count': most_popular_movies.values
})
print(most_popular_movies_df)


# ## Analyze Average Ratings

# In[6]:


# Average rating per movie (for movies with at least 50 ratings)
movie_avg_ratings = ratings.groupby('movieId').agg(
    rating_count=('rating', 'count'),
    rating_avg=('rating', 'mean')
)
movie_avg_ratings = movie_avg_ratings[movie_avg_ratings['rating_count'] >= 50]

# Plot average ratings distribution
plt.figure(figsize=(12, 6))
sns.histplot(movie_avg_ratings['rating_avg'], bins=20, kde=True)
plt.title('Distribution of Average Movie Ratings (≥ 50 ratings)')
plt.xlabel('Average Rating')
plt.ylabel('Number of Movies')
plt.xlim(1, 5)
plt.show()

# Top rated movies
print("Top 10 highest rated movies (with at least 50 ratings):")
top_rated_movies = movie_avg_ratings.sort_values('rating_avg', ascending=False).head(10)
top_rated_movies_info = movies[movies['movieId'].isin(top_rated_movies.index)]
top_rated_movies_df = pd.DataFrame({
    'movieId': top_rated_movies.index,
    'title': [top_rated_movies_info[top_rated_movies_info['movieId'] == movie_id]['title'].values[0] 
              for movie_id in top_rated_movies.index],
    'rating_count': top_rated_movies['rating_count'].values,
    'rating_avg': top_rated_movies['rating_avg'].values
})
print(top_rated_movies_df)


# ## Analyze User Demographics

# In[7]:


# Age distribution
plt.figure(figsize=(12, 6))
users['age'].hist(bins=20)
plt.title('Distribution of User Ages')
plt.xlabel('Age')
plt.ylabel('Number of Users')
plt.show()

# Gender distribution
plt.figure(figsize=(8, 6))
users['gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Gender Distribution')
plt.ylabel('')
plt.show()

# Occupation distribution
plt.figure(figsize=(14, 8))
users['occupation'].value_counts().sort_values().plot(kind='barh')
plt.title('Occupation Distribution')
plt.xlabel('Number of Users')
plt.ylabel('Occupation')
plt.show()


# ## Analyze Genre Distribution

# In[8]:


# Extract all genres
all_genres = []
for genres in movies['genres'].str.split('|'):
    all_genres.extend(genres)
genre_counts = pd.Series(all_genres).value_counts()

# Plot genre distribution
plt.figure(figsize=(14, 8))
genre_counts.sort_values().plot(kind='barh')
plt.title('Genre Distribution')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.show()

# Top movies in the most popular genre
top_genre = genre_counts.index[0]
print(f"Top movies in the most popular genre ({top_genre}):")
top_genre_movies = movies[movies['genres'].str.contains(top_genre)]
top_genre_ratings = pd.merge(top_genre_movies, ratings, on='movieId')
top_genre_avg_ratings = top_genre_ratings.groupby(['movieId', 'title']).agg(
    rating_count=('rating', 'count'),
    rating_avg=('rating', 'mean')
).reset_index()
top_genre_avg_ratings = top_genre_avg_ratings[top_genre_avg_ratings['rating_count'] >= 50]
top_genre_avg_ratings = top_genre_avg_ratings.sort_values('rating_avg', ascending=False).head(10)
print(top_genre_avg_ratings)


# ## Analyze User Rating Patterns

# In[9]:


# Average rating per user
user_avg_ratings = ratings.groupby('userId').agg(
    rating_count=('rating', 'count'),
    rating_avg=('rating', 'mean')
)

# Plot average ratings distribution
plt.figure(figsize=(12, 6))
sns.histplot(user_avg_ratings['rating_avg'], bins=20, kde=True)
plt.title('Distribution of Average User Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Number of Users')
plt.xlim(1, 5)
plt.show()

# Plot relationship between number of ratings and average rating
plt.figure(figsize=(12, 6))
sns.scatterplot(x='rating_count', y='rating_avg', data=user_avg_ratings, alpha=0.3)
plt.title('Relationship Between Number of Ratings and Average Rating per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Average Rating')
plt.xscale('log')
plt.show()


# ## Analyze Temporal Patterns

# In[10]:


# Convert timestamp to datetime
ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings['date'] = ratings['datetime'].dt.date
ratings['year'] = ratings['datetime'].dt.year
ratings['month'] = ratings['datetime'].dt.month
ratings['day'] = ratings['datetime'].dt.day
ratings['hour'] = ratings['datetime'].dt.hour

# Plot ratings over time
plt.figure(figsize=(14, 8))
ratings.groupby('date').size().plot()
plt.title('Number of Ratings Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Ratings')
plt.show()

# Plot ratings by hour of day
plt.figure(figsize=(12, 6))
ratings.groupby('hour').size().plot(kind='bar')
plt.title('Number of Ratings by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Number of Ratings')
plt.show()

# Average rating by year
plt.figure(figsize=(12, 6))
ratings.groupby('year')['rating'].mean().plot(kind='bar')
plt.title('Average Rating by Year')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.show()


# ## Analyze Rating Matrix Sparsity

# In[11]:


# Create user-item rating matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Calculate sparsity
num_users, num_items = user_item_matrix.shape
num_ratings = len(ratings)
max_ratings = num_users * num_items
sparsity = 100.0 - (num_ratings / max_ratings * 100)

print(f"User-Item Matrix Shape: {user_item_matrix.shape}")
print(f"Number of Users: {num_users}")
print(f"Number of Movies: {num_items}")
print(f"Number of Ratings: {num_ratings}")
print(f"Maximum Possible Ratings: {max_ratings}")
print(f"Sparsity: {sparsity:.2f}%")


# ## Summary of Key Findings

# In[12]:


# Display summary of key metrics
summary = pd.DataFrame({
    'Metric': [
        'Number of Users',
        'Number of Movies',
        'Number of Ratings',
        'Average Ratings per User',
        'Average Ratings per Movie',
        'Average Rating Value',
        'Matrix Sparsity',
        'Most Common Rating'
    ],
    'Value': [
        len(users),
        len(movies),
        len(ratings),
        num_ratings / len(users),
        num_ratings / len(movies),
        ratings['rating'].mean(),
        f"{sparsity:.2f}%",
        ratings['rating'].mode()[0]
    ]
})

print("Summary of Key Metrics:")
print(summary)

# Plot overall summary
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
ratings['rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')

plt.subplot(2, 2, 2)
sns.histplot(user_ratings_count, bins=30, kde=True)
plt.title('Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.xlim(0, 500)

plt.subplot(2, 2, 3)
sns.histplot(movie_ratings_count, bins=30, kde=True)
plt.title('Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Movies')
plt.xlim(0, 500)

plt.subplot(2, 2, 4)
top_genres = genre_counts.head(10)
top_genres.plot(kind='bar')
plt.title('Top 10 Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
