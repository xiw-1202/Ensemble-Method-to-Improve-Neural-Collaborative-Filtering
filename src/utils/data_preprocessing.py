#!/usr/bin/env python
"""
Data preprocessing utilities for the MovieLens dataset
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

# Local paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

def load_movielens_1m():
    """
    Load the MovieLens 1M dataset from raw data directory
    """
    # Specify the encoding to handle special characters
    ratings_path = os.path.join(RAW_DATA_DIR, "ml-1m", "ratings.dat")
    users_path = os.path.join(RAW_DATA_DIR, "ml-1m", "users.dat")
    movies_path = os.path.join(RAW_DATA_DIR, "ml-1m", "movies.dat")
    
    # Load ratings
    ratings = pd.read_csv(
        ratings_path, 
        sep='::', 
        engine='python',
        names=['userId', 'movieId', 'rating', 'timestamp'],
        encoding='ISO-8859-1'
    )
    
    # Load users
    users = pd.read_csv(
        users_path, 
        sep='::', 
        engine='python',
        names=['userId', 'gender', 'age', 'occupation', 'zipcode'],
        encoding='ISO-8859-1'
    )
    
    # Load movies
    movies = pd.read_csv(
        movies_path, 
        sep='::', 
        engine='python',
        names=['movieId', 'title', 'genres'],
        encoding='ISO-8859-1'
    )
    
    return ratings, users, movies

def process_dataset(min_user_ratings=20, min_movie_ratings=10, test_size=0.2, val_size=0.1):
    """
    Process the dataset and split into train, validation, and test sets
    """
    print("Loading dataset...")
    ratings, users, movies = load_movielens_1m()
    
    print(f"Original ratings shape: {ratings.shape}")
    
    # Filter users and movies with too few ratings
    user_counts = ratings.groupby('userId').size()
    movie_counts = ratings.groupby('movieId').size()
    
    valid_users = user_counts[user_counts >= min_user_ratings].index
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
    
    ratings_filtered = ratings[
        ratings['userId'].isin(valid_users) & 
        ratings['movieId'].isin(valid_movies)
    ].copy()
    
    print(f"Filtered ratings shape: {ratings_filtered.shape}")
    
    # Create user and movie indices
    print("Creating user and movie indices...")
    user_to_idx = {user: idx for idx, user in enumerate(ratings_filtered['userId'].unique())}
    movie_to_idx = {movie: idx for idx, movie in enumerate(ratings_filtered['movieId'].unique())}
    
    # Add indices to dataframe
    ratings_filtered['userIdx'] = ratings_filtered['userId'].map(user_to_idx)
    ratings_filtered['movieIdx'] = ratings_filtered['movieId'].map(movie_to_idx)
    
    # Normalize ratings (0 to 1 scale)
    rating_min = ratings_filtered['rating'].min()
    rating_max = ratings_filtered['rating'].max()
    ratings_filtered['rating_norm'] = (ratings_filtered['rating'] - rating_min) / (rating_max - rating_min)
    
    # Split into train, validation, and test sets
    print("Splitting data...")
    train_data, test_data = train_test_split(
        ratings_filtered, 
        test_size=test_size, 
        stratify=ratings_filtered['userId'], 
        random_state=42
    )
    
    train_data, val_data = train_test_split(
        train_data, 
        test_size=val_size/(1-test_size), 
        stratify=train_data['userId'], 
        random_state=42
    )
    
    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")
    
    # Save data to disk
    print("Saving processed data...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Save mappings
    mappings = {
        'user_to_idx': user_to_idx,
        'idx_to_user': {idx: user for user, idx in user_to_idx.items()},
        'movie_to_idx': movie_to_idx,
        'idx_to_movie': {idx: movie for movie, idx in movie_to_idx.items()},
        'rating_min': rating_min,
        'rating_max': rating_max
    }
    
    with open(os.path.join(PROCESSED_DATA_DIR, 'mappings.pkl'), 'wb') as f:
        pickle.dump(mappings, f)
    
    # Save dataframes
    train_data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train.csv'), index=False)
    val_data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'validation.csv'), index=False)
    test_data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test.csv'), index=False)
    movies.to_csv(os.path.join(PROCESSED_DATA_DIR, 'movies.csv'), index=False)
    users.to_csv(os.path.join(PROCESSED_DATA_DIR, 'users.csv'), index=False)
    
    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data,
        'movies': movies,
        'users': users,
        'mappings': mappings
    }

if __name__ == "__main__":
    # Process dataset with default parameters
    process_dataset()
