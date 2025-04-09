#!/usr/bin/env python
"""
Demonstrate movie recommendations with trained models
"""
import os
import argparse
import pickle
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from models.ncf import NCF
from models.gat import GATModel
from models.ensemble import EnsembleModel
from utils.data_utils import load_data, create_edge_index, get_recommendations, denormalize_rating

def load_model(model_name, model_path, data):
    """
    Load a trained model
    
    Parameters:
    -----------
    model_name : str
        Name of the model ('ncf', 'gat', or 'ensemble')
    model_path : str
        Path to the model file
    data : dict
        Data dictionary containing mappings, etc.
        
    Returns:
    --------
    model : torch.nn.Module
        Loaded model
    """
    if model_name == 'ncf':
        model = NCF(
            num_users=data['num_users'],
            num_items=data['num_items']
        )
    elif model_name == 'gat':
        model = GATModel(
            num_users=data['num_users'],
            num_items=data['num_items']
        )
    elif model_name == 'ensemble':
        model = EnsembleModel(
            num_users=data['num_users'],
            num_items=data['num_items']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model

def get_user_info(user_id, data):
    """
    Get information about a user
    
    Parameters:
    -----------
    user_id : int
        User ID
    data : dict
        Data dictionary containing user data
        
    Returns:
    --------
    dict
        User information
    """
    # Get user index
    user_idx = data['mappings']['user_to_idx'].get(user_id)
    
    if user_idx is None:
        return None
    
    # Get user's ratings
    user_ratings = data['train_df'][data['train_df']['userIdx'] == user_idx]
    
    # Get user's favorite movies
    user_fav_movies = user_ratings.sort_values('rating', ascending=False).head(5)
    user_fav_movies = user_fav_movies.merge(data['movies_df'], on='movieId')
    
    # Get user's rated genres
    movie_ids = user_ratings['movieId'].values
    rated_movies = data['movies_df'][data['movies_df']['movieId'].isin(movie_ids)]
    
    # Extract genres
    all_genres = []
    for genres in rated_movies['genres'].str.split('|'):
        all_genres.extend(genres)
    genre_counts = pd.Series(all_genres).value_counts()
    
    return {
        'user_id': user_id,
        'user_idx': user_idx,
        'num_ratings': len(user_ratings),
        'avg_rating': user_ratings['rating'].mean(),
        'favorite_movies': user_fav_movies,
        'top_genres': genre_counts.head(5)
    }

def plot_user_profile(user_info):
    """
    Plot user profile
    
    Parameters:
    -----------
    user_info : dict
        User information
    """
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot favorite movies
    plt.subplot(2, 1, 1)
    plt.barh(user_info['favorite_movies']['title'], user_info['favorite_movies']['rating'])
    plt.title(f"User {user_info['user_id']} - Favorite Movies")
    plt.xlabel('Rating')
    plt.ylabel('Movie')
    plt.xlim(0, 5.5)
    
    # Plot top genres
    plt.subplot(2, 1, 2)
    plt.barh(user_info['top_genres'].index, user_info['top_genres'].values)
    plt.title(f"User {user_info['user_id']} - Top Genres")
    plt.xlabel('Count')
    plt.ylabel('Genre')
    
    plt.tight_layout()
    plt.show()

def compare_recommendations(user_id, models, edge_index, data, num_recommendations=10):
    """
    Compare recommendations from different models
    
    Parameters:
    -----------
    user_id : int
        User ID
    models : dict
        Dictionary of model name to model
    edge_index : torch.Tensor
        Graph edge index
    data : dict
        Data dictionary
    num_recommendations : int
        Number of recommendations to show
    """
    # Get recommendations from each model
    all_recommendations = {}
    
    for model_name, model in models.items():
        recs = get_recommendations(
            model=model,
            user_id=user_id,
            mappings=data['mappings'],
            edge_index=edge_index,
            movies_df=data['movies_df'],
            n=num_recommendations,
            exclude_seen=True,
            train_df=data['train_df']
        )
        
        # Denormalize ratings if needed
        if 'rating_min' in data['mappings'] and 'rating_max' in data['mappings']:
            recs['prediction_original'] = recs['prediction'].apply(
                lambda x: denormalize_rating(x, data['mappings']['rating_min'], data['mappings']['rating_max'])
            )
        else:
            recs['prediction_original'] = recs['prediction'] * 5  # Assume 0-1 to 0-5 scale
        
        all_recommendations[model_name] = recs
    
    # Create a merged dataframe with recommendations from all models
    merged_recs = pd.DataFrame()
    
    for model_name, recs in all_recommendations.items():
        # Keep only essential columns
        model_recs = recs[['movieId', 'title', 'prediction_original']]
        model_recs.columns = ['movieId', 'title', f'{model_name}_score']
        
        if merged_recs.empty:
            merged_recs = model_recs
        else:
            merged_recs = pd.merge(merged_recs, model_recs, on=['movieId', 'title'], how='outer')
    
    # Fill NaN with 0
    merged_recs = merged_recs.fillna(0)
    
    # Sort by ensemble score if available
    if 'ensemble_score' in merged_recs.columns:
        merged_recs = merged_recs.sort_values('ensemble_score', ascending=False)
    elif 'ncf_score' in merged_recs.columns:
        merged_recs = merged_recs.sort_values('ncf_score', ascending=False)
    
    # Keep top recommendations
    merged_recs = merged_recs.head(num_recommendations)
    
    # Plot recommendations
    plt.figure(figsize=(15, 10))
    
    # Plot recommendation scores
    score_cols = [col for col in merged_recs.columns if col.endswith('_score')]
    
    for i, col in enumerate(score_cols):
        plt.subplot(len(score_cols), 1, i+1)
        
        # Sort by this model's score
        model_recs = merged_recs.sort_values(col, ascending=False)
        
        plt.barh(model_recs['title'], model_recs[col])
        plt.title(f"Top Recommendations - {col.replace('_score', '')}")
        plt.xlabel('Predicted Rating')
        plt.xlim(0, 5.5)
    
    plt.tight_layout()
    plt.show()
    
    # Return merged recommendations
    return merged_recs

def predict_specific_movies(user_id, movie_ids, models, edge_index, data):
    """
    Predict ratings for specific movies
    
    Parameters:
    -----------
    user_id : int
        User ID
    movie_ids : list
        List of movie IDs
    models : dict
        Dictionary of model name to model
    edge_index : torch.Tensor
        Graph edge index
    data : dict
        Data dictionary
        
    Returns:
    --------
    pandas.DataFrame
        Predictions for specific movies
    """
    # Get user index
    user_idx = data['mappings']['user_to_idx'].get(user_id)
    
    if user_idx is None:
        return None
    
    # Filter movies that exist in the dataset
    valid_movie_ids = []
    for movie_id in movie_ids:
        if movie_id in data['mappings']['movie_to_idx']:
            valid_movie_ids.append(movie_id)
    
    if not valid_movie_ids:
        return None
    
    # Get movie information
    movies_info = data['movies_df'][data['movies_df']['movieId'].isin(valid_movie_ids)]
    
    # Create predictions dataframe
    predictions = pd.DataFrame({
        'movieId': valid_movie_ids,
        'title': [movies_info[movies_info['movieId'] == movie_id]['title'].values[0] 
                  for movie_id in valid_movie_ids]
    })
    
    # Get predictions from each model
    for model_name, model in models.items():
        model_preds = []
        
        for movie_id in valid_movie_ids:
            movie_idx = data['mappings']['movie_to_idx'][movie_id]
            
            # Create tensors
            user_tensor = torch.tensor([user_idx], dtype=torch.long)
            movie_tensor = torch.tensor([movie_idx], dtype=torch.long)
            
            # Get prediction
            with torch.no_grad():
                try:
                    pred = model(user_tensor, movie_tensor, edge_index).item()
                except TypeError:
                    pred = model(user_tensor, movie_tensor).item()
            
            # Denormalize rating if needed
            if 'rating_min' in data['mappings'] and 'rating_max' in data['mappings']:
                pred = denormalize_rating(pred, data['mappings']['rating_min'], data['mappings']['rating_max'])
            else:
                pred = pred * 5  # Assume 0-1 to 0-5 scale
            
            model_preds.append(pred)
        
        predictions[f'{model_name}_score'] = model_preds
    
    return predictions

def main(args):
    """
    Main function for demonstrating recommendations
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Load data
    print("Loading data...")
    data = load_data(args.data_dir)
    
    # Create edge index for graph models
    edge_index = create_edge_index(data['train_df'], data['num_users'])
    
    # Load models
    print("Loading models...")
    models = {}
    
    for model_name in args.models:
        # Try different model files (regular or tuned)
        model_path = os.path.join(args.models_dir, f"{model_name}_model.pth")
        tuned_model_path = os.path.join(args.models_dir, f"{model_name}_tuned_model.pth")
        
        if os.path.exists(tuned_model_path):
            print(f"Loading tuned {model_name} model from {tuned_model_path}")
            models[model_name] = load_model(model_name, tuned_model_path, data)
        elif os.path.exists(model_path):
            print(f"Loading {model_name} model from {model_path}")
            models[model_name] = load_model(model_name, model_path, data)
        else:
            print(f"Warning: Could not find model file for {model_name}")
    
    if not models:
        print("Error: No models loaded. Please check model paths.")
        return
    
    # User ID selection
    user_id = args.user_id
    
    if user_id is None:
        # Randomly select a user
        user_id = random.choice(list(data['mappings']['user_to_idx'].keys()))
        print(f"Randomly selected user ID: {user_id}")
    
    # Get user information
    print(f"Getting information for user {user_id}...")
    user_info = get_user_info(user_id, data)
    
    if user_info is None:
        print(f"Error: User {user_id} not found in the dataset.")
        return
    
    # Print user information
    print("\nUser Information:")
    print(f"User ID: {user_info['user_id']}")
    print(f"Number of ratings: {user_info['num_ratings']}")
    print(f"Average rating: {user_info['avg_rating']:.2f}")
    
    print("\nTop genres:")
    for genre, count in user_info['top_genres'].items():
        print(f"  - {genre}: {count}")
    
    print("\nFavorite movies:")
    for _, movie in user_info['favorite_movies'].iterrows():
        print(f"  - {movie['title']} (Rating: {movie['rating']})")
    
    # Plot user profile
    if args.plot:
        print("\nPlotting user profile...")
        plot_user_profile(user_info)
    
    # Get recommendations
    print("\nGenerating recommendations...")
    recommendations = compare_recommendations(
        user_id=user_id,
        models=models,
        edge_index=edge_index,
        data=data,
        num_recommendations=args.num_recommendations
    )
    
    # Print recommendations
    print("\nTop Recommendations:")
    for model_name in models.keys():
        score_col = f'{model_name}_score'
        if score_col in recommendations.columns:
            print(f"\n{model_name.upper()} Model Recommendations:")
            model_recs = recommendations.sort_values(score_col, ascending=False)[['title', score_col]]
            for i, (_, rec) in enumerate(model_recs.iterrows(), 1):
                print(f"  {i}. {rec['title']} (Predicted Rating: {rec[score_col]:.2f})")
    
    # Predict specific movies if requested
    if args.movie_ids:
        print("\nPredicting ratings for specific movies...")
        predictions = predict_specific_movies(
            user_id=user_id,
            movie_ids=args.movie_ids,
            models=models,
            edge_index=edge_index,
            data=data
        )
        
        if predictions is not None:
            print("\nPredicted Ratings for Specific Movies:")
            for _, pred in predictions.iterrows():
                print(f"\nMovie: {pred['title']}")
                for model_name in models.keys():
                    score_col = f'{model_name}_score'
                    if score_col in pred:
                        print(f"  {model_name.upper()} prediction: {pred[score_col]:.2f}")
        else:
            print("Error: Could not predict ratings for the specified movies.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate movie recommendations with trained models")
    
    # Path arguments
    parser.add_argument("--data_dir", type=str, default="../data/processed", help="Directory containing processed data")
    parser.add_argument("--models_dir", type=str, default="../results", help="Directory containing trained models")
    
    # Model arguments
    parser.add_argument("--models", type=str, nargs="+", default=["ncf", "gat", "ensemble"], help="Models to use for recommendations")
    
    # User arguments
    parser.add_argument("--user_id", type=int, help="User ID to generate recommendations for (random if not specified)")
    parser.add_argument("--movie_ids", type=int, nargs="+", help="Specific movie IDs to predict ratings for")
    
    # Display arguments
    parser.add_argument("--num_recommendations", type=int, default=10, help="Number of recommendations to show")
    parser.add_argument("--plot", action="store_true", help="Plot user profile and recommendations")
    
    args = parser.parse_args()
    main(args)
