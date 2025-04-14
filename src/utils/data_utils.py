#!/usr/bin/env python
"""
Utility functions for data loading and processing
"""
import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(data_dir, batch_size=128):
    """
    Load processed data and create DataLoaders
    
    Parameters:
    -----------
    data_dir : str
        Directory containing processed data
    batch_size : int
        Batch size for DataLoaders
        
    Returns:
    --------
    dict
        Dictionary containing DataLoaders, mappings, and metadata
    """
    # Convert relative path to absolute path if needed
    if not os.path.isabs(data_dir):
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        data_dir = os.path.join(project_dir, data_dir)
    
    # Check if the directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
    # Print actual path for debugging
    print(f"Loading data from: {data_dir}")
    
    # Load dataframes
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'validation.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    movies_df = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
    
    # Load mappings
    with open(os.path.join(data_dir, 'mappings.pkl'), 'rb') as f:
        mappings = pickle.load(f)
    
    # Create tensors for train data
    train_users = torch.tensor(train_df['userIdx'].values, dtype=torch.long)
    train_items = torch.tensor(train_df['movieIdx'].values, dtype=torch.long)
    train_ratings = torch.tensor(train_df['rating_norm'].values, dtype=torch.float)
    
    # Create tensors for validation data
    val_users = torch.tensor(val_df['userIdx'].values, dtype=torch.long)
    val_items = torch.tensor(val_df['movieIdx'].values, dtype=torch.long)
    val_ratings = torch.tensor(val_df['rating_norm'].values, dtype=torch.float)
    
    # Create tensors for test data
    test_users = torch.tensor(test_df['userIdx'].values, dtype=torch.long)
    test_items = torch.tensor(test_df['movieIdx'].values, dtype=torch.long)
    test_ratings = torch.tensor(test_df['rating_norm'].values, dtype=torch.float)
    
    # Create datasets
    train_dataset = TensorDataset(train_users, train_items, train_ratings)
    val_dataset = TensorDataset(val_users, val_items, val_ratings)
    test_dataset = TensorDataset(test_users, test_items, test_ratings)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get dimensions
    num_users = len(mappings['user_to_idx'])
    num_items = len(mappings['movie_to_idx'])
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'movies_df': movies_df,
        'mappings': mappings,
        'num_users': num_users,
        'num_items': num_items
    }

def create_edge_index(train_df, num_users):
    """
    Create edge index tensor for graph neural networks
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training data containing user-item interactions
    num_users : int
        Number of users in the dataset
        
    Returns:
    --------
    torch.Tensor
        Edge index tensor (2 x num_edges)
    """
    # Extract user and item indices
    users = train_df['userIdx'].values
    items = train_df['movieIdx'].values
    
    # Create edge index
    # Edge direction: user -> item
    edge_index_user_to_item = torch.tensor([
        users,
        items + num_users  # Offset for item indices
    ], dtype=torch.long)
    
    # Edge direction: item -> user (for message passing in both directions)
    edge_index_item_to_user = torch.tensor([
        items + num_users,  # Offset for item indices
        users
    ], dtype=torch.long)
    
    # Combine both directions
    edge_index = torch.cat([edge_index_user_to_item, edge_index_item_to_user], dim=1)
    
    return edge_index

def get_recommendations(model, user_id, mappings, edge_index, movies_df, n=10, exclude_seen=True, train_df=None):
    """
    Get top N recommendations for a specific user
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained recommendation model
    user_id : int
        Original user ID
    mappings : dict
        Dictionary containing mapping from IDs to indices
    edge_index : torch.Tensor
        Edge index tensor for graph models
    movies_df : pandas.DataFrame
        DataFrame containing movie metadata
    n : int
        Number of recommendations to return
    exclude_seen : bool
        Whether to exclude movies the user has already seen
    train_df : pandas.DataFrame
        Training data containing user-item interactions (for seen exclusion)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing top N recommendations
    """
    model.eval()
    
    # Convert user ID to index
    user_idx = mappings['user_to_idx'].get(user_id)
    
    if user_idx is None:
        raise ValueError(f"User ID {user_id} not found in mappings")
    
    # Create tensor for all movies
    all_movie_indices = list(mappings['movie_to_idx'].values())
    all_movies_tensor = torch.tensor(all_movie_indices, dtype=torch.long)
    
    # Create user tensor of the same length as movies tensor
    user_tensor = torch.tensor([user_idx] * len(all_movie_indices), dtype=torch.long)
    
    # Get predictions
    with torch.no_grad():
        try:
            from models.gat import GATModel
            from models.ensemble import EnsembleModel
            
            if isinstance(model, GATModel) or isinstance(model, EnsembleModel):
                # For GAT or Ensemble models that can use edge_index
                if edge_index is not None:
                    predictions = model(user_tensor, all_movies_tensor, edge_index)
                else:
                    predictions = model(user_tensor, all_movies_tensor)
            else:
                # For NCF model that doesn't use edge_index
                predictions = model(user_tensor, all_movies_tensor)
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            raise
    
    # Create a dataframe with movie indices and predictions
    recs_df = pd.DataFrame({
        'movieIdx': all_movie_indices,
        'prediction': predictions.cpu().numpy()
    })
    
    # Exclude movies the user has already seen if requested
    if exclude_seen and train_df is not None:
        seen_movie_indices = train_df[train_df['userIdx'] == user_idx]['movieIdx'].values
        recs_df = recs_df[~recs_df['movieIdx'].isin(seen_movie_indices)]
    
    # Sort by prediction score
    recs_df = recs_df.sort_values('prediction', ascending=False)
    
    # Get top N
    recs_df = recs_df.head(n)
    
    # Convert movieIdx back to movieId
    idx_to_movie = mappings['idx_to_movie']
    recs_df['movieId'] = recs_df['movieIdx'].map(idx_to_movie)
    
    # Merge with movies dataframe to get movie metadata
    recs_df = recs_df.merge(movies_df, on='movieId')
    
    return recs_df[['movieId', 'title', 'genres', 'prediction']]

def denormalize_rating(rating_norm, min_rating, max_rating):
    """
    Convert normalized rating back to original scale
    
    Parameters:
    -----------
    rating_norm : float
        Normalized rating (0-1)
    min_rating : float
        Minimum rating in original scale
    max_rating : float
        Maximum rating in original scale
        
    Returns:
    --------
    float
        Rating in original scale
    """
    return min_rating + rating_norm * (max_rating - min_rating)
