#!/usr/bin/env python
"""
Test script for running recommendation models on a small dataset
"""
import os
import argparse
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models.ncf import NCF
from src.models.gat import GATModel
from src.models.ensemble import EnsembleModel
from src.utils.data_utils import create_edge_index

def create_test_dataset(size=1000, n_users=100, n_items=50, sparsity=0.1):
    """
    Create a synthetic test dataset
    
    Parameters:
    -----------
    size : int
        Number of ratings
    n_users : int
        Number of users
    n_items : int
        Number of items
    sparsity : float
        Sparsity of the rating matrix
        
    Returns:
    --------
    dict
        Dictionary containing train, validation, and test DataFrames
    """
    print(f"Creating test dataset with {n_users} users and {n_items} items")
    
    # Generate user and item indices
    n_samples = min(int(n_users * n_items * sparsity), size)
    user_indices = np.random.randint(0, n_users, n_samples)
    item_indices = np.random.randint(0, n_items, n_samples)
    
    # Generate ratings (0 to 1)
    ratings = np.random.rand(n_samples)
    
    # Create dataframe
    df = pd.DataFrame({
        'userIdx': user_indices,
        'movieIdx': item_indices,
        'rating_norm': ratings
    })
    
    # Remove duplicates to avoid having multiple ratings for same user-item pair
    df = df.drop_duplicates(subset=['userIdx', 'movieIdx'])
    
    # Create user and movie ID mappings (simulate real IDs with offset)
    user_ids = np.arange(1, n_users + 1) * 10
    movie_ids = np.arange(1, n_items + 1) * 5
    
    # Create mappings
    user_to_idx = {int(user_ids[i]): i for i in range(n_users)}
    movie_to_idx = {int(movie_ids[i]): i for i in range(n_items)}
    
    # Add IDs to dataframe
    df['userId'] = df['userIdx'].map({i: user_ids[i] for i in range(n_users)})
    df['movieId'] = df['movieIdx'].map({i: movie_ids[i] for i in range(n_items)})
    
    # Create movie titles (for display purposes)
    movie_titles = {movie_ids[i]: f"Test Movie {movie_ids[i]}" for i in range(n_items)}
    movie_df = pd.DataFrame({
        'movieId': list(movie_titles.keys()),
        'title': list(movie_titles.values()),
        'genres': ['Test Genre'] * n_items
    })
    
    # Split into train, validation and test
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size+val_size].copy()
    test_df = df.iloc[train_size+val_size:].copy()
    
    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")
    
    return {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'movies_df': movie_df,
        'mappings': {
            'user_to_idx': user_to_idx,
            'movie_to_idx': movie_to_idx,
            'idx_to_user': {idx: user_id for user_id, idx in user_to_idx.items()},
            'idx_to_movie': {idx: movie_id for movie_id, idx in movie_to_idx.items()},
            'rating_min': 0.0,
            'rating_max': 1.0
        }
    }

def create_data_loaders(data, batch_size=32):
    """
    Create PyTorch DataLoaders from pandas DataFrames
    
    Parameters:
    -----------
    data : dict
        Dictionary containing train_df, val_df, and test_df
    batch_size : int
        Batch size
        
    Returns:
    --------
    dict
        Dictionary containing DataLoaders
    """
    # Create tensors
    train_users = torch.tensor(data['train_df']['userIdx'].values, dtype=torch.long)
    train_items = torch.tensor(data['train_df']['movieIdx'].values, dtype=torch.long)
    train_ratings = torch.tensor(data['train_df']['rating_norm'].values, dtype=torch.float)
    
    val_users = torch.tensor(data['val_df']['userIdx'].values, dtype=torch.long)
    val_items = torch.tensor(data['val_df']['movieIdx'].values, dtype=torch.long)
    val_ratings = torch.tensor(data['val_df']['rating_norm'].values, dtype=torch.float)
    
    test_users = torch.tensor(data['test_df']['userIdx'].values, dtype=torch.long)
    test_items = torch.tensor(data['test_df']['movieIdx'].values, dtype=torch.long)
    test_ratings = torch.tensor(data['test_df']['rating_norm'].values, dtype=torch.float)
    
    # Create datasets
    train_dataset = TensorDataset(train_users, train_items, train_ratings)
    val_dataset = TensorDataset(val_users, val_items, val_ratings)
    test_dataset = TensorDataset(test_users, test_items, test_ratings)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'num_users': len(data['mappings']['user_to_idx']),
        'num_items': len(data['mappings']['movie_to_idx'])
    }

def train_model(model, train_loader, val_loader, edge_index, num_epochs=3, lr=0.001, 
               model_name='model', device='cpu'):
    """
    Train a recommendation model (simplified version for testing)
    
    Parameters:
    -----------
    model : torch.nn.Module
        Model to train
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data
    edge_index : torch.Tensor
        Edge index tensor for graph models
    num_epochs : int
        Number of training epochs
    lr : float
        Learning rate
    model_name : str
        Name of the model for printing
    device : str
        Device to use for training ('cpu' or 'cuda')
        
    Returns:
    --------
    tuple
        (train_loss, val_loss, training_time)
    """
    # Move model to device
    model = model.to(device)
    if edge_index is not None:
        edge_index = edge_index.to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for user_batch, item_batch, rating_batch in train_loader:
            # Move batches to device
            user_batch = user_batch.to(device)
            item_batch = item_batch.to(device)
            rating_batch = rating_batch.to(device)
            
            # Forward pass
            try:
                if isinstance(model, GATModel) or isinstance(model, EnsembleModel):
                    # For GAT or Ensemble models that can use edge_index
                    if edge_index is not None:
                        predictions = model(user_batch, item_batch, edge_index)
                    else:
                        predictions = model(user_batch, item_batch)
                else:
                    # For NCF model that doesn't use edge_index
                    predictions = model(user_batch, item_batch)
            except Exception as e:
                print(f"Error in forward pass: {e}")
                break
            
            loss = criterion(predictions, rating_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * user_batch.size(0)
        
        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for user_batch, item_batch, rating_batch in val_loader:
                # Move batches to device
                user_batch = user_batch.to(device)
                item_batch = item_batch.to(device)
                rating_batch = rating_batch.to(device)
                
                # Forward pass
                try:
                    if isinstance(model, GATModel) or isinstance(model, EnsembleModel):
                        # For GAT or Ensemble models that can use edge_index
                        if edge_index is not None:
                            predictions = model(user_batch, item_batch, edge_index)
                        else:
                            predictions = model(user_batch, item_batch)
                    else:
                        # For NCF model that doesn't use edge_index
                        predictions = model(user_batch, item_batch)
                except Exception as e:
                    print(f"Error in validation forward pass: {e}")
                    continue
                
                loss = criterion(predictions, rating_batch)
                
                # Update statistics
                val_loss += loss.item() * user_batch.size(0)
        
        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)
        
        print(f"{model_name} - Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    training_time = time.time() - start_time
    print(f"{model_name} - Training completed in {training_time:.2f} seconds")
    
    return train_loss, val_loss, training_time

def generate_sample_recommendations(model, data, edge_index, device, model_name):
    """
    Generate sample recommendations for a few users
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model
    data : dict
        Dictionary containing data
    edge_index : torch.Tensor
        Edge index tensor for graph models
    device : str
        Device for computation
    model_name : str
        Name of the model for display
    """
    print(f"\nSample recommendations from {model_name} model:")
    model.eval()
    
    # Take a few sample users
    sample_users = data['test_df']['userId'].unique()[:3]
    
    for user_id in sample_users:
        user_idx = data['mappings']['user_to_idx'][user_id]
        print(f"\nRecommendations for User {user_id} (Index {user_idx}):")
        
        # Get all items
        all_items = torch.tensor(list(data['mappings']['movie_to_idx'].values()), dtype=torch.long).to(device)
        all_user = torch.tensor([user_idx] * len(all_items), dtype=torch.long).to(device)
        
        # Get predictions
        with torch.no_grad():
            try:
                if isinstance(model, GATModel) or isinstance(model, EnsembleModel):
                    # For GAT or Ensemble models that can use edge_index
                    if edge_index is not None:
                        predictions = model(all_user, all_items, edge_index)
                    else:
                        predictions = model(all_user, all_items)
                else:
                    # For NCF model that doesn't use edge_index
                    predictions = model(all_user, all_items)
            except Exception as e:
                print(f"Error generating recommendations: {e}")
                continue
        
        # Convert to numpy
        predictions = predictions.cpu().numpy()
        
        # Create dataframe
        idx_to_movie = data['mappings']['idx_to_movie']
        recs_df = pd.DataFrame({
            'movieIdx': all_items.cpu().numpy(),
            'prediction': predictions
        })
        
        # Map indices to IDs
        recs_df['movieId'] = recs_df['movieIdx'].map(idx_to_movie)
        
        # Sort by prediction
        recs_df = recs_df.sort_values('prediction', ascending=False)
        
        # Get top 5
        top_recs = recs_df.head(5)
        
        # Merge with movies dataframe
        top_recs = top_recs.merge(data['movies_df'], on='movieId')
        
        # Print recommendations
        for _, row in top_recs.iterrows():
            print(f"  - {row['title']} (Score: {row['prediction']:.4f})")

def test_all_models(args):
    """
    Test all recommendation models
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    print("=" * 80)
    print("TESTING RECOMMENDATION MODELS ON SYNTHETIC DATASET")
    print("=" * 80)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create test dataset
    data = create_test_dataset(
        size=args.dataset_size,
        n_users=args.n_users,
        n_items=args.n_items,
        sparsity=args.sparsity
    )
    
    # Create data loaders
    loaders = create_data_loaders(data, batch_size=args.batch_size)
    
    # Create edge index for graph models
    edge_index = create_edge_index(data['train_df'], loaders['num_users'])
    
    # Results storage
    results = []
    
    # Test NCF model
    if 'ncf' in args.models:
        print("\nTesting NCF model...")
        ncf_model = NCF(
            num_users=loaders['num_users'],
            num_items=loaders['num_items'],
            embedding_dim=args.embedding_dim,
            layers=[args.layer_size] * args.num_layers,
            dropout=args.dropout
        )
        print(f"NCF model parameters: {sum(p.numel() for p in ncf_model.parameters())}")
        
        ncf_train_loss, ncf_val_loss, ncf_time = train_model(
            model=ncf_model,
            train_loader=loaders['train_loader'],
            val_loader=loaders['val_loader'],
            edge_index=None,
            num_epochs=args.epochs,
            lr=args.learning_rate,
            model_name='NCF',
            device=device
        )
        
        # Generate sample recommendations
        generate_sample_recommendations(ncf_model, data, None, device, 'NCF')
        
        results.append({
            'Model': 'NCF', 
            'Train Loss': f"{ncf_train_loss:.4f}", 
            'Val Loss': f"{ncf_val_loss:.4f}", 
            'Time (s)': f"{ncf_time:.2f}"
        })
    
    # Test GAT model
    if 'gat' in args.models:
        print("\nTesting GAT model...")
        gat_model = GATModel(
            num_users=loaders['num_users'],
            num_items=loaders['num_items'],
            embedding_dim=args.embedding_dim,
            heads=args.gat_heads,
            dropout=args.dropout,
            num_layers=args.gat_layers,
            residual=args.gat_residual,
            subsampling_rate=args.gat_subsampling_rate
        )
        print(f"GAT model parameters: {sum(p.numel() for p in gat_model.parameters())}")
        
        gat_train_loss, gat_val_loss, gat_time = train_model(
            model=gat_model,
            train_loader=loaders['train_loader'],
            val_loader=loaders['val_loader'],
            edge_index=edge_index,
            num_epochs=args.epochs,
            lr=args.learning_rate,
            model_name='GAT',
            device=device
        )
        
        # Generate sample recommendations
        generate_sample_recommendations(gat_model, data, edge_index, device, 'GAT')
        
        results.append({
            'Model': 'GAT', 
            'Train Loss': f"{gat_train_loss:.4f}", 
            'Val Loss': f"{gat_val_loss:.4f}", 
            'Time (s)': f"{gat_time:.2f}"
        })
    
    # Test Ensemble model
    if 'ensemble' in args.models:
        print("\nTesting Ensemble model...")
        ensemble_model = EnsembleModel(
            num_users=loaders['num_users'],
            num_items=loaders['num_items'],
            embedding_dim=args.embedding_dim,
            mlp_layers=[args.layer_size] * args.num_layers,
            gat_heads=args.gat_heads,
            dropout=args.dropout,
            share_embeddings=args.share_embeddings,
            ensemble_method=args.ensemble_method,
            gat_layers=args.gat_layers,
            gat_residual=args.gat_residual,
            gat_subsampling_rate=args.gat_subsampling_rate
        )
        print(f"Ensemble model parameters: {sum(p.numel() for p in ensemble_model.parameters())}")
        
        ens_train_loss, ens_val_loss, ens_time = train_model(
            model=ensemble_model,
            train_loader=loaders['train_loader'],
            val_loader=loaders['val_loader'],
            edge_index=edge_index,
            num_epochs=args.epochs,
            lr=args.learning_rate,
            model_name='Ensemble',
            device=device
        )
        
        # Generate sample recommendations
        generate_sample_recommendations(ensemble_model, data, edge_index, device, 'Ensemble')
        
        results.append({
            'Model': 'Ensemble', 
            'Train Loss': f"{ens_train_loss:.4f}", 
            'Val Loss': f"{ens_val_loss:.4f}", 
            'Time (s)': f"{ens_time:.2f}"
        })
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    # Print results as a table
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("\nTest completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test recommendation models on a small dataset")
    
    # Dataset arguments
    parser.add_argument("--dataset_size", type=int, default=1000, help="Number of ratings in test dataset")
    parser.add_argument("--n_users", type=int, default=100, help="Number of users in test dataset")
    parser.add_argument("--n_items", type=int, default=50, help="Number of items in test dataset")
    parser.add_argument("--sparsity", type=float, default=0.1, help="Sparsity of rating matrix")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/test", help="Directory to save results")
    
    # Model selection
    parser.add_argument("--models", type=str, nargs="+", default=["ncf", "gat", "ensemble"], 
                        choices=["ncf", "gat", "ensemble"], help="Models to test")
    
    # Model arguments
    parser.add_argument("--embedding_dim", type=int, default=16, help="Dimension of embeddings")
    parser.add_argument("--layer_size", type=int, default=32, help="Size of MLP layers")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of MLP layers")
    parser.add_argument("--gat_heads", type=int, default=2, help="Number of attention heads for GAT")
    parser.add_argument("--gat_layers", type=int, default=3, help="Number of GAT layers")
    parser.add_argument("--gat_residual", action="store_true", help="Use residual connections in GAT")
    parser.add_argument("--gat_subsampling_rate", type=float, default=0.8, help="Subsampling rate for large edge indices in GAT")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--share_embeddings", action="store_true", help="Share embeddings between NCF and GAT in ensemble model")
    parser.add_argument("--ensemble_method", type=str, default="weighted", 
                        choices=["weighted", "concat", "gate"], help="Method for ensemble combination")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    
    args = parser.parse_args()
    test_all_models(args)