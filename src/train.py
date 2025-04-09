#!/usr/bin/env python
"""
Training script for recommendation models
"""
import os
import argparse
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.ncf import NCF
from models.gat import GATModel
from models.ensemble import EnsembleModel
from utils.data_utils import load_data, create_edge_index
from evaluation.metrics import evaluate_recommendations

def train_model(model, train_loader, val_loader, edge_index, num_epochs=30, lr=0.001, weight_decay=1e-5, 
               patience=5, model_path='model.pth', device='cpu'):
    """
    Train a recommendation model
    
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
    weight_decay : float
        L2 regularization strength
    patience : int
        Patience for early stopping
    model_path : str
        Path to save the best model
    device : str
        Device to use for training ('cpu' or 'cuda')
        
    Returns:
    --------
    dict
        Dictionary containing training history
    """
    # Move model to device
    model = model.to(device)
    if edge_index is not None:
        edge_index = edge_index.to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch_times': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Start timer
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for user_batch, item_batch, rating_batch in train_pbar:
            # Move batches to device
            user_batch = user_batch.to(device)
            item_batch = item_batch.to(device)
            rating_batch = rating_batch.to(device)
            
            # Forward pass
            try:
                predictions = model(user_batch, item_batch, edge_index)
            except TypeError:
                # If model doesn't support edge_index (e.g., NCF)
                predictions = model(user_batch, item_batch)
            
            loss = criterion(predictions, rating_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * user_batch.size(0)
            train_pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
        
        with torch.no_grad():
            for user_batch, item_batch, rating_batch in val_pbar:
                # Move batches to device
                user_batch = user_batch.to(device)
                item_batch = item_batch.to(device)
                rating_batch = rating_batch.to(device)
                
                # Forward pass
                try:
                    predictions = model(user_batch, item_batch, edge_index)
                except TypeError:
                    # If model doesn't support edge_index (e.g., NCF)
                    predictions = model(user_batch, item_batch)
                
                loss = criterion(predictions, rating_batch)
                
                # Update statistics
                val_loss += loss.item() * user_batch.size(0)
                val_pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)
        
        # End timer
        epoch_time = time.time() - start_time
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epoch_times'].append(epoch_time)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save the model
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return history

def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Parameters:
    -----------
    history : dict
        Dictionary containing training history
    save_path : str
        Path to save the plot (if None, plot is displayed)
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot epoch times
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch_times'])
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title('Epoch Training Time')
    
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main(args):
    """
    Main function for training and evaluating recommendation models
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = load_data(args.data_dir, args.batch_size)
    
    # Create edge index for graph models
    edge_index = create_edge_index(data['train_df'], data['num_users'])
    
    # Create model
    print(f"Creating {args.model} model...")
    if args.model == 'ncf':
        model = NCF(
            num_users=data['num_users'],
            num_items=data['num_items'],
            embedding_dim=args.embedding_dim,
            layers=[args.layer_sizes] * args.num_layers if isinstance(args.layer_sizes, int) else args.layer_sizes,
            dropout=args.dropout
        )
    elif args.model == 'gat':
        model = GATModel(
            num_users=data['num_users'],
            num_items=data['num_items'],
            embedding_dim=args.embedding_dim,
            heads=args.gat_heads,
            dropout=args.dropout
        )
    elif args.model == 'ensemble':
        model = EnsembleModel(
            num_users=data['num_users'],
            num_items=data['num_items'],
            embedding_dim=args.embedding_dim,
            mlp_layers=[args.layer_sizes] * args.num_layers if isinstance(args.layer_sizes, int) else args.layer_sizes,
            gat_heads=args.gat_heads,
            dropout=args.dropout,
            share_embeddings=args.share_embeddings,
            ensemble_method=args.ensemble_method
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Print model summary
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("Training model...")
    model_path = os.path.join(args.output_dir, f"{args.model}_model.pth")
    history = train_model(
        model=model,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        edge_index=edge_index,
        num_epochs=args.epochs,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        model_path=model_path,
        device=device
    )
    
    # Plot training history
    history_plot_path = os.path.join(args.output_dir, f"{args.model}_history.png")
    plot_training_history(history, history_plot_path)
    
    # Save training history
    history_path = os.path.join(args.output_dir, f"{args.model}_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    # Load best model
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_recommendations(
        model=model,
        test_data=data['test_df'],
        user_mapping=data['mappings']['user_to_idx'],
        item_mapping=data['mappings']['movie_to_idx'],
        edge_index=edge_index,
        rating_threshold=3.5,
        k_values=[5, 10, 20]
    )
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f"{args.model}_metrics.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    # Print recommendations for a few sample users
    print("\nSample Recommendations:")
    for user_id in data['test_df']['userId'].unique()[:3]:
        user_idx = data['mappings']['user_to_idx'][user_id]
        print(f"\nRecommendations for User {user_id} (Index {user_idx}):")
        
        # Get recommendations
        user_tensor = torch.tensor([user_idx], dtype=torch.long).to(device)
        
        # Get all items
        all_items = torch.tensor(list(data['mappings']['movie_to_idx'].values()), dtype=torch.long).to(device)
        all_user = torch.tensor([user_idx] * len(all_items), dtype=torch.long).to(device)
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            try:
                predictions = model(all_user, all_items, edge_index)
            except TypeError:
                predictions = model(all_user, all_items)
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate recommendation models")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"), help="Directory containing processed data")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"), help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="ensemble", choices=["ncf", "gat", "ensemble"], help="Model to train")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Dimension of embeddings")
    parser.add_argument("--layer_sizes", type=int, nargs="+", default=[128, 64, 32], help="Sizes of MLP layers")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of MLP layers (if layer_sizes is a single value)")
    parser.add_argument("--gat_heads", type=int, default=4, help="Number of attention heads for GAT")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--share_embeddings", action="store_true", help="Share embeddings between NCF and GAT in ensemble model")
    parser.add_argument("--ensemble_method", type=str, default="weighted", choices=["weighted", "concat", "gate"], help="Method for ensemble combination")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 regularization)")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    
    args = parser.parse_args()
    main(args)
