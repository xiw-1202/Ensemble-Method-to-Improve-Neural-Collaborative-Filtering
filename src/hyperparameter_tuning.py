#!/usr/bin/env python
"""
Hyperparameter tuning for recommendation models
"""
import os
import argparse
import time
import pickle
import itertools
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
from train import train_model

def grid_search(model_class, param_grid, train_loader, val_loader, edge_index, 
               num_epochs=10, patience=3, device='cpu', output_dir='../results'):
    """
    Perform grid search over hyperparameters
    
    Parameters:
    -----------
    model_class : class
        Model class to tune
    param_grid : dict
        Dictionary of parameter names and possible values
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data
    edge_index : torch.Tensor
        Edge index tensor for graph models
    num_epochs : int
        Number of training epochs
    patience : int
        Patience for early stopping
    device : str
        Device to use for training
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    tuple
        Best parameters and all results
    """
    # Create directory for tuning results
    tuning_dir = os.path.join(output_dir, 'tuning')
    os.makedirs(tuning_dir, exist_ok=True)
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"Total parameter combinations: {len(param_combinations)}")
    
    # Store results
    results = []
    best_val_loss = float('inf')
    best_params = None
    
    # Iterate over all parameter combinations
    for i, param_combination in enumerate(param_combinations):
        # Create parameter dictionary
        params = dict(zip(param_names, param_combination))
        param_str = ', '.join(f"{name}={value}" for name, value in params.items())
        print(f"\nTrying parameters ({i+1}/{len(param_combinations)}): {param_str}")
        
        # Create model with these parameters
        try:
            model = model_class(**params)
        except TypeError as e:
            print(f"Error creating model: {e}")
            continue
        
        # Train model
        model_path = os.path.join(tuning_dir, f"model_{i}.pth")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            edge_index=edge_index,
            num_epochs=num_epochs,
            patience=patience,
            model_path=model_path,
            device=device
        )
        
        # Get validation loss
        final_val_loss = history['val_loss'][-1]
        
        # Store results
        result = {
            'params': params,
            'val_loss': final_val_loss,
            'epochs': len(history['val_loss']),
            'model_path': model_path
        }
        results.append(result)
        
        # Update best parameters
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_params = params
            print(f"New best validation loss: {best_val_loss:.4f}")
    
    # Sort results by validation loss
    results.sort(key=lambda x: x['val_loss'])
    
    # Save results
    results_path = os.path.join(tuning_dir, 'tuning_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({'best_params': best_params, 'all_results': results}, f)
    
    # Print best parameters
    print("\nBest parameters:")
    for name, value in best_params.items():
        print(f"{name}: {value}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return best_params, results

def plot_tuning_results(results, param_names, output_dir):
    """
    Plot tuning results for visualization
    
    Parameters:
    -----------
    results : list
        List of dictionaries containing tuning results
    param_names : list
        List of parameter names to plot
    output_dir : str
        Directory to save plots
    """
    # Convert results to dataframe
    df = pd.DataFrame([{**r['params'], 'val_loss': r['val_loss']} for r in results])
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'tuning', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot validation loss for each parameter
    for param in param_names:
        if param in df.columns:
            plt.figure(figsize=(10, 6))
            
            # Group by parameter value and calculate mean loss
            param_loss = df.groupby(param)['val_loss'].mean().reset_index()
            
            # Sort by parameter value
            param_loss = param_loss.sort_values(param)
            
            # Plot
            plt.plot(param_loss[param], param_loss['val_loss'], 'o-')
            plt.title(f'Validation Loss vs {param}')
            plt.xlabel(param)
            plt.ylabel('Validation Loss')
            plt.grid(True)
            
            # Save plot
            plot_path = os.path.join(plots_dir, f'{param}_vs_loss.png')
            plt.savefig(plot_path)
            plt.close()
    
    # For pairs of parameters, create heatmaps
    if len(param_names) >= 2:
        for i, param1 in enumerate(param_names):
            for param2 in param_names[i+1:]:
                if param1 in df.columns and param2 in df.columns:
                    # Group by both parameters and calculate mean loss
                    heatmap_data = df.groupby([param1, param2])['val_loss'].mean().reset_index()
                    
                    # Create pivot table
                    pivot_table = heatmap_data.pivot(index=param1, columns=param2, values='val_loss')
                    
                    # Plot heatmap
                    plt.figure(figsize=(10, 8))
                    plt.imshow(pivot_table.values, cmap='viridis', aspect='auto', interpolation='nearest')
                    plt.colorbar(label='Validation Loss')
                    
                    # Set ticks and labels
                    plt.yticks(range(len(pivot_table.index)), pivot_table.index)
                    plt.xticks(range(len(pivot_table.columns)), pivot_table.columns, rotation=45)
                    
                    plt.title(f'Validation Loss Heatmap: {param1} vs {param2}')
                    plt.xlabel(param2)
                    plt.ylabel(param1)
                    
                    # Save plot
                    plot_path = os.path.join(plots_dir, f'{param1}_vs_{param2}_heatmap.png')
                    plt.tight_layout()
                    plt.savefig(plot_path)
                    plt.close()

def main(args):
    """
    Main function for hyperparameter tuning
    
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
    if device.type == 'cuda':
        edge_index = edge_index.to(device)
    
    # Define parameter grids for each model
    if args.model == 'ncf':
        model_class = NCF
        param_grid = {
            'num_users': [data['num_users']],
            'num_items': [data['num_items']],
            'embedding_dim': [32, 64, 128],
            'layers': [[128, 64, 32], [256, 128, 64], [64, 32, 16]],
            'dropout': [0.1, 0.2, 0.3]
        }
    elif args.model == 'gat':
        model_class = GATModel
        param_grid = {
            'num_users': [data['num_users']],
            'num_items': [data['num_items']],
            'embedding_dim': [32, 64, 128],
            'heads': [2, 4, 8],
            'dropout': [0.1, 0.2, 0.3]
        }
    elif args.model == 'ensemble':
        model_class = EnsembleModel
        param_grid = {
            'num_users': [data['num_users']],
            'num_items': [data['num_items']],
            'embedding_dim': [32, 64, 128],
            'mlp_layers': [[128, 64, 32], [256, 128, 64], [64, 32, 16]],
            'gat_heads': [2, 4, 8],
            'dropout': [0.1, 0.2, 0.3],
            'share_embeddings': [True, False],
            'ensemble_method': ['weighted', 'concat', 'gate']
        }
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Reduce parameter grid if specified
    if args.reduced_grid:
        print("Using reduced parameter grid")
        for param, values in param_grid.items():
            if param not in ['num_users', 'num_items'] and len(values) > 1:
                param_grid[param] = [values[len(values) // 2]]  # Choose middle value
    
    # If custom parameter grid is provided
    if args.param_grid:
        print("Using custom parameter grid")
        param_grid.update(args.param_grid)
    
    # Run grid search
    print(f"Running grid search for {args.model} model...")
    best_params, results = grid_search(
        model_class=model_class,
        param_grid=param_grid,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        edge_index=edge_index,
        num_epochs=args.epochs,
        patience=args.patience,
        device=device,
        output_dir=args.output_dir
    )
    
    # Plot results
    print("Plotting tuning results...")
    plot_tuning_results(results, list(param_grid.keys()), args.output_dir)
    
    # Train final model with best parameters
    print("Training final model with best parameters...")
    final_model = model_class(**best_params)
    
    model_path = os.path.join(args.output_dir, f"{args.model}_tuned_model.pth")
    history = train_model(
        model=final_model,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        edge_index=edge_index,
        num_epochs=args.final_epochs,
        patience=args.patience,
        model_path=model_path,
        device=device
    )
    
    # Evaluate final model
    print("Evaluating final model...")
    final_model.load_state_dict(torch.load(model_path, map_location=device))
    
    metrics = evaluate_recommendations(
        model=final_model,
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
    metrics_path = os.path.join(args.output_dir, f"{args.model}_tuned_metrics.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for recommendation models")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"), help="Directory containing processed data")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"), help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="ensemble", choices=["ncf", "gat", "ensemble"], help="Model to tune")
    parser.add_argument("--reduced_grid", action="store_true", help="Use reduced parameter grid for faster tuning")
    parser.add_argument("--param_grid", type=dict, default=None, help="Custom parameter grid (overrides defaults)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs for each trial")
    parser.add_argument("--final_epochs", type=int, default=30, help="Number of training epochs for final model")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    
    args = parser.parse_args()
    main(args)
