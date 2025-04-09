#!/usr/bin/env python
"""
Compare different recommendation models
"""
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def load_metrics(results_dir, model_names):
    """
    Load evaluation metrics for different models
    
    Parameters:
    -----------
    results_dir : str
        Directory containing results
    model_names : list
        List of model names to compare
        
    Returns:
    --------
    dict
        Dictionary containing metrics for each model
    """
    metrics = {}
    
    for model_name in model_names:
        # Try to load metrics from original training
        metrics_path = os.path.join(results_dir, f"{model_name}_metrics.pkl")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                metrics[model_name] = pickle.load(f)
        
        # Try to load metrics from tuned model
        tuned_metrics_path = os.path.join(results_dir, f"{model_name}_tuned_metrics.pkl")
        if os.path.exists(tuned_metrics_path):
            with open(tuned_metrics_path, 'rb') as f:
                metrics[f"{model_name}_tuned"] = pickle.load(f)
    
    return metrics

def plot_metrics_comparison(metrics, k_values, save_dir=None):
    """
    Plot comparison of metrics for different models
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing metrics for each model
    k_values : list
        List of k values for @k metrics
    save_dir : str
        Directory to save plots
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data for plotting
    plot_data = {
        'precision': {},
        'recall': {},
        'ndcg': {}
    }
    
    # Extract metrics for each model
    for model_name, model_metrics in metrics.items():
        for k in k_values:
            # Precision
            key = f'precision@{k}'
            if key in model_metrics:
                if k not in plot_data['precision']:
                    plot_data['precision'][k] = {}
                plot_data['precision'][k][model_name] = model_metrics[key]
            
            # Recall
            key = f'recall@{k}'
            if key in model_metrics:
                if k not in plot_data['recall']:
                    plot_data['recall'][k] = {}
                plot_data['recall'][k][model_name] = model_metrics[key]
            
            # NDCG
            key = f'ndcg@{k}'
            if key in model_metrics:
                if k not in plot_data['ndcg']:
                    plot_data['ndcg'][k] = {}
                plot_data['ndcg'][k][model_name] = model_metrics[key]
    
    # Create bar plots for each metric and k value
    for metric_name, k_data in plot_data.items():
        for k, model_data in k_data.items():
            # Create dataframe for plotting
            df = pd.DataFrame({
                'Model': list(model_data.keys()),
                f'{metric_name.capitalize()}@{k}': list(model_data.values())
            })
            
            # Sort by metric value
            df = df.sort_values(f'{metric_name.capitalize()}@{k}', ascending=False)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Model', y=f'{metric_name.capitalize()}@{k}', data=df, palette='viridis')
            plt.title(f'{metric_name.capitalize()}@{k} Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save or display
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'{metric_name}_at_{k}.png'))
                plt.close()
            else:
                plt.show()
    
    # Create plot for MAP
    map_data = {}
    for model_name, model_metrics in metrics.items():
        if 'map' in model_metrics:
            map_data[model_name] = model_metrics['map']
    
    if map_data:
        # Create dataframe for plotting
        df = pd.DataFrame({
            'Model': list(map_data.keys()),
            'MAP': list(map_data.values())
        })
        
        # Sort by MAP value
        df = df.sort_values('MAP', ascending=False)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='MAP', data=df, palette='viridis')
        plt.title('Mean Average Precision (MAP) Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save or display
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'map.png'))
            plt.close()
        else:
            plt.show()

def compare_training_history(results_dir, model_names, save_dir=None):
    """
    Compare training history for different models
    
    Parameters:
    -----------
    results_dir : str
        Directory containing results
    model_names : list
        List of model names to compare
    save_dir : str
        Directory to save plots
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Load training history for each model
    histories = {}
    
    for model_name in model_names:
        # Try to load history from original training
        history_path = os.path.join(results_dir, f"{model_name}_history.pkl")
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                histories[model_name] = pickle.load(f)
        
        # Try to load history from tuned model
        tuned_history_path = os.path.join(results_dir, f"{model_name}_tuned_history.pkl")
        if os.path.exists(tuned_history_path):
            with open(tuned_history_path, 'rb') as f:
                histories[f"{model_name}_tuned"] = pickle.load(f)
    
    if not histories:
        print("No training history found")
        return
    
    # Plot training loss
    plt.figure(figsize=(12, 6))
    for model_name, history in histories.items():
        if 'train_loss' in history:
            plt.plot(history['train_loss'], label=f"{model_name} (Train)")
        if 'val_loss' in history:
            plt.plot(history['val_loss'], linestyle='--', label=f"{model_name} (Val)")
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save or display
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training_loss.png'))
        plt.close()
    else:
        plt.show()
    
    # Plot epoch times
    plt.figure(figsize=(12, 6))
    for model_name, history in histories.items():
        if 'epoch_times' in history:
            plt.plot(history['epoch_times'], label=model_name)
    
    plt.title('Epoch Training Time')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    # Save or display
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'epoch_times.png'))
        plt.close()
    else:
        plt.show()

def main(args):
    """
    Main function for comparing models
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics
    print("Loading metrics...")
    metrics = load_metrics(args.results_dir, args.models)
    
    if not metrics:
        print("No metrics found for the specified models")
        return
    
    # Plot metrics comparison
    print("Plotting metrics comparison...")
    plot_metrics_comparison(metrics, args.k_values, args.output_dir)
    
    # Compare training history
    print("Comparing training history...")
    compare_training_history(args.results_dir, args.models, args.output_dir)
    
    # Create summary table
    print("\nMetrics Summary:")
    
    # Prepare summary data
    summary_data = []
    
    for model_name, model_metrics in metrics.items():
        row = {'Model': model_name}
        
        # Add precision, recall, and ndcg at different k values
        for k in args.k_values:
            for metric in ['precision', 'recall', 'ndcg']:
                key = f'{metric}@{k}'
                if key in model_metrics:
                    row[key] = model_metrics[key]
        
        # Add MAP
        if 'map' in model_metrics:
            row['MAP'] = model_metrics['map']
        
        summary_data.append(row)
    
    # Create dataframe
    summary_df = pd.DataFrame(summary_data)
    
    # Print summary
    print(summary_df.to_string(index=False))
    
    # Save summary
    if args.output_dir:
        summary_path = os.path.join(args.output_dir, 'metrics_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare different recommendation models")
    
    # Path arguments
    parser.add_argument("--results_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"), help="Directory containing results")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "comparison"), help="Directory to save comparison results")
    
    # Model arguments
    parser.add_argument("--models", type=str, nargs="+", default=["ncf", "gat", "ensemble"], help="Models to compare")
    
    # Metric arguments
    parser.add_argument("--k_values", type=int, nargs="+", default=[5, 10, 20], help="k values for @k metrics")
    
    args = parser.parse_args()
    main(args)
