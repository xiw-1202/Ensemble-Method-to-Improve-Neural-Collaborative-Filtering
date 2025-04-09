#!/usr/bin/env python
"""
Run the entire recommendation system pipeline
"""
import os
import argparse
import subprocess
import time
from tqdm import tqdm

def run_command(command, description=None):
    """
    Run a shell command and display output
    
    Parameters:
    -----------
    command : str
        Command to run
    description : str
        Description of the command
        
    Returns:
    --------
    int
        Return code
    """
    if description:
        print(f"\n{'='*80}")
        print(f"  {description}")
        print(f"{'='*80}\n")
    
    print(f"Running: {command}")
    start_time = time.time()
    
    # Use subprocess.Popen to capture output in real-time
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for process to complete
    process.wait()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nCommand completed in {duration:.2f} seconds with return code {process.returncode}")
    
    return process.returncode

def main(args):
    """
    Main function to run the pipeline
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Current directory
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(src_dir)
    
    # Create output directories
    os.makedirs(os.path.join(project_dir, "data", "raw"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "data", "processed"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "results"), exist_ok=True)
    
    # Step 1: Download data
    if args.download_data or args.all:
        run_command(
            f"python {os.path.join(src_dir, 'utils', 'download_data.py')}",
            "Downloading MovieLens dataset"
        )
    
    # Step 2: Preprocess data
    if args.preprocess_data or args.all:
        run_command(
            f"python {os.path.join(src_dir, 'utils', 'data_preprocessing.py')}",
            "Preprocessing data"
        )
    
    # Use absolute paths for data and output directories
    data_dir = os.path.join(project_dir, "data", "processed")
    results_dir = os.path.join(project_dir, "results")
    
    # Step 3: Train models
    if args.train_models or args.all:
        models = args.models if args.models else ["ncf", "gat", "ensemble"]
        
        for model in models:
            cuda_arg = "--use_cuda" if args.use_cuda else ""
            run_command(
                f"python {os.path.join(src_dir, 'train.py')} --model {model} --epochs {args.epochs} --data_dir \"{data_dir}\" --output_dir \"{results_dir}\" {cuda_arg}",
                f"Training {model.upper()} model"
            )
    
    # Step 4: Hyperparameter tuning
    if args.tune_hyperparams or args.all:
        models = args.tune_models if args.tune_models else ["ensemble"]
        
        for model in models:
            reduced_grid_arg = "--reduced_grid" if args.reduced_grid else ""
            cuda_arg = "--use_cuda" if args.use_cuda else ""
            run_command(
                f"python {os.path.join(src_dir, 'hyperparameter_tuning.py')} --model {model} --epochs {args.tune_epochs} --final_epochs {args.epochs} --data_dir \"{data_dir}\" --output_dir \"{results_dir}\" {reduced_grid_arg} {cuda_arg}",
                f"Tuning hyperparameters for {model.upper()} model"
            )
    
    # Step 5: Compare models
    if args.compare_models or args.all:
        models = args.models if args.models else ["ncf", "gat", "ensemble"]
        models_arg = " ".join(models)
        
        run_command(
            f"python {os.path.join(src_dir, 'compare_models.py')} --models {models_arg} --results_dir \"{results_dir}\" --output_dir \"{os.path.join(results_dir, 'comparison')}\"",
            "Comparing models"
        )
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the recommendation system pipeline")
    
    # Pipeline steps
    parser.add_argument("--download_data", action="store_true", help="Download MovieLens dataset")
    parser.add_argument("--preprocess_data", action="store_true", help="Preprocess data")
    parser.add_argument("--train_models", action="store_true", help="Train models")
    parser.add_argument("--tune_hyperparams", action="store_true", help="Tune hyperparameters")
    parser.add_argument("--compare_models", action="store_true", help="Compare models")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    # Model arguments
    parser.add_argument("--models", type=str, nargs="+", help="Models to train and compare")
    parser.add_argument("--tune_models", type=str, nargs="+", help="Models to tune hyperparameters")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--tune_epochs", type=int, default=10, help="Number of epochs for each tuning trial")
    parser.add_argument("--reduced_grid", action="store_true", help="Use reduced parameter grid for faster tuning")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    
    args = parser.parse_args()
    
    # If no specific steps are specified, run all
    if not any([args.download_data, args.preprocess_data, args.train_models, 
                args.tune_hyperparams, args.compare_models, args.all]):
        print("No specific steps specified, running all steps")
        args.all = True
    
    main(args)
