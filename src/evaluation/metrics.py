#!/usr/bin/env python
"""
Evaluation metrics for recommender systems
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

def rmse(predictions, targets):
    """
    Root Mean Squared Error
    
    Parameters:
    -----------
    predictions : array-like
        Predicted ratings
    targets : array-like
        Ground truth ratings
        
    Returns:
    --------
    float
        RMSE score
    """
    return np.sqrt(mean_squared_error(targets, predictions))

def precision_at_k(predicted_items, relevant_items, k=10):
    """
    Calculate precision@k for recommendations
    
    Parameters:
    -----------
    predicted_items : array-like
        List of predicted item IDs, ordered by relevance
    relevant_items : array-like
        List of relevant (true positive) item IDs
    k : int
        Number of top items to consider
        
    Returns:
    --------
    float
        Precision@k score
    """
    # Keep only top-k recommendations
    predicted_items = predicted_items[:k]
    
    # Count number of relevant items in top-k recommendations
    num_relevant = len(set(predicted_items) & set(relevant_items))
    
    # Calculate precision
    precision = num_relevant / k if k > 0 else 0.0
    
    return precision

def recall_at_k(predicted_items, relevant_items, k=10):
    """
    Calculate recall@k for recommendations
    
    Parameters:
    -----------
    predicted_items : array-like
        List of predicted item IDs, ordered by relevance
    relevant_items : array-like
        List of relevant (true positive) item IDs
    k : int
        Number of top items to consider
        
    Returns:
    --------
    float
        Recall@k score
    """
    # Keep only top-k recommendations
    predicted_items = predicted_items[:k]
    
    # Count number of relevant items in top-k recommendations
    num_relevant = len(set(predicted_items) & set(relevant_items))
    
    # Calculate recall
    recall = num_relevant / len(relevant_items) if len(relevant_items) > 0 else 0.0
    
    return recall

def average_precision(predicted_items, relevant_items):
    """
    Calculate Average Precision (AP) for recommendations
    
    Parameters:
    -----------
    predicted_items : array-like
        List of predicted item IDs, ordered by relevance
    relevant_items : array-like
        List of relevant (true positive) item IDs
        
    Returns:
    --------
    float
        Average Precision score
    """
    if not relevant_items:
        return 0.0
    
    relevant_items = set(relevant_items)
    precisions = []
    num_relevant = 0
    
    for i, item in enumerate(predicted_items):
        if item in relevant_items:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))
    
    return sum(precisions) / len(relevant_items) if precisions else 0.0

def mean_average_precision(predictions, relevant_items_list):
    """
    Calculate Mean Average Precision (MAP) for multiple users
    
    Parameters:
    -----------
    predictions : list
        List of predicted item lists for multiple users
    relevant_items_list : list
        List of relevant item lists for multiple users
        
    Returns:
    --------
    float
        MAP score
    """
    if not predictions or not relevant_items_list:
        return 0.0
    
    ap_scores = [average_precision(preds, rel) for preds, rel in zip(predictions, relevant_items_list)]
    return sum(ap_scores) / len(ap_scores)

def dcg_at_k(predicted_items, relevant_items_with_scores, k=10):
    """
    Calculate Discounted Cumulative Gain (DCG) at k
    
    Parameters:
    -----------
    predicted_items : array-like
        List of predicted item IDs, ordered by relevance
    relevant_items_with_scores : dict
        Dictionary mapping item IDs to relevance scores
    k : int
        Number of top items to consider
        
    Returns:
    --------
    float
        DCG@k score
    """
    # Keep only top-k recommendations
    predicted_items = predicted_items[:k]
    
    dcg = 0.0
    for i, item in enumerate(predicted_items):
        # Get relevance score (default to 0 if item not in relevant_items)
        rel = relevant_items_with_scores.get(item, 0)
        
        # Calculate DCG
        # Using log base 2 as is common in IR
        dcg += (2 ** rel - 1) / np.log2(i + 2)  # i+2 because i starts at 0
    
    return dcg

def idcg_at_k(relevant_items_with_scores, k=10):
    """
    Calculate Ideal Discounted Cumulative Gain (IDCG) at k
    
    Parameters:
    -----------
    relevant_items_with_scores : dict
        Dictionary mapping item IDs to relevance scores
    k : int
        Number of top items to consider
        
    Returns:
    --------
    float
        IDCG@k score
    """
    # Sort relevant items by score in descending order
    relevant_items = sorted(relevant_items_with_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Keep only top-k relevant items
    relevant_items = relevant_items[:k]
    
    idcg = 0.0
    for i, (_, rel) in enumerate(relevant_items):
        # Calculate IDCG
        idcg += (2 ** rel - 1) / np.log2(i + 2)  # i+2 because i starts at 0
    
    return idcg

def ndcg_at_k(predicted_items, relevant_items_with_scores, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at k
    
    Parameters:
    -----------
    predicted_items : array-like
        List of predicted item IDs, ordered by relevance
    relevant_items_with_scores : dict
        Dictionary mapping item IDs to relevance scores
    k : int
        Number of top items to consider
        
    Returns:
    --------
    float
        NDCG@k score
    """
    dcg = dcg_at_k(predicted_items, relevant_items_with_scores, k)
    idcg = idcg_at_k(relevant_items_with_scores, k)
    
    # Avoid division by zero
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def evaluate_recommendations(model, test_data, user_mapping, item_mapping, edge_index=None, rating_threshold=3.5, k_values=[5, 10, 20]):
    """
    Evaluate a recommendation model using various metrics
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained recommendation model
    test_data : pandas.DataFrame
        Test data containing user-item interactions
    user_mapping : dict
        Mapping from original user IDs to indices
    item_mapping : dict
        Mapping from original item IDs to indices
    edge_index : torch.Tensor, optional
        Graph edge indices for GAT (default: None)
    rating_threshold : float
        Threshold to consider an item as relevant
    k_values : list
        List of k values for @k metrics
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    model.eval()
    results = {}
    
    # Initialize metrics
    for k in k_values:
        results[f'precision@{k}'] = []
        results[f'recall@{k}'] = []
        results[f'ndcg@{k}'] = []
    
    # Group test data by user
    user_groups = test_data.groupby('userIdx')
    
    with torch.no_grad():
        for user_idx, group in user_groups:
            # Get actual items liked by user (rating above threshold)
            actual_items = group[group['rating'] >= rating_threshold]['movieIdx'].values
            
            if len(actual_items) == 0:
                continue
                
            # Get all items this user hasn't rated yet
            all_items = np.array(list(item_mapping.values()))
            rated_items = group['movieIdx'].values
            candidate_items = np.setdiff1d(all_items, rated_items)
            
            # If we have too many candidates, sample a subset
            if len(candidate_items) > 1000:
                np.random.seed(42)
                candidate_items = np.random.choice(candidate_items, 1000, replace=False)
            
            # Add back some items the user has rated for evaluation
            candidate_items = np.concatenate([candidate_items, rated_items])
            
            # Create tensors for prediction
            user_tensor = torch.tensor([user_idx] * len(candidate_items), dtype=torch.long)
            item_tensor = torch.tensor(candidate_items, dtype=torch.long)
            
            # Get predictions
            try:
                from models.gat import GATModel
                from models.ensemble import EnsembleModel
                
                if isinstance(model, GATModel) or isinstance(model, EnsembleModel):
                    # For GAT or Ensemble models that can use edge_index
                    if edge_index is not None:
                        predictions = model(user_tensor, item_tensor, edge_index)
                    else:
                        predictions = model(user_tensor, item_tensor)
                else:
                    # For NCF model that doesn't use edge_index
                    predictions = model(user_tensor, item_tensor)
            except Exception as e2:
                print(f"Error during evaluation: {e2}")
                return {metric: 0.0 for metric in ['precision@5', 'precision@10', 'precision@20',
                                                   'recall@5', 'recall@10', 'recall@20',
                                                   'ndcg@5', 'ndcg@10', 'ndcg@20', 'map']}
                
            # Create a dataframe for sorting
            pred_df = pd.DataFrame({
                'movieIdx': candidate_items,
                'prediction': predictions.cpu().numpy()
            })
            
            # Sort by prediction score
            pred_df = pred_df.sort_values('prediction', ascending=False)
            
            # Get top predicted items
            predicted_items = pred_df['movieIdx'].values
            
            # Create relevance scores dictionary for NDCG
            # Using actual ratings as relevance scores
            relevant_items_with_scores = {
                row['movieIdx']: (row['rating'] - rating_threshold) / (5 - rating_threshold)
                for _, row in group[group['rating'] >= rating_threshold].iterrows()
            }
            
            # Calculate metrics for each k
            for k in k_values:
                if len(predicted_items) >= k:
                    results[f'precision@{k}'].append(precision_at_k(predicted_items, actual_items, k))
                    results[f'recall@{k}'].append(recall_at_k(predicted_items, actual_items, k))
                    results[f'ndcg@{k}'].append(ndcg_at_k(predicted_items, relevant_items_with_scores, k))
    
    # Calculate average metrics
    for metric, values in results.items():
        results[metric] = sum(values) / len(values) if values else 0.0
    
    # Calculate MAP - Using 'movieIdx' column from pred_df which matches the column name in the dataframe
    try:
        results['map'] = mean_average_precision([pred_df['movieIdx'].values for user_idx, pred_df in user_groups], 
                                              [group[group['rating'] >= rating_threshold]['movieIdx'].values.tolist() 
                                               for user_idx, group in user_groups])
    except KeyError as e:
        print(f"Warning: KeyError in MAP calculation: {e}")
        print(f"Available columns in pred_df: {next(iter(user_groups))[1].columns.tolist() if user_groups else []}")
        results['map'] = 0.0
    
    return results
