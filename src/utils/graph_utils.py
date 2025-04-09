#!/usr/bin/env python
"""
Utilities for creating and manipulating graphs for the GAT model
"""
import torch
import numpy as np
import pandas as pd
from scipy import sparse
from torch_geometric.data import Data

def create_user_item_matrix(ratings_df, num_users, num_items):
    """
    Create a sparse user-item interaction matrix
    
    Parameters:
    -----------
    ratings_df : pandas.DataFrame
        DataFrame containing user-item interactions
    num_users : int
        Number of users
    num_items : int
        Number of items
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        Sparse user-item interaction matrix
    """
    # Extract user and item indices
    user_indices = ratings_df['userIdx'].values
    item_indices = ratings_df['movieIdx'].values
    ratings = ratings_df['rating_norm'].values
    
    # Create sparse matrix
    matrix = sparse.csr_matrix((ratings, (user_indices, item_indices)), 
                              shape=(num_users, num_items))
    
    return matrix

def create_edge_index(ratings_df, num_users, num_items):
    """
    Create edge index tensor for PyTorch Geometric
    
    Parameters:
    -----------
    ratings_df : pandas.DataFrame
        DataFrame containing user-item interactions
    num_users : int
        Number of users
    num_items : int
        Number of items
        
    Returns:
    --------
    torch.Tensor
        Edge index tensor (2 x num_edges)
    """
    # Extract user and item indices
    user_indices = ratings_df['userIdx'].values
    item_indices = ratings_df['movieIdx'].values
    
    # Create edge indices (user -> item)
    edge_index_user_to_item = torch.tensor([
        user_indices,
        item_indices + num_users  # Offset for item indices
    ], dtype=torch.long)
    
    # Create edge indices (item -> user) for bidirectional graph
    edge_index_item_to_user = torch.tensor([
        item_indices + num_users,  # Offset for item indices
        user_indices
    ], dtype=torch.long)
    
    # Combine both directions
    edge_index = torch.cat([edge_index_user_to_item, edge_index_item_to_user], dim=1)
    
    return edge_index

def create_graph_data(ratings_df, num_users, num_items):
    """
    Create PyTorch Geometric Data object for the user-item graph
    
    Parameters:
    -----------
    ratings_df : pandas.DataFrame
        DataFrame containing user-item interactions
    num_users : int
        Number of users
    num_items : int
        Number of items
        
    Returns:
    --------
    torch_geometric.data.Data
        Graph data object
    """
    # Create edge index
    edge_index = create_edge_index(ratings_df, num_users, num_items)
    
    # Extract ratings as edge attributes
    edge_attr_user_to_item = torch.tensor(ratings_df['rating_norm'].values, dtype=torch.float)
    # Use the same ratings for the reverse edges
    edge_attr = torch.cat([edge_attr_user_to_item, edge_attr_user_to_item])
    
    # Create data object
    data = Data(edge_index=edge_index, edge_attr=edge_attr, 
                num_nodes=num_users + num_items)
    
    return data

def compute_similarity_matrix(embeddings, k=10):
    """
    Compute cosine similarity matrix for embeddings and keep only top-k similar items
    
    Parameters:
    -----------
    embeddings : torch.Tensor
        Item embeddings
    k : int
        Number of similar items to keep for each item
        
    Returns:
    --------
    torch.Tensor
        Sparse similarity matrix
    """
    # Normalize embeddings
    norm = torch.norm(embeddings, dim=1, keepdim=True)
    normalized_embeddings = embeddings / norm.clamp(min=1e-10)
    
    # Compute dot product (cosine similarity for normalized vectors)
    similarity = torch.mm(normalized_embeddings, normalized_embeddings.t())
    
    # Get top-k similar items for each item
    _, indices = torch.topk(similarity, k=k+1, dim=1)  # k+1 because the most similar item is itself
    
    # Remove self-similarity
    indices = indices[:, 1:]
    
    # Create sparse adjacency matrix
    rows = torch.arange(embeddings.size(0)).unsqueeze(1).expand_as(indices)
    sparse_adj = torch.zeros_like(similarity)
    sparse_adj[rows.flatten(), indices.flatten()] = 1
    
    return sparse_adj
