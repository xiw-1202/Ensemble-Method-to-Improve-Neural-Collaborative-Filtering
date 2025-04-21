#!/usr/bin/env python
"""
Graph Attention Network (GAT) model implementation for collaborative filtering
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np

class GATModel(nn.Module):
    """
    Graph Attention Network model for collaborative filtering
    
    Parameters:
    -----------
    num_users : int
        Number of users in the dataset
    num_items : int
        Number of items in the dataset
    embedding_dim : int
        Dimension of user and item embeddings
    heads : int
        Number of attention heads
    dropout : float
        Dropout probability for regularization
    num_layers : int
        Number of GAT layers
    residual : bool
        Whether to use residual connections
    """
    def __init__(self, num_users, num_items, embedding_dim=64, heads=4, dropout=0.2, 
                 num_layers=3, residual=True, subsampling_rate=0.8):
        super(GATModel, self).__init__()
        
        # Embeddings (shared with NCF if using ensemble)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings with improved normalization
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Store dimensions for computing node indices
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.residual = residual
        self.subsampling_rate = subsampling_rate
        
        # Graph Attention layers
        self.gat_layers = nn.ModuleList()
        
        # First layer with multiple attention heads
        self.gat_layers.append(GATConv(embedding_dim, embedding_dim//heads, heads=heads, dropout=dropout))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(embedding_dim, embedding_dim//heads, heads=heads, dropout=dropout))
        
        # Final layer combining attention heads to original dimension
        self.gat_layers.append(GATConv(embedding_dim, embedding_dim, dropout=dropout))
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # MLP for prediction instead of simple dot product
        self.prediction_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )
        
    def forward(self, user_indices, item_indices, edge_index=None):
        """
        Forward pass of the GAT model
        
        Parameters:
        -----------
        user_indices : torch.Tensor
            Batch of user indices
        item_indices : torch.Tensor
            Batch of item indices
        edge_index : torch.Tensor, optional
            Graph edge indices (2 x num_edges)
            
        Returns:
        --------
        torch.Tensor
            Predicted ratings (0-1 scale)
        """
        # Check if edge_index is provided
        if edge_index is None:
            # Fallback to embedding lookup with MLP when edge_index is not provided
            user_emb = self.user_embedding(user_indices)
            item_emb = self.item_embedding(item_indices)
            # Use MLP for prediction
            combined = torch.cat([user_emb, item_emb], dim=1)
            predictions = torch.sigmoid(self.prediction_mlp(combined).squeeze(-1))
            return predictions
        
        # Create node features by combining user and item embeddings
        x = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        # Store original embeddings for residual connection
        original_x = x
        
        # Apply GAT layers with residual connections
        for i, gat_layer in enumerate(self.gat_layers):
            # Only subsample for very large edge indices
            if edge_index.size(1) > 500000:
                # Use a higher sampling rate to preserve more connections
                sample_size = int(edge_index.size(1) * self.subsampling_rate)
                perm = torch.randperm(edge_index.size(1))
                edge_index_sample = edge_index[:, perm[:sample_size]]
            else:
                edge_index_sample = edge_index
                
            # Apply GAT layer
            x_new = gat_layer(x, edge_index_sample)
            x_new = F.elu(x_new)
            
            # Apply residual connection if dimensions match and not first layer
            if self.residual and i > 0 and x.shape == x_new.shape:
                x = x_new + x
            else:
                x = x_new
            
            # Apply layer normalization for stability
            if hasattr(x, 'shape') and len(x.shape) > 1:
                x = self.layer_norm(x)
        
        # Extract relevant embeddings for the batch
        user_emb = x[user_indices]
        item_emb = x[item_indices + self.num_users]  # Offset for item indices
        
        # Combine embeddings using MLP for prediction
        combined = torch.cat([user_emb, item_emb], dim=1)
        predictions = torch.sigmoid(self.prediction_mlp(combined).squeeze(-1))
        
        return predictions

    def create_edge_index(self, user_item_interactions):
        """
        Create edge index for the user-item interaction graph
        
        Parameters:
        -----------
        user_item_interactions : pandas.DataFrame
            DataFrame containing user-item interactions
            
        Returns:
        --------
        torch.Tensor
            Edge index tensor (2 x num_edges)
        """
        # Extract user and item indices
        users = user_item_interactions['userIdx'].values
        items = user_item_interactions['movieIdx'].values
        
        # Use all interactions instead of limiting to a fixed number
        # If the dataset is extremely large, use a high sampling rate instead of fixed number
        if len(users) > 200000:
            # Sample a larger subset of interactions
            sample_size = min(int(len(users) * 0.8), 200000)
            indices = np.random.choice(len(users), sample_size, replace=False)
            users = users[indices]
            items = items[indices]
            
        # Create edge index for user -> item direction
        user_to_item_src = users.astype(np.int64)
        user_to_item_dst = (items + self.num_users).astype(np.int64)
        
        # Create edge index for item -> user direction
        item_to_user_src = user_to_item_dst.copy()
        item_to_user_dst = user_to_item_src.copy()
        
        # Combine both directions
        src_nodes = np.concatenate([user_to_item_src, item_to_user_src])
        dst_nodes = np.concatenate([user_to_item_dst, item_to_user_dst])
        
        # Create tensor
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        
        # Ensure no duplicate edges
        unique_indices = torch.unique(edge_index, dim=1)
        
        return unique_indices