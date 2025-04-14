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
    """
    def __init__(self, num_users, num_items, embedding_dim=64, heads=4, dropout=0.2):
        super(GATModel, self).__init__()
        
        # Embeddings (shared with NCF if using ensemble)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        # Store dimensions for computing node indices
        self.num_users = num_users
        self.num_items = num_items
        
        # Graph Attention layers
        # First layer: multiple attention heads
        self.gat1 = GATConv(embedding_dim, embedding_dim//heads, heads=heads, dropout=dropout)
        # Second layer: combine attention heads
        self.gat2 = GATConv(embedding_dim, embedding_dim, dropout=dropout)
        
        # Output layer
        self.output_layer = nn.Linear(embedding_dim, 1)
        
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
            # Fallback to simple embedding lookup when edge_index is not provided
            user_emb = self.user_embedding(user_indices)
            item_emb = self.item_embedding(item_indices)
        else:
            # Create node features by combining user and item embeddings
            x = torch.cat([
                self.user_embedding.weight,
                self.item_embedding.weight
            ], dim=0)
            
            # Use a subset of edge_index to avoid memory issues
            # and ensure consistent dimensions for batch processing
            if edge_index.size(1) > 100000:
                # If edge_index is too large, sample a subset
                perm = torch.randperm(edge_index.size(1))
                sample_size = min(100000, edge_index.size(1))
                edge_index_sample = edge_index[:, perm[:sample_size]]
            else:
                edge_index_sample = edge_index
                
            # Apply GAT layers
            x = F.elu(self.gat1(x, edge_index_sample))
            x = F.elu(self.gat2(x, edge_index_sample))
            
            # Extract relevant embeddings for the batch
            user_emb = x[user_indices]
            item_emb = x[item_indices + self.num_users]  # Offset for item indices
        
        # Compute dot product for prediction
        # Element-wise product followed by sum
        dot_product = (user_emb * item_emb).sum(dim=1)
        
        # Apply sigmoid for final prediction
        predictions = torch.sigmoid(dot_product)
        
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
        
        # Limit the number of edges if there are too many
        if len(users) > 50000:
            # Sample a subset of interactions 
            sample_size = 50000
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
