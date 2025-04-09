#!/usr/bin/env python
"""
Graph Attention Network (GAT) model implementation for collaborative filtering
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

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
        
    def forward(self, edge_index, batch_user_idx, batch_item_idx):
        """
        Forward pass of the GAT model
        
        Parameters:
        -----------
        edge_index : torch.Tensor
            Graph edge indices (2 x num_edges)
        batch_user_idx : torch.Tensor
            Batch of user indices
        batch_item_idx : torch.Tensor
            Batch of item indices
            
        Returns:
        --------
        torch.Tensor
            Predicted ratings (0-1 scale)
        """
        # Create node features by combining user and item embeddings
        x = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        # Apply GAT layers
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        
        # Extract relevant embeddings for the batch
        user_emb = x[batch_user_idx]
        item_emb = x[batch_item_idx + self.num_users]  # Offset for item indices
        
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
        
        # Create edge index
        # Edge direction: user -> item
        edge_index_user_to_item = torch.tensor([
            users,
            items + self.num_users  # Offset for item indices
        ], dtype=torch.long)
        
        # Edge direction: item -> user (for message passing in both directions)
        edge_index_item_to_user = torch.tensor([
            items + self.num_users,  # Offset for item indices
            users
        ], dtype=torch.long)
        
        # Combine both directions
        edge_index = torch.cat([edge_index_user_to_item, edge_index_item_to_user], dim=1)
        
        return edge_index
