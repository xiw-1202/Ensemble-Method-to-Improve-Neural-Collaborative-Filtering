#!/usr/bin/env python
"""
Neural Collaborative Filtering (NCF) model implementation
"""
import torch
import torch.nn as nn

class NCF(nn.Module):
    """
    Neural Collaborative Filtering model
    
    Parameters:
    -----------
    num_users : int
        Number of users in the dataset
    num_items : int
        Number of items in the dataset
    embedding_dim : int
        Dimension of user and item embeddings
    layers : list
        List of layer sizes for the MLP
    dropout : float
        Dropout probability for regularization
    """
    def __init__(self, num_users, num_items, embedding_dim=64, layers=[128, 64, 32], dropout=0.2):
        super(NCF, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        # MLP layers
        self.fc_layers = nn.ModuleList()
        input_size = 2 * embedding_dim  # Concatenated user and item embeddings
        
        for i, output_size in enumerate(layers):
            self.fc_layers.append(nn.Linear(input_size, output_size))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout))
            input_size = output_size
        
        # Output layer
        self.output_layer = nn.Linear(layers[-1], 1)
        
    def forward(self, user_indices, item_indices):
        """
        Forward pass of the NCF model
        
        Parameters:
        -----------
        user_indices : torch.Tensor
            Tensor of user indices
        item_indices : torch.Tensor
            Tensor of item indices
            
        Returns:
        --------
        torch.Tensor
            Predicted ratings (0-1 scale)
        """
        # Get embeddings
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        
        # Concatenate user and item embeddings
        x = torch.cat([user_embedding, item_embedding], dim=1)
        
        # Feed through MLP layers
        for layer in self.fc_layers:
            x = layer(x)
        
        # Output layer
        output = torch.sigmoid(self.output_layer(x))
        
        return output.squeeze()
