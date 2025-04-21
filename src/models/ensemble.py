#!/usr/bin/env python
"""
Ensemble model combining Neural Collaborative Filtering and Graph Attention Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ncf import NCF
from .gat import GATModel

class EnsembleModel(nn.Module):
    """
    Ensemble model combining NCF and GAT for collaborative filtering
    
    Parameters:
    -----------
    num_users : int
        Number of users in the dataset
    num_items : int
        Number of items in the dataset
    embedding_dim : int
        Dimension of user and item embeddings
    mlp_layers : list
        List of layer sizes for the MLP in NCF
    gat_heads : int
        Number of attention heads for GAT
    dropout : float
        Dropout probability for regularization
    share_embeddings : bool
        Whether to share embeddings between NCF and GAT
    ensemble_method : str
        Method for combining NCF and GAT predictions ('weighted', 'concat', or 'gate')
    """
    def __init__(self, num_users, num_items, embedding_dim=64, mlp_layers=[128, 64, 32], 
                 gat_heads=4, dropout=0.2, share_embeddings=True, ensemble_method='weighted',
                 gat_layers=3, gat_residual=True, gat_subsampling_rate=0.8):
        super(EnsembleModel, self).__init__()
        
        # Create NCF model
        self.ncf = NCF(num_users, num_items, embedding_dim, mlp_layers, dropout)
        
        # Create GAT model with improved architecture
        self.gat = GATModel(
            num_users, 
            num_items, 
            embedding_dim, 
            gat_heads, 
            dropout,
            num_layers=gat_layers,
            residual=gat_residual,
            subsampling_rate=gat_subsampling_rate
        )
        
        # Share embeddings if specified
        if share_embeddings:
            self.gat.user_embedding = self.ncf.user_embedding
            self.gat.item_embedding = self.ncf.item_embedding
        
        self.ensemble_method = ensemble_method
        
        # Different ensemble methods
        if ensemble_method == 'weighted':
            # Learnable weights for combining predictions
            self.weight_ncf = nn.Parameter(torch.tensor(0.5))
            self.weight_gat = nn.Parameter(torch.tensor(0.5))
        elif ensemble_method == 'concat':
            # Concatenate outputs and use a linear layer
            if share_embeddings:
                self.ensemble_layer = nn.Sequential(
                    nn.Linear(2, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, 1)
                )
            else:
                self.ensemble_layer = nn.Sequential(
                    nn.Linear(2, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, 1)
                )
        elif ensemble_method == 'gate':
            # Gating mechanism to adaptively combine predictions
            self.gate_network = nn.Sequential(
                nn.Linear(embedding_dim * 4, 32),  # Concatenated user and item embeddings from both models
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
    def forward(self, user_indices, item_indices, edge_index=None):
        """
        Forward pass of the ensemble model
        
        Parameters:
        -----------
        user_indices : torch.Tensor
            Tensor of user indices
        item_indices : torch.Tensor
            Tensor of item indices
        edge_index : torch.Tensor, optional
            Graph edge indices
            
        Returns:
        --------
        torch.Tensor
            Predicted ratings (0-1 scale)
        """
        # Get predictions from each model
        try:
            # Try to get NCF predictions
            ncf_preds = self.ncf(user_indices, item_indices)
        except Exception as e:
            print(f"Error in NCF model: {e}")
            ncf_preds = torch.zeros_like(user_indices, dtype=torch.float)
            
        try:
            # Try to get GAT predictions
            if edge_index is not None:
                gat_preds = self.gat(user_indices, item_indices, edge_index)
            else:
                gat_preds = self.gat(user_indices, item_indices)
        except Exception as e:
            print(f"Error in GAT model: {e}")
            gat_preds = torch.zeros_like(user_indices, dtype=torch.float)
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'weighted':
            # Normalize weights
            w_sum = self.weight_ncf + self.weight_gat
            w_ncf = self.weight_ncf / w_sum
            w_gat = self.weight_gat / w_sum
            
            # Weighted average
            final_preds = w_ncf * ncf_preds + w_gat * gat_preds
        
        elif self.ensemble_method == 'concat':
            # Concatenate and use ensemble layer
            concat_preds = torch.stack([ncf_preds, gat_preds], dim=1)
            final_preds = torch.sigmoid(self.ensemble_layer(concat_preds).squeeze())
        
        elif self.ensemble_method == 'gate':
            # Get user and item embeddings from both models
            user_emb_ncf = self.ncf.user_embedding(user_indices)
            item_emb_ncf = self.ncf.item_embedding(item_indices)
            
            # If not sharing embeddings, get separate embeddings from GAT
            if self.gat.user_embedding is not self.ncf.user_embedding:
                user_emb_gat = self.gat.user_embedding(user_indices)
                item_emb_gat = self.gat.item_embedding(item_indices)
            else:
                user_emb_gat = user_emb_ncf
                item_emb_gat = item_emb_ncf
            
            # Concatenate all embeddings
            gate_input = torch.cat([user_emb_ncf, item_emb_ncf, user_emb_gat, item_emb_gat], dim=1)
            
            # Compute gating weight
            gate = self.gate_network(gate_input)
            
            # Apply gate
            final_preds = gate * ncf_preds + (1 - gate) * gat_preds
        
        return final_preds
