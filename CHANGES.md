# Project Improvements and Changes

This document summarizes the improvements made to address the GAT model's poor performance.

## Summary of Issues Found

The original GAT model was performing significantly worse than the NCF model and didn't contribute meaningfully to the ensemble:

```
Metrics Summary:
   Model  precision@5  recall@5   ndcg@5  precision@10  recall@10  ndcg@10  precision@20  recall@20  ndcg@20      MAP
     ncf     0.145258  0.051859 0.132270      0.127171   0.088506 0.137239      0.110399   0.149216 0.157874 0.680765
     gat     0.018303  0.005458 0.012715      0.022479   0.012927 0.017180      0.028519   0.033686 0.028062 0.680765
ensemble     0.145624  0.055335 0.137159      0.129867   0.094350 0.144088      0.111905   0.152814 0.164696 0.680765
```

## Key Issues Identified

1. **Graph Structure Limitations**: 
   - Severe edge subsampling (limit of 50k) 
   - Forward pass subsampling (limit of 100k)

2. **Architectural Weaknesses**:
   - Shallow network (only 2 GAT layers)
   - Simple dot product for prediction
   - No residual connections
   - Poor initialization

3. **Training Issues**:
   - No layer normalization for stability
   - Suboptimal hyperparameters

## Implemented Improvements

### 1. Enhanced GAT Model Architecture
- **Deeper Network**: Increased from 2 to configurable layers (default: 3)
- **Residual Connections**: Added to improve gradient flow in deeper networks
- **Layer Normalization**: Added for training stability
- **MLP Prediction Head**: Replaced simple dot product with MLP for complex patterns
- **Xavier Initialization**: Improved weight initialization for better convergence

### 2. Improved Graph Utilization
- **Expanded Edge Sampling**: Increased from fixed limit to percentage-based (80-90%)
- **Adaptive Subsampling**: Only subsample very large graphs (>500k edges)
- **Better Graph Construction**: Improved handling of the bipartite graph structure

### 3. Code Integration and Organization
- **Integrated Optimizations**: Added as parameters to existing code structure
- **Command-line Arguments**: Added new options for all optimization parameters
- **Shell Script Integration**: Added --optimized-gat flag to run.sh
- **Documentation Updates**: Updated README.md and added detailed README_IMPROVED.md

### 4. Hyperparameter Tuning Support
- **Extended Parameter Grid**: Added new parameters to hyperparameter tuning
- **Optimized Default Values**: Set empirically good defaults for new parameters

## File Changes

1. **src/models/gat.py**: 
   - Complete rewrite of GAT implementation
   - Added multiple layers, residual connections, layer normalization, MLP prediction

2. **src/train.py**:
   - Added command-line arguments for new GAT parameters
   - Updated model instantiation to use new parameters

3. **src/models/ensemble.py**:
   - Updated to use improved GAT model
   - Added parameters for GAT configuration

4. **src/run_pipeline.py**:
   - Added optimized GAT training option
   - Added command-line arguments for all new parameters

5. **src/hyperparameter_tuning.py**:
   - Extended parameter grid for GAT and ensemble models

6. **run.sh**:
   - Added --optimized-gat option
   - Updated help message and examples

7. **README.md** and **README_IMPROVED.md**:
   - Added documentation for optimized GAT model
   - Updated usage examples

## New Parameters

The following parameters were added to control the optimized GAT model:

- `--gat_layers`: Number of GAT layers (default: 3)
- `--gat_residual`: Use residual connections (flag)
- `--gat_subsampling_rate`: Rate of edge sampling (default: 0.9)
- `--gat_embedding_dim`: Embedding dimension (default: 128)
- `--gat_heads`: Number of attention heads (default: 4)
- `--gat_dropout`: Dropout rate (default: 0.2)
- `--gat_learning_rate`: Learning rate (default: 0.0005)
- `--gat_weight_decay`: Weight decay (default: 1e-6)
- `--gat_epochs`: Training epochs (default: 50)
- `--gat_patience`: Early stopping patience (default: 10)
- `--gat_batch_size`: Batch size (default: 256)

## Usage Examples

### Running with Shell Script
```bash
./run.sh --train gat --optimized-gat
```

### Running with Python
```bash
python src/run_pipeline.py --train_models --models gat --optimized_gat
```

### Custom Parameters
```bash
python src/run_pipeline.py --train_models --models gat --optimized_gat \
  --gat_embedding_dim 128 --gat_layers 4 --gat_residual
```

## Expected Results

The optimized GAT model should now show significantly improved performance metrics:
- Higher precision@k and recall@k scores
- Better NDCG values
- Improved contribution to the ensemble model
