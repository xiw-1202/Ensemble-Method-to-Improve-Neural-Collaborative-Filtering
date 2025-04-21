# Improved Recommendation System with Enhanced GAT Model

This documentation provides an overview of the improvements made to the Graph Attention Network (GAT) model and instructions on how to utilize these enhancements.

## Key Improvements

The GAT model has been significantly enhanced with the following improvements:

1. **Enhanced Architecture**
   - More GAT layers (configurable, default: 3)
   - Residual connections for better gradient flow
   - Layer normalization for training stability
   - MLP-based prediction instead of simple dot product

2. **Improved Graph Utilization**
   - Increased edge sampling rates (configurable, default: 80% vs. previous 50k limit)
   - Better subsampling strategy that scales with dataset size

3. **Better Initialization**
   - Xavier uniform initialization for more stable training
   - Improved parameter configurations

4. **Training Optimizations**
   - More robust optimizer settings
   - Better hyperparameter selection

## Usage Instructions

### Training the Optimized GAT Model

To train the enhanced GAT model with optimized hyperparameters, use the integrated option in the main run script:

```bash
# Train the optimized GAT model
./run.sh --train gat --optimized-gat
```

Or for more control over the parameters:

```bash
# Run directly with python
python src/run_pipeline.py --train_models --models gat --optimized_gat
```

### Manual Training with Custom Parameters

You can also train the model manually with custom parameters using the run_pipeline.py script:

```bash
python src/run_pipeline.py --train_models --models gat --optimized_gat \
  --gat_embedding_dim 128 \
  --gat_heads 4 \
  --gat_layers 3 \
  --gat_residual \
  --gat_subsampling_rate 0.9 \
  --gat_dropout 0.2 \
  --gat_learning_rate 0.0005 \
  --gat_weight_decay 1e-6 \
  --gat_epochs 50 \
  --gat_batch_size 256 \
  --gat_patience 10
```

Or directly with the training script if you need more control:

```bash
python src/train.py --model gat \
  --embedding_dim 128 \
  --gat_heads 4 \
  --gat_layers 3 \
  --gat_residual \
  --gat_subsampling_rate 0.9 \
  --dropout 0.2 \
  --learning_rate 0.0005 \
  --weight_decay 1e-6 \
  --epochs 50 \
  --batch_size 256 \
  --patience 10
```

### Hyperparameter Tuning

For tuning the hyperparameters of the improved GAT model:

```bash
python src/hyperparameter_tuning.py --model gat
```

The tuning process will explore various configurations of:
- Embedding dimensions
- Number of attention heads
- Number of GAT layers
- Residual connection usage
- Subsampling rates
- Dropout values

### Comparing Models

After training both the original and improved models, you can compare their performance:

```bash
python src/compare_models.py --models ncf gat ensemble
```

## Explanation of New Parameters

- `--gat_layers`: Number of GAT layers (default: 3)
- `--gat_residual`: Whether to use residual connections (flag)
- `--gat_subsampling_rate`: Rate of edge sampling for large graphs (default: 0.8)

## Expected Performance Improvements

The enhanced GAT model is expected to significantly outperform the original implementation with:

- Higher precision@k and recall@k scores
- Better NDCG@k performance
- More stable training process
- Improved generalization to new user-item pairs

## Troubleshooting

If you encounter memory issues with very large datasets:
- Reduce `embedding_dim` to 64
- Reduce `gat_heads` to 2
- Decrease `batch_size` to 128
- Adjust `gat_subsampling_rate` to 0.7

## Further Research Directions

Potential areas for further improvement:
- Exploring edge weighting schemes based on rating values
- Incorporating item metadata as node features
- Testing different graph attention mechanisms
- Implementing more sophisticated ensemble methods
