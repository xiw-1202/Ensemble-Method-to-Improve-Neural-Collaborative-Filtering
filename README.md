# Ensemble Method to Improve Neural Collaborative Filtering

This project implements an ensemble approach to improve Neural Collaborative Filtering (NCF) for movie recommendations by combining NCF with Graph Attention Networks.

## Project Overview

Movie recommendation systems using Neural Collaborative Filtering (NCF) serve as benchmark models in the field for their high accuracy. This project enhances the traditional NCF approach by incorporating Graph Attention Networks to capture complex user-item relationships and improve recommendation accuracy.

Key features:
- Neural Collaborative Filtering (NCF) with multi-layer perceptron
- Graph Attention Network (GAT) for capturing complex user-item relationships
- Ensemble methods to combine both approaches for improved recommendation accuracy
- Comprehensive evaluation using Precision@K, Recall@K, MAP, and NDCG metrics
- Robust cross-platform compatibility with dynamic path resolution
- Error handling and fallback mechanisms for model training and inference

## Project Structure

```
project/
├── data/                # Raw and processed data
│   ├── raw/             # Original MovieLens dataset
│   └── processed/       # Preprocessed data files
├── notebooks/           # Jupyter notebooks for EDA and exploration
├── src/                 # Source code
│   ├── models/          # Model implementations (NCF, GAT, Ensemble)
│   │   ├── ncf.py       # Neural Collaborative Filtering model
│   │   ├── gat.py       # Graph Attention Network model
│   │   └── ensemble.py  # Ensemble model combining NCF and GAT
│   ├── utils/           # Utility functions for data processing
│   ├── evaluation/      # Evaluation metrics implementation
│   ├── train.py         # Training script
│   ├── hyperparameter_tuning.py  # Hyperparameter tuning script
│   ├── compare_models.py  # Model comparison script
│   └── demo_recommendations.py  # Demo script for recommendations
└── results/             # Results and visualizations
```

## Setup

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the pipeline to download the data, preprocess it, train the models, and evaluate:
   ```
   python src/run_pipeline.py --all
   ```

## Usage

### Using the Provided Shell Script

We provide a convenient shell script to run different components of the system:

```bash
# Run the complete pipeline
./run.sh --all

# Run specific components
./run.sh --download --preprocess                # Just prepare the data
./run.sh --train ncf gat ensemble               # Train specific models
./run.sh --tune ensemble                        # Tune a specific model
./run.sh --demo 123                             # Generate recommendations for user 123
```

### Using Python Directly

You can also use the Python scripts directly:

#### Run the complete pipeline

```bash
python src/run_pipeline.py --all
```

This will:
1. Download the MovieLens dataset
2. Preprocess the data
3. Train the NCF, GAT, and ensemble models
4. Tune hyperparameters for the ensemble model
5. Compare model performance

#### Run individual steps

Download and preprocess data:
```bash
python src/run_pipeline.py --download_data --preprocess_data
```

Train specific models:
```bash
python src/run_pipeline.py --train_models --models ncf gat ensemble
```

Tune hyperparameters for a specific model:
```bash
python src/run_pipeline.py --tune_hyperparams --tune_models ensemble
```

Compare model performance:
```bash
python src/run_pipeline.py --compare_models
```

#### Demo recommendations

Generate recommendations for a specific user:
```bash
python src/demo_recommendations.py --user_id 123 --plot
```

Generate recommendations for a random user:
```bash
python src/demo_recommendations.py --plot
```

## Model Details

### Neural Collaborative Filtering (NCF)

The NCF model utilizes embedding layers for users and items, followed by a multi-layer perceptron (MLP) to capture non-linear interactions between users and items.

```python
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, layers=[128, 64, 32], dropout=0.2):
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers for learning non-linear interactions
        self.fc_layers = nn.ModuleList()
        input_size = 2 * embedding_dim  # Concatenated embeddings
        
        for output_size in layers:
            self.fc_layers.append(nn.Linear(input_size, output_size))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout))
            input_size = output_size
```

### Graph Attention Network (GAT)

The GAT model applies graph attention mechanisms to capture complex relationships in the user-item interaction graph, using multi-head self-attention to learn important connections. The model handles large graphs by employing efficient sampling strategies.

```python
class GATModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, heads=4, dropout=0.2):
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Graph attention layers
        self.gat1 = GATConv(embedding_dim, embedding_dim//heads, heads=heads, dropout=dropout)
        self.gat2 = GATConv(embedding_dim, embedding_dim, dropout=dropout)
```

### Ensemble Model

The ensemble model combines the NCF and GAT approaches using one of several methods:

- **Weighted**: Learn weights to combine NCF and GAT predictions
  ```python
  # Weighted combination
  w_ncf = self.weight_ncf / (self.weight_ncf + self.weight_gat)
  w_gat = self.weight_gat / (self.weight_ncf + self.weight_gat)
  final_preds = w_ncf * ncf_preds + w_gat * gat_preds
  ```

- **Concatenation**: Concatenate intermediate features from both models
  ```python
  # Concatenate predictions and use a neural network
  concat_preds = torch.stack([ncf_preds, gat_preds], dim=1)
  final_preds = torch.sigmoid(self.ensemble_layer(concat_preds).squeeze())
  ```

- **Gating**: Use a learned gating mechanism to adaptively combine predictions
  ```python
  # Compute a gate value between 0 and 1
  gate = self.gate_network(torch.cat([user_emb_ncf, item_emb_ncf, user_emb_gat, item_emb_gat], dim=1))
  final_preds = gate * ncf_preds + (1 - gate) * gat_preds
  ```

## Evaluation Metrics

The models are evaluated using standard recommendation system metrics:

- **Precision@K**: Proportion of recommended items that are relevant
  ```python
  precision = num_relevant / k if k > 0 else 0.0
  ```

- **Recall@K**: Proportion of relevant items that are recommended
  ```python
  recall = num_relevant / len(relevant_items) if len(relevant_items) > 0 else 0.0
  ```

- **Mean Average Precision (MAP)**: Average precision across all users
  ```python
  ap_scores = [average_precision(preds, rel) for preds, rel in zip(predictions, relevant_items_list)]
  return sum(ap_scores) / len(ap_scores)
  ```

- **Normalized Discounted Cumulative Gain (NDCG)**: Measures ranking quality by accounting for position in result list
  ```python
  ndcg = dcg / idcg if idcg > 0 else 0.0
  ```

## Project Highlights

### Cross-Platform Compatibility

The system is designed to work across different platforms with robust path handling:

```python
# Convert relative path to absolute path if needed
if not os.path.isabs(data_dir):
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_dir = os.path.join(project_dir, data_dir)

# Check if the directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}")
```

### Error Handling

Robust error handling for model training and inference:

```python
try:
    # Try different models with and without edge information
    if isinstance(model, GATModel) or isinstance(model, EnsembleModel):
        if edge_index is not None:
            predictions = model(user_tensor, item_tensor, edge_index)
        else:
            predictions = model(user_tensor, item_tensor)
    else:
        predictions = model(user_tensor, item_tensor)
except Exception as e:
    print(f"Error during evaluation: {e}")
    # Graceful fallback with default metrics
```

### Efficient Graph Handling

The system can handle large graphs by using sampling techniques:

```python
# Use a subset of edge_index to avoid memory issues
if edge_index.size(1) > 100000:
    # If edge_index is too large, sample a subset
    perm = torch.randperm(edge_index.size(1))
    sample_size = min(100000, edge_index.size(1))
    edge_index_sample = edge_index[:, perm[:sample_size]]
```

## Team

- Xiaofei Wang
- Andy Wu

## Acknowledgments

This project was inspired by:
- He et al.'s Neural Collaborative Filtering approach (2017)
- Neural Graph Collaborative Filtering (NGCF) by Wang et al.
- The MovieLens dataset by GroupLens Research
- PyTorch and PyTorch Geometric libraries
