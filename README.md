# Ensemble Method to Improve Neural Collaborative Filtering

This project implements an ensemble approach to improve Neural Collaborative Filtering (NCF) for movie recommendations by combining NCF with Graph Attention Networks.

## Project Overview

Movie recommendation systems using Neural Collaborative Filtering (NCF) serve as benchmark models in the field for their high accuracy. This project enhances the traditional NCF approach by incorporating Graph Attention Networks to capture complex user-item relationships and improve recommendation accuracy.

Key features:
- Neural Collaborative Filtering (NCF) with multi-layer perceptron
- Graph Attention Network (GAT) for capturing complex user-item relationships
- Ensemble methods to combine both approaches for improved recommendation accuracy
- Comprehensive evaluation using Precision@K, Recall@K, MAP, and NDCG metrics

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

### Run the complete pipeline

```bash
python src/run_pipeline.py --all
```

This will:
1. Download the MovieLens dataset
2. Preprocess the data
3. Train the NCF, GAT, and ensemble models
4. Tune hyperparameters for the ensemble model
5. Compare model performance

### Run individual steps

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

### Demo recommendations

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

### Graph Attention Network (GAT)

The GAT model applies graph attention mechanisms to capture complex relationships in the user-item interaction graph, using multi-head self-attention to learn important connections.

### Ensemble Model

The ensemble model combines the NCF and GAT approaches using one of several methods:
- Weighted: Learn weights to combine NCF and GAT predictions
- Concatenation: Concatenate intermediate features from both models
- Gating: Use a learned gating mechanism to adaptively combine predictions

## Evaluation Metrics

The models are evaluated using standard recommendation system metrics:
- Precision@K: Proportion of recommended items that are relevant
- Recall@K: Proportion of relevant items that are recommended
- Mean Average Precision (MAP): Average precision across all users
- Normalized Discounted Cumulative Gain (NDCG): Measures ranking quality

## Team

- Xiaofei Wang
- Andy Wu

## Acknowledgments

This project was inspired by:
- He et al.'s Neural Collaborative Filtering approach (2017)
- Neural Graph Collaborative Filtering (NGCF)
- The MovieLens dataset by GroupLens Research
