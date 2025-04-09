#!/bin/bash
# Script to run the recommendation system pipeline

# Function to display help
show_help() {
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --all                 Run the complete pipeline"
    echo "  --download            Download the MovieLens dataset"
    echo "  --preprocess          Preprocess the data"
    echo "  --train [models]      Train specific models (ncf, gat, ensemble)"
    echo "  --tune [models]       Tune hyperparameters for specific models"
    echo "  --compare             Compare model performance"
    echo "  --demo [user_id]      Generate recommendations"
    echo "  --help                Display this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh --all"
    echo "  ./run.sh --download --preprocess"
    echo "  ./run.sh --train ncf gat ensemble"
    echo "  ./run.sh --tune ensemble"
    echo "  ./run.sh --demo 123"
}

# Check if no arguments are provided
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --all)
            python src/run_pipeline.py --all
            shift
            ;;
        --download)
            DOWNLOAD_FLAG="--download_data"
            shift
            ;;
        --preprocess)
            PREPROCESS_FLAG="--preprocess_data"
            shift
            ;;
        --train)
            TRAIN_FLAG="--train_models"
            shift
            # Collect model names
            MODELS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODELS="$MODELS $1"
                shift
            done
            if [ -n "$MODELS" ]; then
                TRAIN_MODELS="--models $MODELS"
            fi
            ;;
        --tune)
            TUNE_FLAG="--tune_hyperparams"
            shift
            # Collect model names
            TUNE_MODELS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                TUNE_MODELS="$TUNE_MODELS $1"
                shift
            done
            if [ -n "$TUNE_MODELS" ]; then
                TUNE_MODELS_ARG="--tune_models $TUNE_MODELS"
            fi
            ;;
        --compare)
            COMPARE_FLAG="--compare_models"
            shift
            ;;
        --demo)
            DEMO_FLAG=true
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                USER_ID="--user_id $1"
                shift
            fi
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run pipeline with collected flags
if [ -n "$DOWNLOAD_FLAG" ] || [ -n "$PREPROCESS_FLAG" ] || [ -n "$TRAIN_FLAG" ] || [ -n "$TUNE_FLAG" ] || [ -n "$COMPARE_FLAG" ]; then
    python src/run_pipeline.py $DOWNLOAD_FLAG $PREPROCESS_FLAG $TRAIN_FLAG $TRAIN_MODELS $TUNE_FLAG $TUNE_MODELS_ARG $COMPARE_FLAG
fi

# Run demo if requested
if [ "$DEMO_FLAG" = true ]; then
    python src/demo_recommendations.py $USER_ID --plot
fi

echo "Done!"
