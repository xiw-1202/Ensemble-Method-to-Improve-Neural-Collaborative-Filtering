#!/bin/bash
# Script to run the recommendation system pipeline

# Function to display help
show_help() {
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --all                 Run the complete pipeline (always uses optimized GAT)"
    echo "  --download            Download the MovieLens dataset"
    echo "  --preprocess          Preprocess the data"
    echo "  --train [models]      Train specific models (ncf, gat, ensemble)"
    echo "                        Note: GAT always uses optimized configuration"
    echo "  --tune [models]       Tune hyperparameters for specific models"
    echo "  --compare             Compare model performance"
    echo "  --demo [user_id]      Generate recommendations"
    echo "  --test [models]       Run quick tests with a small synthetic dataset"
    echo "  --help                Display this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh --all"
    echo "  ./run.sh --download --preprocess"
    echo "  ./run.sh --train ncf gat ensemble"
    echo "  ./run.sh --train gat"
    echo "  ./run.sh --tune ensemble"
    echo "  ./run.sh --demo 123"
    echo "  ./run.sh --test ncf gat ensemble"
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
            # Run the complete pipeline with optimized GAT
            python3 src/run_pipeline.py --all --optimized_gat
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
            GAT_INCLUDED=false
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODELS="$MODELS $1"
                if [[ "$1" == "gat" ]]; then
                    GAT_INCLUDED=true
                fi
                shift
            done
            if [ -n "$MODELS" ]; then
                TRAIN_MODELS="--models $MODELS"
            fi
            # Always use optimized GAT if GAT is included
            if [ "$GAT_INCLUDED" = true ]; then
                OPTIMIZED_GAT_FLAG="--optimized_gat"
            fi
            ;;
        --optimized-gat)
            # This option is kept for backward compatibility
            # but doesn't do anything special now as GAT is always optimized
            shift
            ;;
        --tune)
            TUNE_FLAG="--tune_hyperparams"
            shift
            # Collect model names
            TUNE_MODELS=""
            GAT_TUNE_INCLUDED=false
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                TUNE_MODELS="$TUNE_MODELS $1"
                if [[ "$1" == "gat" ]]; then
                    GAT_TUNE_INCLUDED=true
                fi
                shift
            done
            if [ -n "$TUNE_MODELS" ]; then
                TUNE_MODELS_ARG="--tune_models $TUNE_MODELS"
            fi
            # Always include optimized parameters for GAT tuning
            if [ "$GAT_TUNE_INCLUDED" = true ]; then
                OPTIMIZED_GAT_FLAG="--optimized_gat"
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
        --test)
            TEST_FLAG=true
            shift
            # Collect model names
            TEST_MODELS=""
            GAT_TEST_INCLUDED=false
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                TEST_MODELS="$TEST_MODELS $1"
                if [[ "$1" == "gat" ]]; then
                    GAT_TEST_INCLUDED=true
                fi
                shift
            done
            if [ -n "$TEST_MODELS" ]; then
                TEST_MODELS_ARG="--models $TEST_MODELS"
            fi
            # Always use optimized GAT parameters for testing
            if [ "$GAT_TEST_INCLUDED" = true ]; then
                TEST_MODELS_ARG="$TEST_MODELS_ARG --gat_layers 3 --gat_residual --gat_subsampling_rate 0.9"
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
    python3 src/run_pipeline.py $DOWNLOAD_FLAG $PREPROCESS_FLAG $TRAIN_FLAG $TRAIN_MODELS $OPTIMIZED_GAT_FLAG $TUNE_FLAG $TUNE_MODELS_ARG $COMPARE_FLAG
fi

# Run demo if requested
if [ "$DEMO_FLAG" = true ]; then
    python3 src/demo_recommendations.py $USER_ID --plot
fi

# Run test if requested
if [ "$TEST_FLAG" = true ]; then
    python3 test_models.py $TEST_MODELS_ARG
fi

echo "Done!"
