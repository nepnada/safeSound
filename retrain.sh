#!/bin/bash
# Retrain after adding new data (choking clips, etc.)
# Run this after dropping new files in data/raw/<class>/
# Usage: bash retrain.sh

set -e
cd "$(dirname "$0")"
source venv/bin/activate

echo "================================================"
echo " Retrain pipeline (new data detected)"
echo "================================================"

echo -e "\n[1/3] Augmenting new data..."
python src/data/augment.py

echo -e "\n[2/3] Reprocessing dataset..."
python src/data/preprocess.py

echo -e "\n[3/3] Full training pipeline..."
bash run_pipeline.sh

echo -e "\nDone. Check models/student/dscnn_int8.tflite"
