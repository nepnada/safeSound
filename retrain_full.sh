#!/bin/bash
# Full retrain pipeline: clean augmented → re-augment → preprocess → train all steps → evaluate
# Use this after adding new source data to data/raw/
set -e
cd "$(dirname "$0")"
source venv/bin/activate

echo "================================================"
echo " FULL RETRAIN PIPELINE"
echo "================================================"

# Step 0: Clean old augmented files (keep sources only)
echo -e "\n[0/6] Cleaning old augmented files..."
find data/raw -name "aug_*" -type f -delete
echo "  Cleaned."

# Step 1: Re-augment from new sources
echo -e "\n[1/6] Augmenting data..."
python src/data/augment.py

# Step 2: Re-preprocess (leak-free split)
echo -e "\n[2/6] Preprocessing (leak-free split)..."
python src/data/preprocess.py

# Step 3: Train teacher
echo -e "\n[3/6] Training teacher model..."
python src/train.py --step teacher

# Step 4: Knowledge distillation
echo -e "\n[4/6] Knowledge distillation → DS-CNN..."
python src/train.py --step student

# Step 5: QAT + TFLite
echo -e "\n[5/6] QAT + TFLite INT8 conversion..."
python src/train.py --step convert

# Step 6: Evaluate
echo -e "\n[6/6] Final evaluation..."
python src/train.py --step eval

# Export C header
python src/export_model_header.py

# Run simulation test
echo -e "\n[BONUS] Running simulation test..."
python simulation/embedded_simulator.py 2>&1 | tail -15

echo -e "\n================================================"
echo " RETRAIN COMPLETE"
echo " Model: models/student/dscnn_int8.tflite"
echo " Run: python test_model.py  (to verify)"
echo "================================================"
