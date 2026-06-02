#!/bin/bash
# Full auto-pipeline: teacher → distillation → QAT → TFLite → evaluate
# Run from /Users/Apple/Desktop/iot
# Usage: bash run_pipeline.sh

set -e
cd "$(dirname "$0")"
source venv/bin/activate

echo "================================================"
echo " Sound Alert System — Full Training Pipeline"
echo "================================================"

# Step 1: Teacher (skip if already trained)
if [ ! -f "models/teacher/yamnet_finetuned.keras" ]; then
    echo -e "\n[1/4] Training teacher model..."
    python src/train.py --step teacher
else
    echo -e "\n[1/4] Teacher already trained — skipping"
fi

# Step 2: Knowledge Distillation
echo -e "\n[2/4] Knowledge distillation → DS-CNN student..."
python src/train.py --step student

# Step 3: QAT + TFLite conversion
echo -e "\n[3/4] QAT + TFLite INT8 conversion..."
python src/train.py --step convert

# Step 4: Evaluate
echo -e "\n[4/4] Evaluation..."
python src/train.py --step eval

# Step 5: Export C header for firmware
echo -e "\n[5/5] Exporting model_data.h for Arduino..."
python src/export_model_header.py

echo -e "\n================================================"
echo " Pipeline complete!"
echo " Model: models/student/dscnn_int8.tflite"
echo " Header: firmware/esp32/model_data.h"
echo "================================================"
