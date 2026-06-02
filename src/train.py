"""
Main training entry point.
Runs the full pipeline: teacher → distillation → QAT → TFLite

Usage:
    python src/train.py                  # full pipeline
    python src/train.py --step teacher   # teacher only
    python src/train.py --step student   # distillation only
    python src/train.py --step convert   # QAT + TFLite only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_teacher():
    print("\n" + "="*60)
    print("STEP 1: Train Teacher Model (CNN with augmentation)")
    print("="*60)
    from src.models.yamnet_finetune import train
    return train()


def run_distillation():
    print("\n" + "="*60)
    print("STEP 2: Knowledge Distillation → DS-CNN Student")
    print("="*60)
    from src.models.distillation import train_with_distillation
    return train_with_distillation()


def run_convert():
    print("\n" + "="*60)
    print("STEP 3: QAT + TFLite INT8 Conversion")
    print("="*60)
    from src.convert import apply_qat_and_convert
    return apply_qat_and_convert()


def run_evaluate():
    print("\n" + "="*60)
    print("STEP 4: Evaluation")
    print("="*60)
    from src.evaluate import evaluate_tflite
    evaluate_tflite("models/student/dscnn_int8.tflite")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["teacher", "student", "convert", "eval", "all"],
                        default="all")
    args = parser.parse_args()

    if args.step in ("teacher", "all"):
        run_teacher()

    if args.step in ("student", "all"):
        run_distillation()

    if args.step in ("convert", "all"):
        run_convert()

    if args.step in ("eval", "all"):
        run_evaluate()

    print("\n=== Pipeline complete ===")
