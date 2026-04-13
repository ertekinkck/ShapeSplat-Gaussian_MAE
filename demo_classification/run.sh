#!/bin/bash
# Run the ShapeSplat Classification Demo

echo "🚀 Starting Classification Demo..."

# Ensure we are in the project root
cd /home/pc-3968/AES/sahneleme/ShapeSplat-Gaussian_MAE

# Activate Environment
echo "Environment: shape_splat"

conda run -n shape_splat python demo_classification/run_classifier.py
