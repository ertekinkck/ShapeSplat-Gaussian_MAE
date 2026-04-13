#!/bin/bash
# Run the ShapeSplat Reconstruction Demo

echo "🚀 Starting ShapeSplat Reconstruction Demo..."

# Ensure we are in the project root
cd /home/pc-3968/AES/sahneleme/ShapeSplat-Gaussian_MAE

# Activate Environment (if not already active in current shell context, but usually good to be explicit for scripts)
# Note: In an interactive shell, we might rely on the user having activated it.
# But let's try to run with `conda run` pattern which is more robust for scripts
echo "Environment: shape_splat"

conda run -n shape_splat python demo_use_case/run_demo.py

echo ""
echo "✅ Demo execution complete."
echo "📂 Check the output in: demo_use_case/output/":
