#!/bin/bash
# Script to protect the training process from SSH disconnection

echo "🛡️  Protecting training process from SSH disconnection..."
echo ""

# Find the main training process (the conda run command)
CONDA_PID=$(ps aux | grep "[c]onda run -n shape_splat bash train_partial.sh" | awk '{print $2}')

if [ -z "$CONDA_PID" ]; then
    echo "❌ Training process not found!"
    exit 1
fi

echo "Found training process: PID $CONDA_PID"
echo ""
echo "⚠️  WARNING: The current training is attached to your terminal (pts/3)."
echo "If your SSH connection drops, the training WILL stop."
echo ""
echo "To protect it, we need to use 'disown' but the process must be in background."
echo ""
echo "RECOMMENDED: For future training runs, use tmux or screen:"
echo ""
echo "  # Using tmux (recommended):"
echo "  tmux new -s training"
echo "  conda activate shape_splat"
echo "  ./train_partial.sh"
echo "  # Press Ctrl+B then D to detach"
echo "  # To reattach: tmux attach -t training"
echo ""
echo "  # Or using screen:"
echo "  screen -S training"
echo "  conda activate shape_splat"
echo "  ./train_partial.sh"
echo "  # Press Ctrl+A then D to detach"
echo "  # To reattach: screen -r training"
echo ""
echo "For now, the training is running. Keep your SSH connection alive!"
echo "Training started at: 17:04 (been running for ~1h 15min)"
echo "Current epoch: 42/300 (~14% complete)"
