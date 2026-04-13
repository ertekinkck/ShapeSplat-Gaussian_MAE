#!/bin/bash
# Script to monitor ShapeSplat training progress

LOG_FILE="experiments/pretrain_enc_full_group_xyz_1k/pretrain/pretrain_shapesplat_partial/20260119_170459.log"

echo "==================================="
echo "ShapeSplat Training Monitor"
echo "==================================="
echo ""

# Check if training is running
if ps aux | grep -q "[p]ython.*main.py.*pretrain_shapesplat_partial"; then
    echo "✅ Training is RUNNING"
    echo ""
    
    # Get number of processes
    PROC_COUNT=$(ps aux | grep "[p]ython.*main.py.*pretrain_shapesplat_partial" | wc -l)
    echo "📊 Active processes: $PROC_COUNT"
    echo ""
    
    # Show latest training metrics
    echo "📈 Latest Training Metrics:"
    echo "-----------------------------------"
    tail -n 20 "$LOG_FILE" | grep "Epoch"
    echo ""
    
    # Show loss progression
    echo "📉 Loss Progression:"
    echo "-----------------------------------"
    grep "Losses =" "$LOG_FILE" | tail -n 10
    echo ""
    
    # Show GPU usage if nvidia-smi available
    if command -v nvidia-smi &> /dev/null; then
        echo "🖥️  GPU Usage:"
        echo "-----------------------------------"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    fi
else
    echo "❌ Training is NOT running"
fi

echo ""
echo "==================================="
echo "To watch training in real-time:"
echo "tail -f $LOG_FILE"
echo "==================================="
