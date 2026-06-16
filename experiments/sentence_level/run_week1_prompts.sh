#!/bin/bash
# Week 1: Run all prompt variants on validation set with Qwen3.5-2B
# Usage: ./run_week1_prompts.sh [test_size]
# Example: ./run_week1_prompts.sh 50  # Quick test with 50 sentences
#          ./run_week1_prompts.sh      # Full validation (758 sentences)

set -e

# Activate virtual environment
source .venv/bin/activate

# Get test size from argument or use full validation
TEST_SIZE_ARG=""
if [ -n "$1" ]; then
    TEST_SIZE_ARG="--test_size $1"
    echo "Running experiments with $1 sentences per prompt variant"
else
    echo "Running experiments on full validation set (758 sentences)"
fi

# Array of prompts to test
PROMPTS=("default_zero_shot" "nih_k8" "plan_guided" "few_shot")

echo "================================"
echo "Week 1 Prompt Evaluation"
echo "Model: Qwen/Qwen3.5-2B"
echo "Split: val"
echo "================================"
echo ""

for PROMPT in "${PROMPTS[@]}"; do
    echo "----------------------------------------"
    echo "Running: $PROMPT"
    echo "----------------------------------------"
    
    RUN_NAME="qwen35-2b-${PROMPT}-val"
    
    if [ "$PROMPT" = "few_shot" ]; then
        python experiments/sentence_level/run_baseline.py \
            --split val \
            --prompt "$PROMPT" \
            --num_shots 3 \
            --load_in_4bit \
            --run_name "$RUN_NAME" \
            $TEST_SIZE_ARG
    else
        python experiments/sentence_level/run_baseline.py \
            --split val \
            --prompt "$PROMPT" \
            --load_in_4bit \
            --run_name "$RUN_NAME" \
            $TEST_SIZE_ARG
    fi
    
    echo ""
    echo "✓ Completed: $PROMPT"
    echo ""
done

echo "================================"
echo "All experiments completed!"
echo "================================"
echo ""
echo "Results saved to: experiments/sentence_level/results/"
echo ""
echo "Compare results with:"
echo "  python experiments/sentence_level/compare_results.py"
