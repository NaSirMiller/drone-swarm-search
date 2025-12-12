#!/bin/bash

# Directory for logs
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# Benchmark script name
SCRIPT="evaluate.py"

# List of policies
#POLICIES=("multi" "exploratory" "greedy" "random" "sparse" "bb")
POLICIES=("bb" "sparse")

# Number of simulations
SIMS=1000

echo "Running benchmarks for all policies..."
echo "Logs will be saved in $LOG_DIR"
echo "--------------------------------------"

for POLICY in "${POLICIES[@]}"; do
    LOG_FILE="${LOG_DIR}/${POLICY}.log"

    echo "Running policy: $POLICY"
    echo "Logging to: $LOG_FILE"

    {
        echo "========================================="
        echo "Policy: $POLICY"
        echo "Timestamp: $(date)"
        echo "========================================="
    } >> "$LOG_FILE"

    # Run the Python benchmark and append output
    python3 "$SCRIPT" \
        --policy "$POLICY" \
        --num-simulations "$SIMS" \
        --debug \
        > "$LOG_FILE" 2>&1

    echo "Finished $POLICY"
    echo
done

echo "All policies completed."
