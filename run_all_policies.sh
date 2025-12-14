#!/bin/bash
export PYTHONWARNINGS="ignore:pkg_resources is deprecated as an API"
# Directory for logs
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# Benchmark script name
SCRIPT="evaluate.py"

# List of policies
POLICIES=("bb" "sparse" "multi" "exploratory" "greedy" "random")
# POLICIES=("exploratory")
# POLICIES=("bb" "sparse")

# Number of simulations
SIMS=500
NUM_TARGETS=2
NUM_DRONES=1
GRID_SIZE=10
echo "--------------------------------------"
echo "There are $NUM_DRONES searching for $NUM_TARGETS on a $GRID_SIZE x $GRID_SIZE grid across $SIMS" 
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
    \python3 "$SCRIPT" \
        --policy "$POLICY" \
        --num-simulations "$SIMS" \
        --debug \
        --num-targets "$NUM_TARGETS" \
        --num-drones "$NUM_DRONES" \
        --grid-size "$GRID_SIZE" \
    > "$LOG_FILE" 2>&1


    echo "Finished $POLICY policy"
    echo
done

echo "All policies completed."
