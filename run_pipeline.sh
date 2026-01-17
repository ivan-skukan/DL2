#!/bin/bash

# 1. Run the Main Experiment
echo "Starting Main Experiment..."
uv run -m src.main --ks 1 2 4 8 16 --seeds 0 1 2 --output results.pt

# 2. Generate Plots (Results & ECE)
echo "Generating Performance Plots..."
uv run src/plot_results.py --results results.pt

# 3. Generate Human-Readable Summary
echo "Generating Final Results Summary..."
uv run python src/summary.py

echo "Pipeline Complete. Check results_plot.png and the summary table above."