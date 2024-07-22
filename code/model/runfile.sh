#!/bin/bash
# Usage: Simply execute ./runfile.sh from the terminal
# (after giving the file execute permissions with chmod +x runfile.sh)
# Description: Runs the model with the specified parameters
# Parameters:
# --ncust: Number of customers
# --kernel: Kernel type
# --hiericepts: Use hierarchical intercepts
# --samples: Number of samples
# --warmup: Number of warmup samples
# --name: Name of the run
# --out_dir: Name of the subdirectory of "outputs" you want to save to;
#            if left blank, it will save as "routines_model_{datetime}"
#            which is recommended to avoid overwriting previous results
# --process_results: include this flag if you want to automatically process the results
#
# The following code can also be copy/paste directly to the terminal

python -u -m main.run \
--kernel "expon" \
--hiericepts \
--samples 200 \
--warmup 800 \
--name "test" \
--process_results