#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pymc_env
python -u -m model.main.run \
--kernel "expon" \
--hiericepts \
--samples 400 \
--warmup 2000 \
--dir "~/routines/" \
--name "example run file"