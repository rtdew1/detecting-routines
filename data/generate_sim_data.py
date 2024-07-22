# USAGE: 
#   python generate_sim_data.py --n {num_custs} --w {num_weeks} --t {num_test_weeks}
#
# Note: all code assumes weekly style data, as in the paper, with the number
# of dayhours = 168 (i.e., a real Earth week)
#
# To modify the behavior of the simulated data, you can revise the code in
# {root_dir}/code/model/utils/data_io/sim_data.py. 

import os
import sys
import numpy as np
import pandas as pd
import argparse

from sim_config import SimData

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, help="Number of customers")
parser.add_argument("--w", type=int, help="Number of weeks")
parser.add_argument("--t", type=int, help="Number of test weeks", default=1)
args = parser.parse_args()

# Generate data
print("Generating simulated data...")
sim_data = SimData(args.n, args.w)

# Save data, converting from 0-indexed to 1-indexed
y_long = sim_data.y.reshape(-1)
weeks = np.tile(np.repeat(np.arange(args.w), 168), args.n) + 1
dayhours = np.tile(np.arange(168), args.w * args.n) + 1
ids = np.repeat(np.arange(args.n), args.w * 168) + 1

# This code assumes that all customers are acquired in the same week, so week = iweek.
df = pd.DataFrame({'id': ids, 'week': weeks, "iweek": weeks, 'dayhour': dayhours, 'y': y_long})
df = df[df.y > 0]

train_df = df[df.week <= args.w - args.t]
test_df = df[df.week > args.w - args.t]

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
print(f"Simulated data saved")


