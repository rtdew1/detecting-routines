
import numpy as np
import pandas as pd
import os
from utils.config import args

print("Loading training data")

train = pd.read_csv(r"../../data/train.csv")
test = pd.read_csv(r"../../data/test.csv")

train = train[train.id <= args.N_CUST_SAMPLE]
test = test[test.id <= args.N_CUST_SAMPLE]

train["cust_first_week"] = train.week - train.iweek

print("Data loaded, beginning formatting")

n_cust = len(train.id.unique())
n_week_train = len(train.week.unique())
n_week_total = n_week_train + len(test.week.unique())
n_dayhour = 168

# reformat y into (i,w,j) form (to be honest, this seems inefficient)
y = np.zeros((n_cust, n_week_train, n_dayhour))
y[train.id - 1, train.week - 1, train.dayhour - 1] = train.y

# index which observations fall at least 3 weeks after first week
cust_first_week = train.groupby("id")["cust_first_week"].min().values
include_obs = np.zeros_like(y, dtype=bool)
for i in range(n_cust):
    include_obs[i, max(0, cust_first_week[i] + 2) :] = True

# index where y is nonzero for sparse calculations later
nz_mask = (~(y == 0)) & include_obs