
import numpy as np
import pandas as pd
import os

from utils.config import args

os.chdir(os.path.expanduser(args.MAIN_DIR))

print("loading data")

data = pd.read_csv(r"data/data_sample_covs_y.csv")
data = data[data.id <= args.N_CUST_SAMPLE]

# Set cut-offs for train/test:
start_week = 53
train_end = 90

# For each person, figure out their first week in the data
cust_first_week = data.groupby("id")["week"].min().values
cust_first_week = cust_first_week - start_week + 1

# Define train/test sets

train = data.loc[(data.week >= start_week) & (data.week <= train_end)]

id_counts = train.id.value_counts()
ids_to_keep = id_counts.keys()[id_counts >= 5]

new_data = data[np.isin(data.id, ids_to_keep)]
new_data["id"] = new_data.groupby("id").ngroup() + 1

train = new_data.loc[(new_data.week >= start_week) & (new_data.week <= train_end)]
test = new_data[new_data.week > train_end]

train.loc[:, "week"] = train.loc[:, "week"] - start_week + 1
test.loc[:, "week"] = test.loc[:, "week"] - start_week + 1

train.loc[:, "first_week"] = train.loc[:, "first_week"] - start_week + 1
test.loc[:, "first_week"] = test.loc[:, "first_week"] - start_week + 1

print("data loaded, beginning formatting")

train.dayhour = train.dayhour.astype(int)
test.dayhour = test.dayhour.astype(int)


n_cust = len(new_data.id.unique())
n_week_train = train.week.max()
n_week_total = n_week_train + len(test.week.unique())
n_dayhour = 168

# inputs are always just the range of possible values, used for creating GPs later
cust_inputs = np.arange(1, n_cust + 1)[:, None]
week_inputs = np.arange(1, n_week_total + 1)[:, None]
day_inputs = np.arange(1, 8)[:, None]
hour_inputs = np.arange(0, 24)[:, None]
dayhour_inputs = np.arange(1, n_dayhour + 1)[:, None]

# reformat y into (i,w,j) form (to be honest, this seems inefficient)
y = np.zeros((n_cust, n_week_train, n_dayhour))
y[train.id - 1, train.week - 1, train.dayhour - 1] = train.y

# index which observations fall at least 3 weeks after first week
include_obs = np.zeros_like(y, dtype=bool)
for i in range(n_cust):
    include_obs[i, max(0, cust_first_week[i] + 2) :] = True

# index where y is nonzero for sparse calculations later
nz_mask = (~(y == 0)) & include_obs


# --------------------------------------------------------------------------
# Nonsparse X data frame ---------------------------------------------------
# --------------------------------------------------------------------------


covs_data = pd.read_csv(r"data/data_sample_covs_X_weekly.csv")
covs_data = covs_data[np.isin(covs_data.id, ids_to_keep)]
covs_data["id"] = covs_data.groupby("id").ngroup() + 1

covs_data = covs_data[covs_data.id <= args.N_CUST_SAMPLE]
# covs_data = covs_data.loc[(covs_data.week >= start_week)]
# covs_data.loc[:, "week"] = covs_data.loc[:, "week"] - start_week + 1

# Select the subset of variables we're focusing on: prev
prev1_vars = [col for col in covs_data.columns.values if col[:4] == "prev"][:4]
n_covs = len(prev1_vars)

covs_data[prev1_vars] = (
    covs_data[prev1_vars] - covs_data[prev1_vars].mean()
) / covs_data[prev1_vars].std()

# Create a new data.frame that has all id/week/dayhours (dense + long):
nonsparse_df = pd.DataFrame()
nonsparse_df.index = pd.MultiIndex.from_product(
    [np.arange(1, n_cust + 1), np.arange(1, covs_data.week.max() + 1)],
    names=["id", "week"],
)
nonsparse_df = nonsparse_df.reset_index()
nonsparse_df = nonsparse_df.merge(
    covs_data.loc[:, ["id", "week"] + prev1_vars],
    how="left",
    on=["id", "week"],
)
nonsparse_df.loc[:, prev1_vars] = (
    nonsparse_df.loc[:, ["id"] + prev1_vars]
    .groupby("id")
    .transform(lambda x: x.fillna(method="ffill", inplace=False))
)
nonsparse_df = nonsparse_df[(nonsparse_df.week >= start_week)]
nonsparse_df["week"] = nonsparse_df["week"] - start_week + 1
nonsparse_df = nonsparse_df.dropna()

X_weekly = np.zeros(shape=(n_cust, n_week_total, n_covs))
X_weekly[nonsparse_df.id - 1, nonsparse_df.week - 1] = nonsparse_df.loc[:, prev1_vars]
X = X_weekly