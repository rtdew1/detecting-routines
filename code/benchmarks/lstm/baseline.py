"""Execution file to run baseline LSTM model and get results (MAP/CP)"""

# Importing necessary packages and files
import pandas as pd
import plotly.express as px
import numpy as np
from lstm import lstm
from tqdm import tqdm

train_set = pd.read_csv("via_train_test/train.csv")
train_set = train_set.drop(['Unnamed: 0'], axis=1)

test_set = pd.read_csv("via_train_test/test.csv")
test_set = test_set.drop(['Unnamed: 0'], axis=1)

# Converting all the data from tabular format to time series
# Data format for each user: Time series list, with each index representing an hour out of the 48 week period
# 0 if no rides requested in the hour/ 1 if rides requested in the hour

# 2D array to store all of the individual (2000) time series
ts_data = []

for id in sorted(list(set(train_set["id"]))):
    # create full time series, first all filled with observations of 0
    full_timesteps = 48 * 7 * 24  # 48 weeks * 7 days a week * 24 hours a day
    t_series = [0 for x in range(full_timesteps)]

    # selecting from respect DFs to only get data from user
    id_train_set = train_set.loc[train_set["id"] == id]
    id_test_set = test_set.loc[test_set["id"] == id]

    week_dayhour = list(zip(id_train_set["week"], id_train_set["dayhour"])) + \
                   list(zip(id_test_set["week"], id_test_set["dayhour"]))

    # converting each pair (week, dayhour) into raw hour index from 0-8063 and
    # adding that observation to the time series
    for pair in week_dayhour:
        hour_idx = ((pair[0] - 1) * 168) + pair[1]
        # data index starts at 1, convert to index starting at 0
        hour_idx -= 1

        # adding a ride request observation at the time series
        t_series[hour_idx] += 1

    ts_data.append(t_series)


# Creating walk-forward validation train/test splits
def create_train_walk(time_series, train_len, in_sample, out_sample, walk_timestep):
    '''
    Takes in the time series data (full 48) for one full user and
    returns "walks" that represent train/validation splits
    :param int list time_series: 1D array of full 48 week time series data for one user
    :param train_len: Length of training period (in hours);
    Function assumes train period is at beginning of time series
    :param in_sample: Length of sub-time series input into LSTM for one step prediction
    :param out_sample: Length of sub-time series output from LSTM for one step prediction
    :param walk_timestep: Interval of walk
    :return: X, Training set (Each item has dims of in_sample)
    :return: y, Validation set (Each item has dims of out_sample)
    '''
    X, y = list(), list()
    in_start = 0
    for obs in range(int(train_len / walk_timestep)):
        in_end = in_start + in_sample
        out_end = in_end + out_sample

        # checking to make sure there's sufficient data
        if out_end <= train_len:
            x_input = np.array(time_series[in_start: in_end])
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(time_series[in_end: out_end])

        # moving along the walk according to the walk_timestep
        in_start += walk_timestep

    return np.array(X), np.array(y)


# Initializing model and creating recursive call to get all 10 weeks of predictions
# Model uses 0.20 as the threshold for hour that has ride requested
def average(lst):
    '''
    Calculates the average value of a numerical list, omitting NaN values
    :param lst: List to calculate average from
    :return float: Average of list, omitting NaN
    '''
    real_vals_lst = [x for x in lst if not np.isnan(x)]
    return sum(real_vals_lst) / len(real_vals_lst)


def multi_step_lstm(user_idx,
                    train_len,
                    in_sample,
                    out_sample,
                    walk_timestep,
                    epochs,
                    batch_size):
    '''
    Function to create and forecast LSTM model for individual users
    :param user_idx: List index from ts_data that contains time series data for one user
    :param train_len: Length of training period (in hours);
    Function assumes train period is at beginning of time series
    :param in_sample: Length of sub-time series input into LSTM for one step prediction
    :param out_sample: Length of sub-time series output from LSTM for one step prediction
    :param walk_timestep: Interval of walk
    :param epochs: Number of times to train over dataset
    :param batch_size: Number of predictions to feed back into model at once for learning
    :return in_metrics: List with metrics for training period prediction metrics;
    first index contains AP;
    second index contains CP
    :return out_metrics: List with metrics for testing period prediction metrics;
    first index contains AP;
    second index contains CP
    '''
    user_X, user_y = create_train_walk(ts_data[user_idx], train_len, in_sample, out_sample, walk_timestep)

    user_model = lstm("individual", in_sample, out_sample)
    user_model.train(user_X, user_y, epochs, batch_size)

    # Storing AP/Prec respectively for within training data and forecasting out
    in_metrics = []
    out_metrics = []

    # multi-step forecast within training data
    in_forecast_steps = int((train_len - in_sample) / out_sample)
    in_probs, in_preds = user_model.multi_step_forecast(ts_data[user_idx], train_len, 0, in_forecast_steps)

    # multi-step forecasting out of training data
    out_forecast_steps = int((len(ts_data[user_idx]) - train_len) / out_sample)
    out_probs, out_preds = user_model.multi_step_forecast(ts_data[user_idx], train_len, train_len - in_sample,
                                                          out_forecast_steps)

    # adding AP/prec metrics for in sample
    in_metrics.append(user_model.average_precision(ts_data[user_idx][in_sample: train_len], in_probs))
    in_metrics.append(user_model.conditional_precision(ts_data[user_idx][in_sample: train_len], in_preds))

    # adding AP/prec metrics for out sample
    out_metrics.append(user_model.average_precision(ts_data[user_idx][train_len:], out_probs))
    out_metrics.append(user_model.conditional_precision(ts_data[user_idx][train_len:], out_preds))

    return out_preds, in_metrics, out_metrics


# Use the above function to create individual LSTM models for all users and get metrics for training/testing period
# Putting results in a dataframe
def export_results(results_name, out_pred_list, in_ap_list, in_prec_list, out_ap_list, out_prec_list):
    pred_results = pd.DataFrame(out_pred_list)

    pred_results.to_csv(results_name + "_out_predictions.csv", index=False, header=False)

    user_results = pd.DataFrame({
        "User Index": list(range(len(ts_data))),
        "Training Period AP": in_ap_list,
        "Training Period CP": in_prec_list,
        "Testing Period AP": out_ap_list,
        "Testing Period CP": out_prec_list
    })

    user_results.to_csv(results_name + "_user.csv", index=False)

    avg_results = pd.DataFrame({
        "Training Period AP": [average(in_ap_list)],
        "Training Period CP": [average(in_prec_list)],
        "Testing Period AP": [average(out_ap_list)],
        "Testing Period CP": [average(out_prec_list)]
    })

    avg_results.to_csv(results_name + "_avg.csv", index=False)


# Setting parameters for training and forecasting
train_len = 168 * 38
in_sample = 168 * 3
out_sample = 168
walk_timestep = 168
epochs = 8
batch_size = 1

# test predictions
out_pred_list = []

# in training data metrics
in_ap_scores = []
in_prec_scores = []

# out of training data metrics
out_ap_scores = []
out_prec_scores = []

for user_idx in tqdm(range(len(ts_data))):
    out_preds, in_metrics, out_metrics = multi_step_lstm(user_idx, train_len, in_sample, out_sample, walk_timestep, epochs,
                                              batch_size)
    print(in_metrics)
    print(out_metrics)

    out_pred_list.append(out_preds)
    in_ap_scores.append(in_metrics[0])
    in_prec_scores.append(in_metrics[1])
    out_ap_scores.append(out_metrics[0])
    out_prec_scores.append(out_metrics[1])

export_results("individual",
               out_pred_list=out_pred_list,
               in_ap_list=in_ap_scores,
               in_prec_list=in_prec_scores,
               out_ap_list=out_ap_scores,
               out_prec_list=out_prec_scores)


# Create aggregate (1) LSTM model for all users and get metrics for training/testing period
def agg_multi_step_lstm(train_len,
                        in_sample,
                        out_sample,
                        walk_timestep,
                        epochs,
                        batch_size):
    # initializing training data set
    for idx, ts in enumerate(ts_data):
        # retrieving train walks for one user and concatenating it to X, y
        user_X, user_y = create_train_walk(ts, train_len, in_sample, out_sample, walk_timestep)
        # on first iteration, set X and y = to user_X and user_y
        if idx == 0:
            X = user_X
            y = user_y
        else:
            # on subsequent iterations, just concatenate training data from new user with X and y
            X = np.concatenate((X, user_X), axis=0)
            y = np.concatenate((y, user_y), axis=0)

    agg_model = lstm("aggregate", in_sample, out_sample)
    agg_model.train(X, y, epochs, batch_size, shuffle=True, verbose=1)

    # Array to store all the ap/precision scores
    twodim_out_preds = []
    twodim_in_metrics = []
    twodim_out_metrics = []

    # predicting along all the users
    for user_ts in tqdm(ts_data):
        # Array to store the ap/precision scores for current user (in sample and out of sample)
        in_metrics = []
        out_metrics = []

        # multi-step forecast within training data
        in_forecast_steps = int((train_len - in_sample) / out_sample)
        in_probs, in_preds = agg_model.multi_step_forecast(user_ts, train_len, 0, in_forecast_steps)

        # multi-step forecasting out of training data
        out_forecast_steps = int((len(user_ts) - train_len) / out_sample)
        out_probs, out_preds = agg_model.multi_step_forecast(user_ts, train_len, train_len - in_sample,
                                                             out_forecast_steps)

        # adding AP/prec metrics for in sample
        in_metrics.append(agg_model.average_precision(user_ts[in_sample: train_len], in_probs))
        in_metrics.append(agg_model.conditional_precision(user_ts[in_sample: train_len], in_preds))

        # adding AP/prec metrics for out sample
        out_metrics.append(agg_model.average_precision(user_ts[train_len:], out_probs))
        out_metrics.append(agg_model.conditional_precision(user_ts[train_len:], out_preds))

        # Adding metrics from current user_ts to the master metrics list
        print(in_metrics)
        print(out_metrics)
        twodim_out_preds.append(out_preds)
        twodim_in_metrics.append(in_metrics)
        twodim_out_metrics.append(out_metrics)

    return twodim_out_preds, twodim_in_metrics, twodim_out_metrics


# Setting parameters for training and forecasting
train_len = 168 * 38
in_sample = 168 * 3
out_sample = 168
walk_timestep = 168
epochs = 1
batch_size = 16

out_pred_list, in_metrics_list, out_metrics_list = agg_multi_step_lstm(train_len,
                                                                       in_sample, out_sample,
                                                                       walk_timestep,
                                                                       epochs,
                                                                       batch_size)

# Executing agg_model function and exporting results
in_ap_scores = []
in_prec_scores = []

out_ap_scores = []
out_prec_scores = []

for in_pair in in_metrics_list:
    in_ap_scores.append(in_pair[0])
    in_prec_scores.append(in_pair[1])

for out_pair in out_metrics_list:
    out_ap_scores.append(out_pair[0])
    out_prec_scores.append(out_pair[1])

export_results("aggregate",
               out_pred_list=out_pred_list,
               in_ap_list=in_ap_scores,
               in_prec_list=in_prec_scores,
               out_ap_list=out_ap_scores,
               out_prec_list=out_prec_scores)
