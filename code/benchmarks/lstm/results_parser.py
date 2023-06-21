"""File to parse through the html output of the jupyter notebook of the 
execution of multi_step_lstm individual/aggregate model cell"""

import pandas as pd
import numpy as np

filename = "individual_results.html"
output_prefix = "individual"

with open(filename, 'r', encoding='utf8') as results:
    all_lines = results.readlines()
    results.close()

data_indices = []  # line indices in file that contains the relevant output data
metrics_list = []  # after parsing, we'll store the metrics in here

for idx, line in enumerate(all_lines):
    if "&quot;[" in line:
        data_indices.append(idx)

for idx in data_indices:
    # cleaning data and removing trivial pickups from last loop parse
    metric_output = all_lines[idx]
    metric_output = metric_output.strip()

    # removing html characters
    cleaned_string = metric_output.split("[")[1]
    cleaned_string = cleaned_string.split("]")[0]

    # turning string into list
    curr_metrics = cleaned_string.split(', ')
    for str_metric in curr_metrics:
        str_metric = str_metric.strip()
    curr_metrics = np.array(curr_metrics)

    # turning string elements into floats
    curr_metrics = curr_metrics.astype(float)
    metrics_list.append(curr_metrics)

ap_scores = []
prec_scores = []

for pair in metrics_list:
    print(pair)
    ap_scores.append(pair[0])
    prec_scores.append(pair[1])

# in training data metrics
in_ap_scores = ap_scores[::2]
in_prec_scores = prec_scores[::2]

# out of training data metrics
out_ap_scores = ap_scores[1::2]
out_prec_scores = prec_scores[1::2]


# get averages
def average(lst):
    real_vals_lst = [x for x in lst if not np.isnan(x)]
    return sum(real_vals_lst) / len(real_vals_lst)


# Putting results in a dataframe and exporting
def export_results(results_name, in_ap_list, in_prec_list, out_ap_list, out_prec_list):
    user_results = pd.DataFrame({
        "User Index": list(range(len(in_ap_list))),
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

export_results(output_prefix, in_ap_list=in_ap_scores,
               in_prec_list=in_prec_scores,
               out_ap_list=out_ap_scores,
               out_prec_list=out_prec_scores)