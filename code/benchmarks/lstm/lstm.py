"""Class file definitions to create LSTM that serves the purposes and functions of the baseline model"""

import pandas as pd
import numpy as np
from keras.models import Sequential
import keras.layers
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from sklearn.metrics import precision_score


def convert_probs(prob_output, threshold):
    pred_output = []
    for prob in prob_output:
        if prob >= threshold:
            pred_output.append(1)
        else:
            pred_output.append(0)

    return pred_output


class lstm:

    def __init__(self, type, input_steps, output_steps, input_features=1):
        if type == "individual":
            neurons = 32
        elif type == "aggregate":
            neurons = 64
        else:
            neurons = 32

        self.input_steps = input_steps
        self.input_features = input_features
        self.output_steps = output_steps

        # creating model
        # default neurons trained on individual models is 32
        # default neurons trained on aggregate model is 64
        self.model = Sequential()
        self.model.add(LSTM(neurons, activation='tanh', input_shape=(input_steps, input_features), return_sequences=True))
        self.model.add(keras.layers.Flatten())
        self.model.add(Dense(neurons, activation='relu'))
        self.model.add(Dense(output_steps, activation='sigmoid'))

    '''
    Params:
    X, y: Outputs from create_train_walk function
    epochs: Number of times to iterate over training set when training
    batch_size: Number of training sequences to take in at one training step to update model
    
    :return: none
    '''

    def train(self, X, y, epochs, batch_size, shuffle=False, verbose=0):
        # compiling model
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer)
        # training network
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=shuffle)

    '''
    Params:
    input_seq: Input sequence (has to be same length as in_sample parameter
     from create_train_walk function in the baseline_model notebook)
    '''

    def forecast(self, input_seq):
        input_x = np.array(input_seq)
        input_x = input_x.reshape(1, len(input_seq), 1)
        yhat = self.model.predict(input_x, verbose=0)
        yhat = yhat.flatten()
        return yhat

    '''
    Params:
    full_seq: Full time series model (48 weeks in our case)
    train_len: How much of the full sequence counts as training data
    start_point: Where we want to feed input from (whether in the training sample or out of the sample)
    num_steps: number of output sequences to recurse over
    '''

    def multi_step_forecast(self, full_seq, train_len, start_point, num_steps):
        # Array to store all the original probabilities/predictions of all the steps
        multi_step_probs = []
        multi_step_preds = []

        # Defining first index to feed into model (last element of train sequence - in_sample)
        curr_start = start_point
        curr_input_seq = full_seq[curr_start: curr_start + self.input_steps]

        # Number of times forecast used predictions as input
        out_model_count = 0
        for i in range(num_steps):
            # Predict probabilities using forecast and add it to the multi_step_probs array
            curr_step_probs = self.forecast(curr_input_seq)
            multi_step_probs.extend(curr_step_probs)

            # Convert the probabilities to predictions and recurse over them
            curr_step_preds = convert_probs(curr_step_probs, 0.20)
            multi_step_preds.extend(curr_step_preds)

            # Resetting curr_input_seq to include output if we're out of training sequence
            curr_start += self.output_steps
            if (curr_start + self.input_steps) > train_len:
                # if the end of the new curr seq is greater than train_len, get the last available input
                # sequence steps and append the last output to make the new input sequence
                curr_input_seq = curr_input_seq[-(self.input_steps - self.output_steps):] + list(curr_step_preds)
                out_model_count += 1
            else:
                # if we're still in bounds of training sequence, just use next set of training data
                curr_input_seq = full_seq[curr_start: curr_start + self.input_steps]

        print("Number of times model used output predictions as input: ", out_model_count)
        return multi_step_probs, multi_step_preds

    '''
    Params:
    true_data: validation data that we are comparing model output to
    model_probs: probability outputs from LSTM that we want to measure
    
    :return: returns ap averaged across output sample size; If no ap could be calculated because there are
    no true rides, return nan
    '''

    def average_precision(self, true_data, model_probs):
        # Exception handling
        if len(true_data) != len(model_probs):
            raise RuntimeError("true_data and model_probs must have the same length")

        def one_step_ap(a, b):
            probs_df = pd.DataFrame({
                "Probs": b,
                "Time Index": list(range(len(b))),
                "True Label": a
            })
            probs_df.sort_values(["Probs", "True Label"], ascending=False, inplace=True)

            # divide by number of true rides
            denom = a.count(1)
            if denom == 0:
                return np.nan

            # initialize summation of ranked precision
            sigma = 0

            no_correct = 0
            for i in range(len(probs_df["True Label"])):
                if list(probs_df["True Label"])[i] == 1:
                    no_correct += 1
                    p_of_k = no_correct / (i + 1)
                    sigma += p_of_k

            ap = sigma / denom

            return ap

        multi_step_ap = []
        # Apply one_step_ap to predictions based on length conditions
        if len(model_probs) % self.output_steps != 0:
            raise RuntimeError("true_data and model_probs must have length divisible by the output steps")

        # take the ap output step by output step and average them
        ap_steps = int(len(model_probs) / self.output_steps)
        for step in range(ap_steps):
            curr_true_data = true_data[(step * self.output_steps): ((step + 1) * self.output_steps)]
            curr_model_probs = model_probs[(step * self.output_steps): ((step + 1) * self.output_steps)]
            multi_step_ap.append(one_step_ap(curr_true_data, curr_model_probs))

        # Remove nan from list
        multi_step_ap = [item for item in multi_step_ap if not (pd.isnull(item))]

        # If no individual ap's could be calculated (0 actual events in true data), return NaN
        if len(multi_step_ap) == 0:
            return np.nan

        return sum(multi_step_ap) / len(multi_step_ap)

    '''
    Params:
    true_data: validation data that we are comparing model output to
    model_preds: label outputs from LSTM that we want to measure
    
    :return: returns cp averaged across output sample size; If no cp could be calculated because there are
    no true rides, return nan
    '''

    def conditional_precision(self, true_data, model_preds):
        # Exception handling
        if len(true_data) != len(model_preds):
            raise RuntimeError("true_data and model_preds must have the same length")

        def one_step_cp(a, b):
            preds_df = pd.DataFrame({
                "Preds": b,
                "Time Index": list(range(len(b))),
                "True Label": a
            })
            true_preds_df = preds_df.loc[preds_df["True Label"] == 1]

            # divide by number of true rides
            denom = list(true_preds_df["True Label"]).count(1)
            if denom == 0:
                return np.nan

            # Use sklearn metrics to find precision
            return precision_score(true_preds_df["True Label"], true_preds_df["Preds"], zero_division=0)

        multi_step_cp = []
        # Apply one_step_cp to predictions based on length conditions
        if len(model_preds) % self.output_steps != 0:
            raise RuntimeError("true_data and model_preds must have length divisible by the output steps")

        # take the cp output step by output step and average them
        cp_steps = int(len(model_preds) / self.output_steps)
        for step in range(cp_steps):
            curr_true_data = true_data[(step * self.output_steps): ((step + 1) * self.output_steps)]
            curr_model_probs = model_preds[(step * self.output_steps): ((step + 1) * self.output_steps)]
            multi_step_cp.append(one_step_cp(curr_true_data, curr_model_probs))

        # Remove nan from list
        multi_step_cp = [item for item in multi_step_cp if not (pd.isnull(item))]

        # If no individual cp's could be calculated (0 actual events in true data), return NaN
        if len(multi_step_cp) == 0:
            return np.nan

        return sum(multi_step_cp) / len(multi_step_cp)
