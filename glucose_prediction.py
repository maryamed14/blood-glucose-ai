###################### CONFIG EXPERIMENT #############################
batch_size = 1024
patience = 100
max_epoch = 500

history_length = 8  # 2h

horizons = [2, 4]
models = ['Linear', 'LSTM']
repeated_examples = True

###################### END CONFIG EXPERIMENT #############################

import time
import numpy as np
import pandas as pd
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def read_windows_no_duplicated(horizon):
    file_x_train = f"d:/TFM-BLOOD-GLUCOSE/Cluster windows/cluster0/x_train_windows_horizon_{horizon}.npy"
    file_y_train = f"d:/TFM-BLOOD-GLUCOSE/Cluster windows/cluster0/y_train_windows_horizon_{horizon}.npy"
    file_x_val = f"d:/TFM-BLOOD-GLUCOSE/Cluster windows/cluster0/x_val_windows_horizon_{horizon}.npy"
    file_y_val = f"d:/TFM-BLOOD-GLUCOSE/Cluster windows/cluster0/y_val_windows_horizon_{horizon}.npy"
    file_x_test = f"d:/TFM-BLOOD-GLUCOSE/Cluster windows/cluster0/x_test_windows_horizon_{horizon}.npy"
    file_y_test = f"d:/TFM-BLOOD-GLUCOSE/Cluster windows/cluster0/y_test_windows_horizon_{horizon}.npy"

    print(f"Reading training files {file_x_train} and {file_y_train}...")
    print(f"Reading training files {file_x_val} and {file_y_val}...")
    print(f"Reading test files {file_x_test} and {file_y_test}...")

    x_train = np.load(file_x_train)
    y_train = np.load(file_y_train)
    x_val = np.load(file_x_val)
    y_val = np.load(file_y_val)
    x_test = np.load(file_x_test)
    y_test = np.load(file_y_test)

    ####### Remove duplicated ###########

    ### On Train data #############

    # merge arrays
    df_x = pd.DataFrame(x_train)  # All x
    df_y = pd.DataFrame(y_train[:, -1])  # Last column of y
    df = pd.concat([df_x, df_y], axis=1)

    # Find duplicated
    dup = df.duplicated()
    print()
    print(f'Array Train duplicated:')
    print(dup.value_counts())

    # Remove duplicated
    df = df[~dup]
    df.reset_index(inplace=True, drop=True)

    x_train = df.iloc[:, :-1].values  # Get all columns but the last one
    y_train = df.iloc[:, -1:].values  # Get just the last column

    ### On Validation data #############

    # merge arrays
    df_x = pd.DataFrame(x_val)  # All x
    df_y = pd.DataFrame(y_val[:, -1])  # Last column of y
    df = pd.concat([df_x, df_y], axis=1)

    # Find duplicated
    dup = df.duplicated()
    print()
    print(f'Array val duplicated:')
    print(dup.value_counts())

    # Remove duplicated
    df = df[~dup]
    df.reset_index(inplace=True, drop=True)

    x_val = df.iloc[:, :-1].values  # Get all columns but the last one
    y_val = df.iloc[:, -1:].values  # Get just the last column

    ### On Test data #############

    # merge arrays
    df_x = pd.DataFrame(x_test)  # All x
    df_y = pd.DataFrame(y_test[:, -1])  # Last column of y
    df = pd.concat([df_x, df_y], axis=1)

    # Find duplicated
    dup = df.duplicated()
    print()
    print(f'Array test duplicated:')
    print(dup.value_counts())

    # Remove duplicated
    df = df[~dup]
    df.reset_index(inplace=True, drop=True)

    x_test = df.iloc[:, :-1].values  # Get all columns but the last one
    y_test = df.iloc[:, -1:].values  # Get just the last column

    return x_train, y_train, x_val, y_val, x_test, y_test


def read_windows_with_duplicated(horizon):
    file_x_train = f"d:/TFM-BLOOD-GLUCOSE/Cluster windows/cluster0/x_train_windows_horizon_{horizon}.npy"
    file_y_train = f"d:/TFM-BLOOD-GLUCOSE/Cluster windows/cluster0/y_train_windows_horizon_{horizon}.npy"
    file_x_val = f"d:/TFM-BLOOD-GLUCOSE/Cluster windows/cluster0/x_val_windows_horizon_{horizon}.npy"
    file_y_val = f"d:/TFM-BLOOD-GLUCOSE/Cluster windows/cluster0/y_val_windows_horizon_{horizon}.npy"
    file_x_test = f"d:/TFM-BLOOD-GLUCOSE/Cluster windows/cluster0/x_test_windows_horizon_{horizon}.npy"
    file_y_test = f"d:/TFM-BLOOD-GLUCOSE/Cluster windows/cluster0/y_test_windows_horizon_{horizon}.npy"

    print(f"Reading training files {file_x_train} and {file_y_train}...")
    print(f"Reading training files {file_x_val} and {file_y_val}...")
    print(f"Reading test files {file_x_test} and {file_y_test}...")

    x_train = np.load(file_x_train)
    y_train = np.load(file_y_train)
    x_val = np.load(file_x_val)
    y_val = np.load(file_y_val)
    x_test = np.load(file_x_test)
    y_test = np.load(file_y_test)

    # Using only the horizon measurement as a label
    y_train = y_train[:, -1]
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_val = y_val[:, -1]
    y_val = y_val.reshape((y_val.shape[0], 1))
    y_test = y_test[:, -1]
    y_test = y_test.reshape((y_test.shape[0], 1))

    return x_train, y_train, x_val, y_val, x_test, y_test


## LSTM and linear model architectures ################################################
from keras.models import Model
from keras.layers import Dense, LSTM, GRU, Lambda, dot, concatenate, Activation, Input


# LSTM
class LSTMModel:
    def __init__(self, input_shape, nb_output_units, nb_hidden_units=128):
        self.input_shape = input_shape
        self.nb_output_units = nb_output_units
        self.nb_hidden_units = nb_hidden_units

    def __repr__(self):
        return 'LSTM_{0}_units_{1}_layers_dropout={2}_{3}'.format(self.nb_hidden_units, self.nb_layers, self.dropout,
                                                                  self.recurrent_dropout)

    def build(self):
        # input
        i = Input(shape=self.input_shape)

        # add LSTM layer
        x = LSTM(self.nb_hidden_units)(i)

        x = Dense(self.nb_output_units, activation=None)(x)

        return Model(inputs=[i], outputs=[x])


# Linear
class LinearModel:
    def __init__(self, input_shape, nb_output_units):
        self.input_shape = input_shape
        self.nb_output_units = nb_output_units

    def __repr__(self):
        return 'Linear'

    def build(self):
        i = Input(shape=self.input_shape)
        x = Dense(self.nb_output_units, activation=None)(i)

        return Model(inputs=[i], outputs=[x])


## Model Functions ################################################
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
from tensorflow import keras


def RMSE(output, target):
    output = K.cast(output, 'float32')
    target = K.cast(target, 'float32')

    return K.sqrt(K.mean((output - target) ** 2))


def build_model(model, weights=''):
    # build & compile model
    m = model.build()

    m.compile(loss=RMSE,
              optimizer=keras.optimizers.Adam(),
              metrics=[keras.metrics.mean_squared_error, keras.metrics.RootMeanSquaredError()])
    if weights:
        print(f"Weights: {weights}")
        m.load_weights(weights)
    return m


def callbacks(filepath, early_stopping_patience):
    callbacks = []
    callbacks.append(ModelCheckpoint(filepath=filepath,
                                     monitor='loss',
                                     save_best_only=True,
                                     save_weights_only=True))
    callbacks.append(EarlyStopping(monitor='loss', patience=early_stopping_patience))
    return callbacks


import numpy as np


def prepare_model_LSTM(history_length, nb_hidden_units=128, weights=''):
    model = LSTMModel(input_shape=(history_length, 1), nb_output_units=1, nb_hidden_units=nb_hidden_units)
    return build_model(model, weights)


def prepare_model_linear(history_length, weights=''):
    model = LinearModel(input_shape=(history_length,), nb_output_units=1)
    return build_model(model, weights)


############## TRAIN FUNCTION #####################################
def train(x_train, y_train, x_val, y_val, model, horizon, save_filepath="", early_stopping_patience=patience):
    history_length = 8
    x_train = np.reshape(x_train, (x_train.shape[0], history_length, 1))

    x_val = np.reshape(x_val, (x_val.shape[0], history_length, 1))
    validation_data = (x_val, y_val)

    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     validation_data=validation_data,
                     epochs=max_epoch,
                     # shuffle=True,
                     callbacks=callbacks(save_filepath + str(horizon),
                                         early_stopping_patience)
                     )

    return hist, model


# %%
import matplotlib.pyplot as plt


def get_train_plots_loss(hist: keras.callbacks, name: str):
    fig, ax = plt.subplots()

    # data
    x_epoch = hist.epoch
    y_val_loss = hist.history['val_loss']
    y_train_loss = hist.history['loss']

    # Create a line plots
    ax.plot(x_epoch, y_val_loss, label='Validation loss', color='orange', linestyle='-')
    ax.plot(x_epoch, y_train_loss, label='Train loss', color='blue', linestyle='-')

    # Add labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and validation loss')

    ax.legend()

    fig.savefig(name + '_Loss.pdf', dpi=350, bbox_inches='tight')


def get_train_plots_RMSE(hist: keras.callbacks, name: str):
    fig, ax = plt.subplots()

    # Sample data
    x_epoch = hist.epoch
    y_val_rmse = hist.history['val_root_mean_squared_error']
    y_train_rmse = hist.history['root_mean_squared_error']

    # Create a line plot
    ax.plot(x_epoch, y_val_rmse, label='Validation RMSE', color='orange', linestyle='-')
    ax.plot(x_epoch, y_train_rmse, label='Train RMSE', color='blue', linestyle='-')

    # Add labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('Training and validation RMSE')

    # Add a legend
    ax.legend()

    # Display the plot
    fig.savefig(name + '_RMSE.pdf', dpi=350, bbox_inches='tight')


def get_train_plots_MSE(hist: keras.callbacks, name: str):
    fig, ax = plt.subplots()

    # Sample data
    x_epoch = hist.epoch
    y_val_mse = hist.history['val_mean_squared_error']
    y_train_mse = hist.history['mean_squared_error']

    # Create a line plot
    ax.plot(x_epoch, y_val_mse, label='Validation MSE', color='orange', linestyle='-')
    ax.plot(x_epoch, y_train_mse, label='Train MSE', color='blue', linestyle='-')

    # Add labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('Training and validation MSE')

    # Add a legend
    ax.legend()

    # Display the plot
    fig.savefig(name + '_MSE.pdf', dpi=350, bbox_inches='tight')


# Results    ################################################
## Functions ################################################
import matplotlib.pyplot as plt


def clarke_error_grid(ref_values, pred_values, title_string, show_plot=False):
    """
      This function takes in the reference values and the prediction values as lists and returns a list with each index corresponding to the total number
     of points within that zone (0=A, 1=B, 2=C, 3=D, 4=E) and the plot
    """
    # Checking to see if the lengths of the reference and prediction arrays are the same
    assert (len(ref_values) == len(
        pred_values)), "Unequal number of values (reference : {0}) (prediction : {1}).".format(len(ref_values),
                                                                                               len(pred_values))

    # Checks to see if the values are within the normal physiological range, otherwise it gives a warning
    if max(ref_values) > 500 or max(pred_values) > 500:
        print(
            "Input Warning: the maximum reference value {0} or the maximum prediction value {1} exceeds the normal physiological range of glucose (<400 mg/dl).".format(
                max(ref_values), max(pred_values)))
    if min(ref_values) < 0 or min(pred_values) < 0:
        print(
            "Input Warning: the minimum reference value {0} or the minimum prediction value {1} is less than 0 mg/dl.".format(
                min(ref_values), min(pred_values)))

    values_out_grid = sum(value > 500 for value in pred_values) + sum(value < 0 for value in pred_values)
    print(f"Number of values outside the grid: {values_out_grid}")

    if show_plot:
        # Clear plot
        plt.clf()

        # Set up plot
        plt.scatter(ref_values, pred_values, marker='o', color='black', s=8)
        plt.title(title_string + " Clarke Error Grid")
        plt.xlabel("Reference Concentration (mg/dl)")
        plt.ylabel("Prediction Concentration (mg/dl)")
        plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
        plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
        plt.gca().set_facecolor('white')

        # Set axes lengths
        plt.gca().set_xlim([0, 500])
        plt.gca().set_ylim([0, 500])
        plt.gca().set_aspect((500) / (500))

        # Plot zone lines
        plt.plot([0, 500], [0, 500], ':', c='black')  # Theoretical 45 regression line
        plt.plot([0, 175 / 3], [70, 70], '-', c='black')
        # plt.plot([175/3, 320], [70, 500], '-', c='black')
        plt.plot([175 / 3, 500 / 1.2], [70, 500], '-',
                 c='black')  # Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
        plt.plot([70, 70], [84, 500], '-', c='black')
        plt.plot([0, 70], [180, 180], '-', c='black')
        plt.plot([70, 290], [180, 500], '-', c='black')
        # plt.plot([70, 70], [0, 175/3], '-', c='black')
        plt.plot([70, 70], [0, 56], '-', c='black')  # Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
        # plt.plot([70, 500],[175/3, 320],'-', c='black')
        plt.plot([70, 500], [56, 320], '-', c='black')
        plt.plot([180, 180], [0, 70], '-', c='black')
        plt.plot([180, 500], [70, 70], '-', c='black')
        plt.plot([240, 240], [70, 180], '-', c='black')
        plt.plot([240, 500], [180, 180], '-', c='black')
        plt.plot([130, 180], [0, 70], '-', c='black')

        # Add zone titles
        plt.text(30, 15, "A", fontsize=15)
        plt.text(370, 260, "B", fontsize=15)
        plt.text(280, 370, "B", fontsize=15)
        plt.text(160, 370, "C", fontsize=15)
        plt.text(160, 15, "C", fontsize=15)
        plt.text(30, 140, "D", fontsize=15)
        plt.text(370, 120, "D", fontsize=15)
        plt.text(30, 370, "E", fontsize=15)
        plt.text(370, 15, "E", fontsize=15)

        # plt.savefig(f'plots/Clarke_Error_Grid{title_string}.pdf')
        plt.savefig(f'plots/Clarke_Error_Grid{title_string}.jpg', dpi=350, bbox_inches='tight')

    # Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        if (ref_values[i] <= 70 and pred_values[i] <= 70) or (
                pred_values[i] <= 1.2 * ref_values[i] and pred_values[i] >= 0.8 * ref_values[i]):
            zone[0] += 1  # Zone A

        elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (ref_values[i] <= 70 and pred_values[i] >= 180):
            zone[4] += 1  # Zone E

        elif ((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110) or (
                (ref_values[i] >= 130 and ref_values[i] <= 180) and (pred_values[i] <= (7 / 5) * ref_values[i] - 182)):
            zone[2] += 1  # Zone C
        elif (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180)) or (
                ref_values[i] <= 175 / 3 and pred_values[i] <= 180 and pred_values[i] >= 70) or (
                (ref_values[i] >= 175 / 3 and ref_values[i] <= 70) and pred_values[i] >= (6 / 5) * ref_values[i]):
            zone[3] += 1  # Zone D
        else:
            zone[1] += 1  # Zone B

    return zone, values_out_grid


def get_values_per_zone(values):
    keys = ['A', 'B', 'C', 'D', 'E']
    return {key: value for key, value in zip(keys, values)}


def show_results(model, x_test, y_test, plot_name='', plot_flag=False):
    print("Calculating test score")
    test_score = model.evaluate(x_test, y_test)

    # PREDICTION
    print("########################################################################")
    print("Predicting BG values...")
    BG_predicted_values = model.predict(x_test)
    print("########################################################################")
    print()

    # Clarke error grid
    values, out_values = clarke_error_grid(y_test, BG_predicted_values, plot_name, show_plot=plot_flag)

    values_zones = get_values_per_zone(values)
    predicted_values_len = y_test.shape[0]
    perc_values_zones = [value / predicted_values_len * 100 for value in values_zones.values()]

    perc_values_zones = get_values_per_zone(perc_values_zones)
    perc_values_zones['out'] = out_values

    print('Test score: ', test_score)
    print('Percentage values zones: ', perc_values_zones)

    return test_score, perc_values_zones, BG_predicted_values

############################# RUNNING ##########################################################

for current_horizon in horizons:  # For all horizons -------------------------------------
    if repeated_examples:
        x_train, y_train, x_val, y_val, x_test, y_test = read_windows_with_duplicated(horizon=current_horizon)
    else:
        x_train, y_train, x_val, y_val, x_test, y_test = read_windows_no_duplicated(horizon=current_horizon)

    for current_model in models:  # For all models ---------------------------------------

        if current_model == 'Linear':  # prepare linear model
            model = prepare_model_linear(history_length)
        else:
            model = prepare_model_LSTM(history_length)

        print("########################################################################")
        print(f'START training {current_model} horizon={current_horizon}')
        print()

        # Record the TRAINING start time
        start_time = time.time()

        hist, model_trained = train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, model=model,
                                    horizon=current_horizon, save_filepath=f"models/{current_model}")

        # Record the TRAINING end time
        end_time = time.time()
        # Calculate the elapsed time of TRAINING
        elapsed_time = end_time - start_time
        print(f"Training time for {current_model} horizon={current_horizon}: {elapsed_time} seconds")

        get_train_plots_loss(hist, f"plots/{current_model}_H{current_horizon}")
        get_train_plots_MSE(hist, f"plots/{current_model}_H{current_horizon}")

        df_hist = pd.DataFrame(data=hist.history)
        df_hist.to_csv(f'hist/df_hist_{current_model}_H{current_horizon}.csv')

        hist_best = df_hist.iloc[df_hist['loss'].idxmin()]
        hist_best['epoch'] = df_hist.shape[0]
        hist_best.to_csv(f'hist/hist_best_{current_model}_H{current_horizon}.csv')

        print()
        print(f'END training {current_model} horizon={current_horizon}')
        print("########################################################################")
        print()

        # ## TEST ################################

        print(f'Test results for {current_model} horizon={current_horizon}')

        # Record the TEST start time
        start_time = time.time()

        test_score, perc_zones, bg_predict = show_results(model=model_trained, x_test=x_test, y_test=y_test,
                                                          plot_name=f'_{current_model}_H{current_horizon}',
                                                          plot_flag=True)

        # Record the TEST end time
        end_time = time.time()
        # Calculate the elapsed time of TEST
        elapsed_time = end_time - start_time
        print(f"Testing time for {current_model} horizon={current_horizon}: {elapsed_time} seconds")

        # Save scores
        pd.Series(test_score).to_csv(f'test results/test_score_{current_model}_H{current_horizon}.csv')

        # Save zones
        zA = perc_zones['A']
        zB = perc_zones['B']
        zC = perc_zones['C']
        zD = perc_zones['D']
        zE = perc_zones['E']
        out_values = perc_zones['out']
        zAB = zA + zB

        data = {'Zone A': [zA], 'Zone B': [zB], 'Zona A + B': [zAB], 'Zone C': [zC], 'Zone D': [zD], 'Zone E': [zE], 'Out':out_values}
        df_zones = pd.DataFrame(data=data).round(2)
        df_zones.to_csv(f'test results/zones_{current_model}_H{current_horizon}.csv')

        # Save predictions and references
        df_test_results = pd.DataFrame({'y_test': y_test.ravel(), 'y_predict': bg_predict.ravel()})
        df_test_results.to_parquet(f'test results/df_test_results_vectors_{current_model}_H{current_horizon}.parquet')

        print(f'Test results of {current_model} horizon={current_horizon} - DONE')
        print('____________________________________________________________________________')
        print()
