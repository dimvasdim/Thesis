import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sqlalchemy import create_engine

# Credentials to connect to the database
username = "username"
password = "DB_password"
hostname = "DB_host"
dbname = "DB_name"


# Split the training and validation datasets using the valid_fraction
def get_data_splits(dataframe, valid_fraction=0.2):
    valid_size = int(len(dataframe) * valid_fraction)
    if valid_size < 1:
        valid_size = 1
    train = dataframe[:-valid_size]
    valid = dataframe[-valid_size:]
    return train, valid


def neural_network(nodes, input_length):
    '''
        Create the neural network
    '''
    model = Sequential()
    model.add(Dense(nodes, input_dim=input_length, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))
    # Compile model
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=["mae"])
    return model


# Create training and validation datasets
def create_train_valid_set():
    # Connect to the database of the e-shop
    engine = create_engine("mysql+mysqlconnector://{user}:{password}@{host}/{dbname}"
                           .format(user=username,
                                   password=password,
                                   host=hostname,
                                   dbname=dbname))

    nn_data = pd.read_sql_table("nn_data", engine)
    nn_data = nn_data.loc[:, nn_data.columns != "week"]
    nn_data = nn_data.loc[:, nn_data.columns != "product_cost"]
    nn_data = nn_data.loc[:, nn_data.columns != "product_max_bound"]

    train = pd.DataFrame(columns=nn_data.columns)
    valid = pd.DataFrame(columns=nn_data.columns)
    for product in nn_data.product_id.unique():
        dataframe = nn_data.loc[nn_data.product_id == product]
        std = dataframe.iloc[:, -1].std()
        mean = dataframe.iloc[:, -1].mean()
        if std <= mean:
            temp_train, temp_valid = get_data_splits(dataframe)
            train = train.append(temp_train, ignore_index=True)
            valid = valid.append(temp_valid, ignore_index=True)

    X_train = train.iloc[:, 0:-1]
    y_train = train.iloc[:, -1]
    X_valid = valid.iloc[:, 0:-1]
    y_valid = valid.iloc[:, -1]

    product_encoder = LabelEncoder()
    product_encoder.fit(X_train["product_id"])
    X_train["product_id"] = product_encoder.transform(X_train["product_id"])
    X_valid["product_id"] = product_encoder.transform(X_valid["product_id"])

    return X_train, X_valid, y_train, y_valid


# Test the neural network and it's performance
def nn_testing():
    X_train, X_valid, y_train, y_valid = create_train_valid_set()

    model = neural_network(23, X_train.shape[1])
    history = model.fit(X_train, y_train,
                        epochs=100, batch_size=256,
                        validation_data=[X_valid, y_valid],
                        verbose=0)
    history_dict = history.history
    # Plots model's training cost/loss and model's validation split cost/loss
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    plt.plot(loss_values, label='training loss')
    plt.plot(val_loss_values, label='val loss')
    plt.legend()
    plt.show()

    y_train_pred = np.round_(model.predict(X_train))
    y_valid_pred = np.round_(model.predict(X_valid))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    valid_mae = mean_absolute_error(y_valid, y_valid_pred)

    # Calculates and prints r2 score of training and validation data
    print("The R2 score on the Train set is: ", r2_score(y_train, y_train_pred))
    print("The R2 score on the Valid set is: ", r2_score(y_valid, y_valid_pred))
    # Calculates and prints mae of training and validation data
    print("The mae on the Train set is:\t{:0.3f}".format(train_mae))
    print("The mean of the Train set is: ", y_train.mean())
    print("The percentage of mae on Train set is: ", (train_mae / y_train.mean()) * 100)
    print("The mae on the Valid set is:\t{:0.3f}".format(valid_mae))
    print("The mean of the Valid set is: ", y_valid.mean())
    print("The percentage of mae on Valid set is: ", (valid_mae / y_valid.mean()) * 100)


# Final training of the neural network
def nn_final_training():
    engine = create_engine("mysql+mysqlconnector://{user}:{password}@{host}/{dbname}"
                           .format(user="kvavliak",
                                   password="DimKvavliak$789",
                                   host="dimvas.pharm24.gr",
                                   dbname="web_db"))

    nn_data = pd.read_sql_table("nn_data", engine)

    nn_data = nn_data.loc[:, nn_data.columns != "week"]
    nn_data = nn_data.loc[:, nn_data.columns != "product_cost"]
    nn_data = nn_data.loc[:, nn_data.columns != "product_max_bound"]

    X_train = nn_data.iloc[:, 0:-1]
    y_train = nn_data.iloc[:, -1]

    product_encoder = LabelEncoder()
    X_train["product_id"] = product_encoder.fit_transform(X_train["product_id"])

    model = neural_network(23, X_train.shape[1])
    model.fit(X_train, y_train,
              epochs=50, batch_size=16,
              verbose=0)
    model.save("final_model.h5")
    return product_encoder

