import pandas as pd
import numpy as np
import datetime

# convert history into inputs and outputs
def to_supervised(train, n_input, step_size=1, n_out=1, is_y=False):
    # flatten data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, -1])
        # move along defined time step(s)
        in_start += step_size
    #remove y history (as required)
    if is_y == 'False':
        X = np.array(X)
        X = X[:, :, :-1]        
    return np.array(X), np.array(y)


# convert history into inputs and outputs
def seq2seq(train, n_input, step_size=1, n_out=1, is_y=False):
    # flatten data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    #if no sliding window, slice without sliding
    if step_size==0:
        X = data[:(np.floor_divide(data.shape[0], n_input)*n_input)].reshape((np.floor_divide(data.shape[0], n_input)), n_input, data.shape[1])
        y = X[:, :, -1]
        X = X[:, :, :-1]
    else:
        # step over the entire history one time step at a time
        for _ in range(len(train)):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_start + n_input
            # ensure we have enough data for this instance
            if out_end < len(data):
                X.append(data[in_start:in_end, :-1])
                y.append(data[in_start:in_end, -1])
            # move along defined time step(s)
            in_start += step_size
    return np.array(X), np.array(y)


# convert history into inputs and outputs
def seq2last(train, n_input, step_size=1, n_out=1, is_y=False):
    # flatten data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    #if no sliding window, slice without sliding
    if step_size==0:
        X = data[:(np.floor_divide(data.shape[0], n_input)*n_input)].reshape((np.floor_divide(data.shape[0], n_input)), n_input, data.shape[1])
        y = X[:, :, -1]
        X = X[:, :, :-1]
    else:
        # step over the entire history one time step at a time
        for _ in range(len(train)):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_out
            # ensure we have enough data for this instance
            if out_end < len(data):
                X.append(data[in_start:in_end, :-1])
                y.append(data[in_end + n_out, -1])
            # move along defined time step(s)
            in_start += step_size
    return np.array(X), np.array(y)

# split a timeseries dataset into train/test partitions
def split_dataset(data, data_split=0.2):
    data = np.array(data)
    #if number of observations specified manually
    if data_split > 1:
        # split using observation number
        train, test = data[0:-data_split], data[-data_split:]        
    #else use data percentage
    else:
        data_index = round(len(data)*(1-data_split))
        train = data[0:data_index, :]
        test = data[data_index:, :]
    # restructure timeseries into windows of single step data
    train = np.array(np.split(train, len(train) / 1))
    test = np.array(np.split(test, len(test) / 1))
    return train, test


def prepare_data(file_name, w_size, step_size, data_split=0.2, is_y='False', save='False', n_out=1):
    data = pd.read_csv(file_name, header=0, index_col=0)
    train, test = split_dataset(data, data_split)
    trainX, trainy = to_supervised(train, w_size, step_size, n_out, is_y)
    testX, testy = to_supervised(test, w_size, step_size, n_out, is_y)
    t_now = datetime.datetime.today().strftime('%Y%m%d')
    if save == 'True':
        pd.DataFrame(trainX).to_csv(str(t_now)+'_trainX.csv')
        pd.DataFrame(testX).to_csv(str(t_now)+'_testX.csv')
        pd.DataFrame(trainy).to_csv(str(t_now)+'_trainy.csv')
        pd.DataFrame(testy).to_csv(str(t_now)+'_testy.csv')
    return(trainX, trainy, testX, testy)



