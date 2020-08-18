from tl4sm.prepare_data import split_dataset
from numpy import array, stack
from pandas import read_csv, DataFrame
from pathlib import Path
from keras.models import load_model, clone_model
import time
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, RepeatVector, ConvLSTM2D
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
from tl4sm.prepare_data import to_supervised
from keras.utils import to_categorical
from numpy import argmax
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report




#function to bin data and return dataframe
def bin_data(dataset, target, med, high):
    dataset['bin'] = 0
    dataset['bin'][(dataset[target] < high) & (dataset[target] > med)] = 1
    dataset['bin'][(dataset[target] >= high)] = 2
    return dataset

#function to view the training history of each model
def view_loss(history, exp_num):
    plt.plot(history.history['loss'],label='Train')
    plt.plot(history.history['val_loss'],label='Val')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('../Plots/loss-history'+str(exp_num)+'.png')
    plt.show()
    
#function to view the training history of each model
def view_acc(history, exp_num):
    plt.close()
    plt.plot(history.history['acc'],label='Accuracy')
    plt.plot(history.history['loss'],label='Loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('../Plots/loss-history'+str(exp_num)+'.png')
    plt.show()


# function to build a ConvLSTM model
def build_model(train, epochs, batch_size, lr, n_steps, n_length, n_input, source, exp_num, data_percent, verbose, n_out=10, batch_norm=True, plot=False):
    #prepare data
    train_x, train_y = to_supervised(train, n_input, step_size=1, n_out=1, is_y=True)
    #data percentage
    train_ind = int(round(len(train_x)*(data_percent)))
    train_x = train_x[-train_ind:, :]
    train_y = train_y[-train_ind:, :]
    # define parameters
    n_features, n_outputs = train_x.shape[2], train_y.shape[1]
    # reshape into subsequences [samples, timesteps, rows, cols, channels]
    train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features)) 
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    train_y =to_categorical(train_y)
    # define model
    model = Sequential()
    if batch_norm:
        model.add(BatchNormalization(input_shape=(n_steps, 1, n_length, n_features)))
    model.add(ConvLSTM2D(filters=128, kernel_size=(1,4), activation='relu', return_sequences=True, input_shape=(n_steps, 1, n_length, n_features))) 
    model.add(ConvLSTM2D(filters=128, kernel_size=(1,4), activation='relu',  input_shape=(n_steps, 1, n_length, n_features))) 
    model.add(Flatten())
    model.add(RepeatVector(n_outputs))
    if batch_norm:
        model.add(BatchNormalization(input_shape=(n_steps, 1, n_length, n_features)))
    model.add((LSTM(100, activation='relu', dropout=0.05, return_sequences=True)))
    model.add((LSTM(100, activation='relu', dropout=0.1, return_sequences=True)))
    model.add((Dense(500, activation='relu'))) 
    model.add((Dense(3, activation='softmax')))
    opt = optimizers.Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics = ['acc'])
    model.build()
    #use epoch checkpoints to ensure the best epoch is used
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=epochs)
    checkpointer = ModelCheckpoint(filepath="../Models/best_weights.hdf5", 
                               monitor = 'val_acc',
                               verbose=2, 
                               save_best_only=True)
    callbacks_list = [checkpointer, es] #early
    #record time
    tic = time.time()
    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, shuffle=False, callbacks=callbacks_list, batch_size=batch_size, verbose=verbose, validation_split=0.1) 
    #record time
    toc = time.time()
    totalTime = toc-tic
    if plot:
        view_loss(history, str(exp_num))
    model.load_weights('../Models/best_weights.hdf5')
    model.save('../Models/model_'+str(source)+'.h5')
    return model, totalTime

# make a forecast
def forecast(model, history, n_steps, n_length, n_input):
    # flatten data
    data = array(history)
    cols = data.shape[2]
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_ = data[-n_input:, :]
    # reshape into [samples, timesteps, rows, cols, channels]
    input_x = input_.reshape((1, n_steps, 1, n_length, cols))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


#function to evaluate the model
def evaluate_model(train, test, n_input, n_length, batch_size, lr, source, exp_num, epochs, n_out, data_percent, verbose, batch_norm=True, plot=False):
    #define number of subsequence time steps
    n_steps = int(n_input/n_length)
    #build model
    model, tr_time = build_model(train, epochs, batch_size, lr, n_steps, n_length, n_input, source, exp_num, data_percent, verbose, n_out=10, batch_norm=batch_norm, plot=plot)
    # history is a list of training data
    history = [x for x in train]
    # walk-forward validation over each timestep
    predictions = list()
    for i in range(len(test)):
        # predict the timestep
        yhat_sequence = forecast(model, history, n_steps, n_length, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next timestep
        history.append(test[i, :])
    # evaluate predictions days for each timestep
    predictions = array(predictions)
    test1 = test[:, :, -1]
    YPred = argmax(predictions.reshape(predictions.shape[0], (predictions.shape[1]*predictions.shape[2])), out=None, axis=1)
    YPred = YPred.reshape(YPred.shape[0], 1)
    df = stack((YPred, test1))
    df = df.transpose()
    df = df.reshape(df.shape[1], 2)
    DataFrame(df).to_csv('../Results/Files/2DConvLSTMAE_TL_'+str(exp_num)+'.csv')
    cm = confusion_matrix(test1, YPred)
    print(cm)
    f1 = f1_score(test1, YPred, average='weighted')
    acc = accuracy_score(test1, YPred)
    print(classification_report(test1, YPred))
    return f1, acc, tr_time, f1, acc, tr_time

#function to evaluate the model
def evaluate_model_reuse(train, test, n_input, n_length, batch_size, lr, source, exp_num, epochs, model_name, data_percent, n_out, batch_norm=True, plot=False):
    #define number of subsequence time steps
    n_steps = int(n_input/n_length)  
    #prepare data
    train_x, train_y = to_supervised(train, n_input, step_size=1, n_out=1, is_y=True)
    # define parameters
    n_features = train_x.shape[2]
    # define model
    model = load_model(model_name)
    model = clone_model(model)
    model.build()
    opt = optimizers.Adam()
    model.compile(loss='categorical_crossentropy', metrics = ['acc'], optimizer=opt)
    #model.summary()    
    model.load_weights(model_name)
    #data percentage
    train_ind = int(round(len(train_x)*(data_percent)))
    train_x = train_x[-train_ind:, :]
    train_y = train_y[-train_ind:, :]
    # define parameters
    n_features = train_x.shape[2]
    # reshape into subsequences [samples, timesteps, rows, cols, channels]
    train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features))
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1)) 
    train_y =to_categorical(train_y)
    #record time
    tic = time.time()
    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, shuffle=False, batch_size=batch_size, verbose=2) 
    #record time
    toc = time.time()
    totTime = toc-tic
    #model.load_weights('../Models/best_weights_TL.hdf5')
    model.save('../Models/model_TL_'+str(exp_num)+'.h5')
    if plot:
        view_loss(history, str(exp_num))
    # history is a list of training data
    history = [x for x in train]
    # walk-forward validation over each timestep
    predictions = list()
    for i in range(len(test)):
        # predict the timestep
        yhat_sequence = forecast(model, history, n_steps, n_length, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next timestep
        history.append(test[i, :])
    # evaluate predictions days for each timestep
    predictions = array(predictions)
    test1 = test[:, :, -1]
    YPred = argmax(predictions.reshape(predictions.shape[0], (predictions.shape[1]*predictions.shape[2])), out=None, axis=1)
    YPred = YPred.reshape(YPred.shape[0], 1)
    df = stack((YPred, test1))
    df = df.transpose()
    df = df.reshape(df.shape[1], 2)
    DataFrame(df).to_csv('../Results/Files/2DConvLSTMAE_TL_'+str(exp_num)+'.csv')
    cm = confusion_matrix(test1, YPred)
    print(cm)
    f1 = f1_score(test1, YPred, average='weighted')
    acc = accuracy_score(test1, YPred)
    print(classification_report(test1, YPred))    
    return f1, acc, totTime

#function to evaluate the model
def evaluate_model_tl(train, test, n_input, n_length, batch_size, lr, source, exp_num, epochs, model_name, data_percent, n_layers, n_out, batch_norm=False, plot=False):
    #define number of subsequence time steps
    n_steps = int(n_input/n_length)    
    # load pretrained model
    model = load_model(model_name)
    #prepare data
    train_x, train_y = to_supervised(train, n_input, step_size=1, n_out=1, is_y=True)
    #data percentage
    train_ind = int(round(len(train_x)*(data_percent)))
    train_x = train_x[-train_ind:, :]
    train_y = train_y[-train_ind:, :]
    # define parameters
    n_features = train_x.shape[2]
    # reshape into subsequences [samples, timesteps, rows, cols, channels]
    train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features))
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    train_y =to_categorical(train_y)
    # fix the layers as indicated by the papameter 'layers'
    for layer in model.layers[:-n_layers]:
        layer.trainable = False
    #check trainable status of individual layers
    for layer in model.layers:
        print(layer, layer.trainable)                       
    opt = optimizers.Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=opt)
    #model.summary()
    #record time
    tic = time.time()
    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, shuffle=False, batch_size=batch_size, verbose=2) 
    #record time
    toc = time.time()
    totTime = toc-tic
    #model.load_weights('../Models/best_weights_TL.hdf5')
    model.save('../Models/model_TL_'+str(exp_num)+'.h5')
    if plot:
        view_acc(history, str(exp_num))
    
    history = [x for x in train]
    # walk-forward validation over each timestep
    predictions = list()
    for i in range(len(test)):
        # predict the timestep
        yhat_sequence = forecast(model, history, n_steps, n_length, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next timestep
        history.append(test[i, :])
    # evaluate predictions days for each timestep
    predictions = array(predictions)
    test1 = test[:, :, -1]
    YPred = argmax(predictions.reshape(predictions.shape[0], (predictions.shape[1]*predictions.shape[2])), out=None, axis=1)
    YPred = YPred.reshape(YPred.shape[0], 1)
    df = stack((YPred, test1))
    df = df.transpose()
    df = df.reshape(df.shape[1], 2)
    DataFrame(df).to_csv('../Results/Files/2DConvLSTMAE_TL_'+str(exp_num)+'.csv')
    cm = confusion_matrix(test1, YPred)
    print(cm)
    f1 = f1_score(test1, YPred, average='weighted')
    acc = accuracy_score(test1, YPred)
    print(classification_report(test1, YPred))     
    return f1, acc, totTime


