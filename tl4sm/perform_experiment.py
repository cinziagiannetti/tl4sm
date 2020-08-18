from tl4sm.model_prep import bin_data, evaluate_model_reuse, evaluate_model_tl, evaluate_model
from tl4sm.prepare_data import split_dataset
from numpy import array, stack
from pandas import read_csv, DataFrame
from pathlib import Path
from sklearn.preprocessing import LabelEncoder



def perform_experiment(resFile, file_name, n_test, model_, n_out, verbose, med, high):
    #load experimental config from csv file
    df_exp = read_csv(resFile, header=0)
    #create result dataframe lists
    f1_list, train_time, acc_list = list(), list(), list()
    #iterate through rows and take parameters specified
    for index, row in df_exp.iterrows():
        #Experiment Number
        exp_num = row['Exp. Number']
        #Layer
        n_layer = int(row['Layer'])
        #Lookback
        n_input = int(row['Inputs'])
        #Epochs
        epochs = int(row['Epochs'])
        #Batch Size
        batch_size= int(row['Batch'])
        #Source
        source = row['Source']
        #Target
        target = row['Target']
        #Length
        n_length = int(row['Length'])
        #TL type
        tl_type = str(row['TL type'])
        #Data Percent
        data_percent = float(row['Data Percent'])
        #Learning Rate
        lr = float(row['LR'])
        #Source Model Name
        model_name = model_+str(source)+'.h5'        
        # load the new file
        dataset = read_csv(file_name+str(target)+'.csv', header=0, index_col=0)
        #fill NANs
        dataset = dataset.fillna(method='ffill')
        #specify target
        dataset['y'] = dataset['PM2.5']
        dataset.drop(['PM2.5', 'station'], axis=1, inplace=True)
        #bin data
        bin_data(dataset, 'y', med, high)
        #convert categorical to label
        labelEncoder = LabelEncoder()
        labelEncoder.fit(dataset['wd'])
        dataset['wd'] = labelEncoder.transform(dataset['wd'])        
        # split into train and test
        train, test = split_dataset(dataset.values, n_test)        
        #run experiments
        #check if model exists then do TL
        if Path(model_+str(source)+'.h5').is_file():
            print('============= Model Training with TL ===============')
            print('Source', source, 'Target', target)
            print('Data Percent', data_percent)
            print('_____________________________________________________')
            #evaluate model on TL if model does not exist
            if tl_type == 'reuse':
                print('Performing TL')
                score, accuracy, trainTime = evaluate_model_reuse(train, test, n_input, n_length, batch_size, lr, source, exp_num, epochs, model_name, data_percent, n_out, batch_norm=False, plot=False)
            elif tl_type == 'fine-tune':
                print('Fine-tuning')
                score, accuracy, trainTime = evaluate_model_tl(train, test, n_input, n_length, batch_size, lr, source, exp_num, epochs, model_name, data_percent, n_layer, n_out)
            elif tl_type == 'None':
                print('============= Model Training without TL ===============')
                print('Source', source, 'Target', target)
                print('Data Percent', data_percent)
                score, accuracy, trainTime, _, _, _ = evaluate_model(train, test, n_input, n_length, batch_size, lr, source, exp_num, epochs, n_out, data_percent, verbose)
        else:
            print('============= Model Training without TL ===============')
            print('Source', source, 'Target', target)
            print('Data Percent', data_percent)
            score, accuracy, trainTime, f1_base, acc_base, tr_time_base = evaluate_model(train, test, n_input, n_length, batch_size, lr, source, exp_num, epochs, n_out, data_percent, verbose, batch_norm=False)
        #append scores
        f1_list.append(array(score).reshape(1,))
        acc_list.append(array(accuracy).reshape(1,))
        train_time.append(array(trainTime).reshape(1,))
        df_res = stack((f1_list, acc_list, train_time)).transpose().reshape(len(train_time),3)
        df_res = DataFrame(df_res, columns = ['F1_Score', 'Accuracy_Score', 'Train Time'])
        #write to original experiment config
        df_exp['F1_Score'], df_exp['Accuracy_Score'], df_exp['Train Time'] = df_res['F1_Score'], df_res['Accuracy_Score'], df_res['Train Time']
        #output to experiment result file
        df_exp.to_csv(resFile, index=False)
       
