# Transfer Learning for Smart Manufacturing Timeseries (tl4sm)


`tl4sm` is a Python package for performing transfer learning between multivariate (continuous) time series using data from various similar, but not necessarily related sources. The package uses as baseline a ConvLSTM autoencoder model presented in our paper entitled "A Deep Learning Model for Smart Manufacturing Using Convolutional LSTM Neural Network Autoencoders", which can be found [here](https://ieeexplore.ieee.org/iel7/9424/9106618/08967003.pdf). As requirements, the base project directory should be setup according to the following folder structure. 

  - Code
  - Data
  - Results
  - Models

# Installation

  - Type `pip install tl4sm` to install the most recent version (0.19)
  - Python 3.7 is required.
  
# Usage
After installation, the module is called using `import tl4sm` to access the submodules therein. 

### Example 

Below is a practical example of tl4sm_generic for transfer learning using the six cities Beijing air quality data, which can be found [here](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data).

First, we import the required function from the `tl4sm` package, in this case, the `perform_experiment` function. We also import the `read_csv` function from `pandas` as well to read in our input experimental setup file. 

Next, we specify our data folder and get the list of all csv files inside there. The third step is where we specify the location where the models will be saved. For this project, it is inside the `Models` folder, which is specified by the `model_` variable. 

Finally, we provide the location of our input experimental setup csv file (i.e. `grid_search_baseline.csv`). This can be created manually and saved with any given name, but inside the `../Results/` folder. The csv should have the following column names (the order is not necessary). 


    - Exp. Number - `[str]` this specifies the experiment number
    - TL type - `[str]` this specifies the transfer learning type (`['None', 'fine-tune', 'reuse']`)
    - Layer - `[int]` specifies the number of layers to be retrained (from the last one)
    - Inputs - `[int]` specifies the lookback or number of previous timesteps to be used in the model training
    - Length - `[int]` specifies the number of sequence lengths of the ConvLSTM model
    - LR - `[float]` specifies the learning rate for the keras model optimizer
    - Epochs - `[int]` specifies the number of epochs
    - Batch - `[int]` specifies the batch size
    - Source - `[str]` specifies the filename of the source dataset
    - Target - `[str]` specifies the filename of the target dataset
    - Data Percent - `[str]` specifies the data percentage of the source dataset (0.1 means 10% of the data, etc.)
    - F1_Score - `[float]` specifies the f-score of the result
    - Accuracy_Score - `[float]` specifies the accuracy score of the result
    - Train Time - `[float]` specifies the model training time 
    
See the code below.


```sh
from tl4sm import perform_experiment as pf
from pandas import read_csv
import os


file_name = '../Data/'
files = [f for f in os.listdir('../Data/')]
model_ = '../Models/model_'
resFile='../Results/grid_search_baseline.csv'

n_test = 100
n_out=10
verbose=2

pf.perform_experiment(resFile, file_name, n_test, model_, n_out, verbose, med=40, high=100)
```

After this, we specify the model training parameters. The `n_test` variable represents the number of training rows that will be reserved for the model validation. It can either be passed as an absolute value (e.g. 10,000 records) or as a percentage of the dataset (i.e. 0.2 for 20% of the data).

The `n_out` variable represents the number of predictive future steps that is required. The `verbose` parameter specifies the verbosity of the model training (similar to verbose in `keras`).

In the final step, we call the `perform_experiment` function accordingly


    - resFile - `[str]` this specifies the location of the input experimental setup csv.
    - file_name - `[str]` this specifies the location that contains the input datasets (i.e. `../Data/`)
    - n_test - `[int or float]` specifies the number of rows to be set aside for model validation/testing. It can either be passed as an absolute value (e.g. 10,000 records) or as a percentage of the dataset (i.e. 0.2 for 20% of the data).    
    - model_ - `[str]` specifies the location that will contain the saved models (i.e. `../Models/`)
    - n_out - `[int]` specifies the number of predictive future steps that is required
    - verbose - `[int]` specifies the verbosity of the model training (similar to verbose in `keras`)
    - med - `[int]` specifies the medium class of the binned dataset.
    - high - `[int]` specifies the high class of the binned dataset.    
