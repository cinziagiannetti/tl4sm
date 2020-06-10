# Transfer Learning for Smart Manufacturing

A Python package for performing transfer learning between multivariate time series using data from various similar, but not necessarily related sources

The baseline model is a ConvLSTM2D auto-encoder, which comprises two stacked layers of ConvLSTM2D (128*4) and a bi-LSTM (200) layer. A softmax classification layer is included for multi-class classification, and two transfer learning strategies are added functions within the module. First, model weight re-using, whereby the weights from a source model are transferred to a target model. Second, the fine-tuning function can also be used. 

# Installation

The package can be installed from PyPI using the following code

`$ pip install tl4sm`

# Folder Structure
# Files and Directories
1. MatlabAnalysis
2. Python
   * Code
      * BM_TL.py: wrapper script for TL experiments
      * bm_tl.ipynb: Jupyter notebook serving as documentation for TL experiment
      * model_prep: helper utility functions
      * evaluate_forecast: script to use pre-trained model to predict and evaluate
      * activation.py: script to extract activations from pre-trained models
      * compute_dtw.py: script to compute distance between source and target datasets using various distance measures, including Wassertain, DTW, euclidean, etc.
      * prepare_data.py: helper functions to load and pre-process data
      * load_all_speed.py: script to load all speed data and stack column-wise.
      * box_plots.py: script to plot data distribution of datasets
      * errors.py: helper function to compute RMSE for each dataset, given a pre-trained model
      * plots.py: function to plot results
   * Data
   * Models
   * Plots
   * Results
     * Files

# How to use
This package was designed with ease-of-use in mind. Performing a TL experiment couldn't be easier. The only thing to do to get started with this project is setup the folders according to the structure above. After this, simply run the code below. Some parameters can be played around, and all the experimental parameters are included in the xx_experiment_1.csv spreadsheet.

`from tl4sm import perform_experiment` <br>
`perform_experiment(resFile, file_name, n_test, model_, n_out, verbose)`

# Module Parameters

## `resFile (str)`: 
csv file containing experimental parameters with the below structure <br>
|Exp. Number |	TL type |	Layer	| Inputs	| Length |	LR	| Epochs |	Batch |	Source |	Target |	Data Percent|

### `Exp. Number (str)`:
Unique identifier for each experiment (row) to be conducted. For example, E1, E_TL_100, etc.

### `TL type (str) ['None', 'reuse', 'fine-tune']`:
specifies what type of transfer learning strategy to apply. If set to `None`, then no transfer learning is performed and a baseline model is generated. `Reuse` takes the weights from a source model and applies the same to a target model. `Fine-tune` freezes at least one layer from the source model and fine-tunes the remaining layers on the target domain.

### `Layer (int)`:
specifies the number of layers to fine-tune in the `fine-tune` transfer learning strategy. For instance, if `Layer=1`, then the transfer learning regime/experiment will fine-tune the last layer of the source model on the target domain. For `TL type = ['None', 'reuse']`, the `Layer` parameter is not required, but should not be left blank.

### `Inputs (int)`:
specifies the number of lookback steps in the time series to be used as input features, using a sliding window method. For instance, if `Inputs=60`, then the previous `60` time steps are transformed to be used as inputs. For more details, see the Jupyter notebook <a href="https://github.com/nakessien/BM_TL_classification/blob/master/Python/Code/bm_tl.ipynb" target="_blank">here</a> 

### `Length (int)`:
specifies the length of each subsequence in the ConvLSTM2D layer. For more details about this hyperparameter, see the original paper <a href="https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf" target="_blank">here</a> 

### `LR (float)`:
specifies the learning rate for the Adam optimizer

### `Epochs (int)`:
specifies the number of epochs or iterations each model will be trained

### `Batch (int)`:
specifies the batch size for each iteration

### `Source (int)`:
specifies the source data identifier

### `Target (int)`:
specifies the target data identifier

### `Data Percent (float)`:
specifies the percentage fraction of the target training dataset to be applied in the experiment session. N/B: This should be represented as a fraction. For instance, for 50% training data, this parameter should be 0.5


## `file_name (str)`:
It is assumed that all the datasets will have similar/common naming conventions. This parameter represents the common filename for each data source. For instance, `'../Data/transf_exp_/'`. This will typically be in the `Data` folder. See folder structure section above.

## `n_test (int or float)`:
specifies the size (or number of observations) of the test dataset. If `float` provided, then it must be specified as a fraction (i.e. `> 0 and <=1`), and if real number provided, then this represents the number of observations (i.e. rows) that are reserved as the test dataset.

## `model_ (str)`:
specifies the location where the models will be stored/retrieved. Typically, according to the folder structure, this will be `'../Models/model_'`.

## `n_out (int)`:
specifies the number of time steps that will be predicted by the model. For single-step, then this value will be set to 1.

## `verbose (int) [0,1,2]`:
specifies if the model will be required to provide information about each training epoch, same as the verbose parameter in a keras model.


For more details about the individual helper functions and the entire project, run the jupyter notebook <a href="https://github.com/nakessien/BM_TL_classification/blob/master/Python/Code/bm_tl.ipynb" target="_blank">here</a>

