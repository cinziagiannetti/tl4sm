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

`from tl4sm import perform_experiment`
`#perform the TL experiment using the experimental settings from the spreadsheet
perform_experiment(resFile, file_name, n_test, model_, n_out, verbose)`

For more details about the individual helper functions and the entire project, run the jupyter notebook here

