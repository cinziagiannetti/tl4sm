# Transfer Learning for Smart Manufacturing Timeseries (tl4sm)


`tl4sm` is a Python package for performing transfer learning between multivariate (continuous) time series using data from various similar, but not necessarily related sources. The package uses as baseline a ConvLSTM autoencoder model presented in our paper entitled "A Deep Learning Model for Smart Manufacturing Using Convolutional LSTM Neural Network Autoencoders", which can be found [here](https://ieeexplore.ieee.org/iel7/9424/9106618/08967003.pdf). As requirements, the base project directory should be setup according to the following folder structure. 

  - Code
  - Data
  - Results
  - Models

# Installation

  - Type `pip install tl4sm` to install the most recent version (0.19)
  
# Usage
After installation, the module is called using `import tl4sm` to access the submodules therein. 

### Example 

[Here](https://github.com/nakessien/tl4sm/blob/master/Sample/Code/tl4sm_tutorial.ipynb) is a practical example of tl4sm_generic for transfer learning using the six cities Beijing air quality data, which can be found [here](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data).



