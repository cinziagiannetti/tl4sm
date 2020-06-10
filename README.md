==========================================
Transfer Learning for Smart Manufacturing
==========================================

# TL4SM

**A Python package for performing transfer learning between multivariate time series using data from various similar, but not necessarily related sources. **

The baseline model is a ConvLSTM2D auto-encoder, which comprises two stacked layers of ConvLSTM2D (128*4) and a bi-LSTM (200) layer. A softmax classification layer is included for multi-class classification, and two transfer learning strategies are added functions within the module. First, model weight re-using, whereby the weights from a source model are transferred to a target model. Second, the fine-tuning function can also be used. 

# Installation

The package can be installed from PyPI using the following code

`pip install tl4sm`

# How to use


