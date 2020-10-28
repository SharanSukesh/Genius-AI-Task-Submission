# Genius-AI-Task-Submission

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Libraries used](#libraries-used)
* [Dataset used](#dataset-used)
* [Built on](#built-on)
* [Ackowledgements](#ackowledgements)
* [Author](#author)


## About the Project 
This project was done as part of an intern task assignment by Genius AI for their Machine Learning Internship. It begins with first analyzing the dataset and getting insights about patterns within it. Next we explore the quality of our data and do preprocessing in order to clean it.

After this, we then go into feature engineering in which we manipulate the features so that our models will be able to understand them better. As part of this process, we also add a couple new features that could add value to our model. On top of this, we drop those that do not help us and those with high correlation to each other to avoid multicolinearity.

Lastly, we look at a few different classification models on our data and see how well they fare. We then take the best few and examine the SHAP values to gain an understanding of how our model is making the predictions that it does. We then select the best model and make a pickle file from it so that it can be used in a webapp at a later time.

## Libraries used 
* Numpy
* sklearn
* shap
* tensorflow
* Matplotlib
* xgboost
* pandas
* matplotlib
* pickle

```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt                                    # Plotting library for Python programming language and it's numerical mathematics extension NumPy
import seaborn as sns                                              # Provides a high level interface for drawing attractive and informative statistical graphics
%matplotlib inline
import shap
shap.initjs()
from sklearn.utils import resample
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBClassifier
```

## Dataset used 
* __Customer Information Database__ - Genius AI

## Built with
* Jupyter Notebook

## Author - Sharan Sukesh



