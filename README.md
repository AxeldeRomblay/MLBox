MLBox, Machine Learning Box
===========================
[![Build Status](https://travis-ci.org/AxeldeRomblay/MLBox.svg?branch=master)](https://travis-ci.org/AxeldeRomblay/MLBox)
[![GitHub
Issues](https://img.shields.io/github/issues/AxeldeRomblay/MLBox.svg)](https://github.com/AxeldeRomblay/MLBox/issues)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)]((https://img.shields.io/github/license/mashape/apistatus.svg))

__MLBox is a powerful Automated Machine Learning python library.__ It provides the following features:

- Fast reading and distributed data preprocessing/cleaning/formatting
- Highly robust feature selection and leak detection
- Accurate hyper-parameter optimization in high-dimensional space
- State-of-the art predictive models for classification and regression (Deep Learning, Stacking, LightGBM,...)
- Prediction with models interpretation 


__To get it installed__, please refer to [README](https://github.com/AxeldeRomblay/MLBox/blob/master/python-package/README.md)

__For more details__, please refer to [docs](https://github.com/AxeldeRomblay/MLBox/tree/master/docs/documentation.md)

__Experiments__ : https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/leaderboard | Rank : 85/2488

--------------------------


Getting started: 30 seconds to MLBox
====================================

MLBox main package is divided into 3 sub-packages : __preprocessing__, __optimisation__ and __prediction__. Each one of them are respectively aimed at reading and preprocessing data, testing and optimising a wide range of learners and predicting the target on a test dataset.

__Here are a few lines to import the MLBox:__

```python
from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *
```

__Then, all you need to give is :__ 
* the list of paths to your train datasets and test datasets
* the name of the target you try to predict (classification or regression)

```python
paths = ["<file_1>.csv", "<file_2>.csv", ..., "<file_n>.csv"] #to modify
target_name = "<my_target>" #to modify
```
__Now, let the MLBox do the job !__

... to read and preprocess your files : 

```python
data = Reader(sep=",").train_test_split(paths, target_name)  #reading
data = Drift_thresholder().fit_transform(data)  #deleting non-stable variables
```
... to evaluate models (here default configuration):

```python
Optimiser().evaluate(None, data)
```

... or to test and optimize the whole Pipeline [__OPTIONAL__]:
* missing data encoder, aka 'ne'
* categorical variables encoder, aka 'ce'
* feature selector, aka 'fs'
* meta-features stacker, aka 'stck'
* final estimator, aka 'est'

__NB__ : please have a look at all the possibilities you have to configure the Pipeline (steps, parameters and values...) 

```python
space = {
        'ne__numerical_strategy' : {"search":"choice", "space":[0, 'mean']},
                              
        'ce__strategy' : {"search":"choice", "space":["label_encoding", "random_projection"]},
                          
        'fs__strategy' : {"search":"choice", "space":["variance", "l1"]},
        'fs__threshold': {"search":"choice", "space":[0.1,0.2,0.3]},             
        
        'est__strategy' : {"search":"choice", "space":["XGBoost"]},
        'est__max_depth' : {"search":"choice", "space":[5,6]},
        'est__subsample' : {"search":"uniform", space":[0.6,0.9]}
        }
        
best = opt.optimise(space, data, max_evals = 5)
```
... finally to predict on the test set with the best parameters (or None for default configuration):

```python
Predictor().fit_predict(best, data)

```

__That's all !__ You can have a look at the folder "save" where you can find :
* your predictions
* feature importances
* drift coefficients of your variables (0.5 = very stable, 1. = not stable at all)

--------------------------


How to Contribute
=================

MLBox has been developed and used by many active community members. Your help is very valuable to make it better for everyone.

- Check out [call for contributions](https://github.com/AxeldeRomblay/MLBox/labels/call-for-contributions) to see what can be improved, or open an issue if you want something.
- Contribute to the [tests](https://github.com/AxeldeRomblay/MLBox/tree/master/tests) to make it more reliable. 
- Contribute to the [documents](https://github.com/AxeldeRomblay/MLBox/tree/master/docs) to make it clearer for everyone.
- Contribute to the [examples](https://github.com/AxeldeRomblay/MLBox/tree/master/examples) to share your experience with other users.
- Open issue if you met problems during development.

For more details, please refer to [CONTRIBUTING](https://github.com/AxeldeRomblay/MLBox/blob/master/CONTRIBUTING.rst).
