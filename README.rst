.. image:: docs/logos/logo.png

|Documentation Status| |PyPI version| |Build Status| |Windows Build Status| |GitHub Issues| |codecov| |License|

-----------------------

**MLBox is a powerful Automated Machine Learning python library.** It provides the following features:


* Fast reading and distributed data preprocessing/cleaning/formatting
* Highly robust feature selection and leak detection
* Accurate hyper-parameter optimization in high-dimensional space
* State-of-the art predictive models for classification and regression (Deep Learning, Stacking, LightGBM,...)
* Prediction with models interpretation 


**For more details**, please refer to the `official documentation <https://mlbox.readthedocs.io/en/latest/>`__

--------------------------


Getting started: 30 seconds to MLBox
====================================

MLBox main package contains 3 sub-packages : **preprocessing**, **optimisation** and **prediction**. Each one of them are respectively aimed at reading and preprocessing data, testing or optimising a wide range of learners and predicting the target on a test dataset.

**Here are a few lines to import the MLBox:**

.. code-block:: python 

   from mlbox.preprocessing import *
   from mlbox.optimisation import *
   from mlbox.prediction import *


**Then, all you need to give is :** 

* the list of paths to your train datasets and test datasets
* the name of the target you try to predict (classification or regression)

.. code-block:: python 

   paths = ["<file_1>.csv", "<file_2>.csv", ..., "<file_n>.csv"] #to modify
   target_name = "<my_target>" #to modify


**Now, let the MLBox do the job !**

... to read and preprocess your files : 

.. code-block:: python 

   data = Reader(sep=",").train_test_split(paths, target_name)  #reading
   data = Drift_thresholder().fit_transform(data)  #deleting non-stable variables

... to evaluate models (here default configuration):

.. code-block:: python 

   Optimiser().evaluate(None, data)


... or to test and optimize the whole Pipeline [**OPTIONAL**]:

* missing data encoder, aka 'ne'
* categorical variables encoder, aka 'ce'
* feature selector, aka 'fs'
* meta-features stacker, aka 'stck'
* final estimator, aka 'est'

**NB** : please have a look at all the possibilities you have to configure the Pipeline (steps, parameters and values...) 

.. code-block:: python 

   space = {
   
           'ne__numerical_strategy' : {"space" : [0, 'mean']},

           'ce__strategy' : {"space" : ["label_encoding", "random_projection", "entity_embedding"]},

           'fs__strategy' : {"space" : ["variance", "rf_feature_importance"]},
           'fs__threshold': {"search" : "choice", "space" : [0.1, 0.2, 0.3]},             

           'est__strategy' : {"space" : ["LightGBM"]},
           'est__max_depth' : {"search" : "choice", "space" : [5,6]},
           'est__subsample' : {"search" : "uniform", "space" : [0.6,0.9]}
           
           }

   best = opt.optimise(space, data, max_evals = 5)

... finally to predict on the test set with the best parameters (or None for default configuration):

.. code-block:: python 

   Predictor().fit_predict(best, data)


**That's all !** You can have a look at the folder "save" where you can find :

* your predictions
* feature importances
* drift coefficients of your variables (0.5 = very stable, 1. = not stable at all)

--------------------------

How to Contribute
=================

MLBox has been developed and used by many active community members. Your help is very valuable to make it better for everyone.

- Check out `call for contributions <https://github.com/AxeldeRomblay/MLBox/labels/call-for-contributions>`__ to see what can be improved, or open an issue if you want something.
- Contribute to the `tests <https://github.com/AxeldeRomblay/MLBox/tree/master/tests>`__ to make it more reliable. 
- Contribute to the `documents <https://github.com/AxeldeRomblay/MLBox/tree/master/docs>`__ to make it clearer for everyone.
- Contribute to the `examples <https://github.com/AxeldeRomblay/MLBox/tree/master/examples>`__ to share your experience with other users.
- Open `issue <https://github.com/AxeldeRomblay/MLBox/issues>`__ if you met problems during development.

For more details, please refer to `CONTRIBUTING <https://github.com/AxeldeRomblay/MLBox/blob/master/docs/contributing.rst>`__.

.. |Documentation Status| image:: https://readthedocs.org/projects/mlbox/badge/?version=latest
   :target: https://mlbox.readthedocs.io/en/latest/
.. |PyPI version| image:: https://badge.fury.io/py/mlbox.svg
   :target: https://pypi.python.org/pypi/mlbox
.. |Build Status| image:: https://travis-ci.org/AxeldeRomblay/MLBox.svg?branch=master
   :target: https://travis-ci.org/AxeldeRomblay/MLBox
.. |Windows Build Status| image:: https://ci.appveyor.com/api/projects/status/5ypa8vaed6kpmli8?svg=true
   :target: https://ci.appveyor.com/project/AxeldeRomblay/mlbox
.. |GitHub Issues| image:: https://img.shields.io/github/issues/AxeldeRomblay/MLBox.svg
   :target: https://github.com/AxeldeRomblay/MLBox/issues
.. |codecov| image:: https://codecov.io/gh/AxeldeRomblay/MLBox/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/AxeldeRomblay/MLBox
.. |License| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://github.com/AxeldeRomblay/MLBox/blob/master/LICENSE
