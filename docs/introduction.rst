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
