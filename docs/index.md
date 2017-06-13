Welcome to MLBox's documentation!
======================================

This page is the official documentation for MLBox package. You will learn how to use all the features provided by this tool. 


### Table of Contents

* **[encoding](#encoding)**<br>

* **[model](#model)**<br>
  * [classification](#classification) <br>
  * [regression](#regression) <br>

* **[optimisation](#optimisation)**<br>

* **[prediction](#prediction)**<br>

* **[preprocessing](#preprocessing)**<br>


# Documentation

## encoding

```python


class Optimiser

 | 
Optimises hyper-parameters of the whole Pipeline :

 |  

 |  1/ NA
encoder (missing values encoder)

 |  2/ CA encoder
(categorical features encoder)

 | 
3/ Feature selector [OPTIONAL]

 | 
4/ Stacking estimator - feature engineer [OPTIONAL] 

 | 
5/ Estimator (classifier or regressor)

 | 


 | 
Works for both regression and classification (multiclass or binary)
tasks.

 | 


 | 


 | 
Parameters

 | 
----------

 | 


 | 
scoring : string, callable or None, optional, default: None

 |      A string (see model evaluation
documentation) or a scorer callable object / function with
signature``scorer(estimator, X, y)``.

 | 


 |      If None, "log_loss" is used for
classification and "mean_squarred_error" for regression

 |      Available scorings for classification :
"accuracy","roc_auc", "f1", "log_loss",
"precision", "recall"

 |      Available scorings for regression :
"mean_absolute_error", "mean_squared_error",
"median_absolute_error", "r2"

 | 


 | 
n_folds : int, defaut = 2

 |      The number of folds for cross validation
(stratified for classification)

 | 


 | 
random_state : int, defaut = 1

 |      pseudo-random number generator state used
for shuffling

 | 


 | 
to_path : str, defaut = "save"

 |      Name of the folder where models are saved

 |  

 | 
verbose : bool, defaut = True

 |     
Verbose mode

 |  

 | 
Methods defined here:

 | 


 | 
__init__(self, scoring=None, n_folds=2, random_state=1, to_path='save',
verbose=True)

 | 


 | 
evaluate(self, params, df)

 |      Evaluates the scoring function with given
hyper-parameters of the whole Pipeline. If no parameters are set, defaut
configuration for each step is evaluated : no feature selection is applied and
no meta features are created.

 |      

 |      

 |      Parameters

 |      ----------

 |      

 |      params : dict, defaut = None.

 |          Hyper-parameters dictionnary for the
whole pipeline. If params = None, defaut configuration is evaluated.

 |      

 |          - The keys must respect the following
syntax : "enc__param".

 |      

 |          With :

 |              1/ "enc" =
"ne" for na encoder

 |              2/ "enc" =
"ce" for categorical encoder

 |              3/ "enc" =
"fs" for feature selector [OPTIONAL]

 |              4/ "enc" =
"stck"+str(i) to add layer n°i of meta-features (assuming 1 ... i-1
layers are created...) [OPTIONAL]

 |              5/ "enc" =
"est" for the final estimator

 |      

 |          And:

 |              "param" : a correct
associated parameter for each step. (for example : "max_depth" for
"enc"="est", "entity_embedding" for
"enc"="ce")

 |      

 |          - The values are those of the
parameters (for example : 4 for a key = "est__max_depth")

 |      

 |      

 |     
df : dict, defaut = None

 |         
Train dictionnary. Must contain keys "train" and
"target" with the train dataset (pandas DataFrame) and the associated
target (pandas Serie with

 |          dtype='float' for a regression or
dtype='int' for a classification) resp.

 |      

 | 
    

 |      Returns

 |      -------

 |      

 |      score : float.

 |          The score. The higher the better
(positive for a score and negative for a loss)

 | 


 | 
get_params(self, deep=True)

 | 


 | 
optimise(self, space, df, max_evals=40)

 |      Optimises hyper-parameters of the whole
Pipeline with a given scoring function. By defaut, estimator used is 'xgboost'
and no feature selection is applied.

 |      Algorithm used to optimize : Tree Parzen
Estimator (http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html)

 |      IMPORTANT : Try to avoid dependent
parameters and to set one feature selection strategy and one estimator strategy
at a time.

 |      

 |      Parameters

 |      ----------

 |      

 |      space : dict, defaut = None.

 |          Hyper-parameters space.

 |      

 |          - The keys must respect the following
syntax : "enc__param".

 |      

 |          With :

 |              1/ "enc" =
"ne" for na encoder

 |              2/ "enc" =
"ce" for categorical encoder

 |              3/ "enc" =
"fs" for feature selector [OPTIONAL]

 |              4/ "enc" =
"stck"+str(i) to add layer n°i of meta-features (assuming 1 ... i-1
layers are created...) [OPTIONAL]

 |              5/ "enc" =
"est" for the final estimator

 |      

 |          And:

 |              "param" : a correct
associated parameter for each step. (for example : "max_depth" for
"enc"="est", "entity_embedding" for
"enc"="ce")

 |      

 |          - The values must respect the
following syntax : {"search" : strategy, "space" : list}

 |      

 |          With:

 |              "strategy" =
"choice" or "uniform". Defaut = "choice"

 |      

 |          And:

 |             list : a list of values to be
tested if strategy="choice". If strategy = "uniform", list =
[value_min, value_max].

 |      

 |      

 |      df : dict, defaut = None

 |          Train dictionnary. Must contain keys
"train" and "target" with the train dataset (pandas
DataFrame) and the associated target (pandas Serie with

 |          dtype='float' for a regression or
dtype='int' for a classification) resp.

 |      

 |      

 |      max_evals : int, defaut = 40.

 |          Number of iterations. For an accurate
optimal hyper-parameter, max_evals = 40.

 |      

 |      

 |      Returns

 |      -------

 |      

 |      best_params : dict.

 |          The optimal hyper-parameter
dictionnary.

 | 


 | 
set_params(self, **params)


```

## model

### classification

### regression


## optimisation

<br/>
class Optimiser <br/>
 | Optimises hyper-parameters of the whole Pipeline :<br/>
 | <br/> 
 | 1/ NA encoder (missing values encoder) <br/>
 | 2/ CA encoder (categorical features encoder) <br/>
 | 3/ Feature selector [OPTIONAL] <br/>
 | 4/ Stacking estimator - feature engineer [OPTIONAL] <br/>
 | 5/ Estimator (classifier or regressor) <br/>
 | <br/>
 | Works for both regression and classification (multiclass or binary) tasks. <br/>
 | <br/>
 | Parameters<br/>
 | ----------<br/>
 | <br/>
 | scoring : string, callable or None, optional, default: None <br/>
 |      A string (see model evaluation documentation) or a scorer callable object / function with signature <br/>
 |      ``scorer(estimator, X, y)``. <br/>
 | <br/>
 |      If None, "log_loss" is used for classification and "mean_squarred_error" for regression <br/>
 |      Available scorings for classification : "accuracy","roc_auc", "f1", "log_loss", "precision", "recall" <br/>
 |      Available scorings for regression : "mean_absolute_error", "mean_squared_error", "median_absolute_error", "r2" <br/>
 | <br/>
 | n_folds : int, defaut = 2 <br/>
 |      The number of folds for cross validation (stratified for classification) <br/>
 | <br/>
 | random_state : int, defaut = 1 <br/>
 |      pseudo-random number generator state used for shuffling <br/>
 | <br/>
 | to_path : str, defaut = "save" <br/>
 |      Name of the folder where models are saved <br/>
 |  <br/>
 | verbose : bool, defaut = True <br/>
 |     Verbose mode <br/>
 |  <br/>
 | Methods defined here:<br/>
 | <br/>
 | __init__(self, scoring=None, n_folds=2, random_state=1, to_path='save', verbose=True)<br/>
 | <br/>
 | evaluate(self, params, df) <br/>
 |      Evaluates the scoring function with given hyper-parameters of the whole Pipeline. If no parameters are set, <br/>
 |      defaut configuration for each step is evaluated : no feature selection is applied and no meta features are created.<br/>
 |      <br/>
 |      Parameters <br/>
 |      ---------- <br/>
 |      <br/>
 |      params : dict, defaut = None. <br/>
 |          Hyper-parameters dictionnary for the whole pipeline. If params = None, defaut configuration is evaluated. <br/>
 |      <br/>
 |          - The keys must respect the following syntax : "enc__param". <br/>
 |      <br/>
 |          With : <br/>
 |              1/ "enc" = "ne" for na encoder <br/>
 |              2/ "enc" = "ce" for categorical encoder <br/>
 |              3/ "enc" = "fs" for feature selector [OPTIONAL] <br/>
 |              4/ "enc" = "stck"+str(i) to add layer n°i of meta-features (assuming 1 ... i-1 layers are created...) [OPTIONAL] <br/>
 |              5/ "enc" = "est" for the final estimator <br/>
 |      <br/>
 |          And:<br/>
 |              "param" : a correct associated parameter for each step. (for example : "max_depth" for "enc"="est", <br/>
 |              "entity_embedding" for "enc"="ce")<br/>
 | <br/>
 |          - The values are those of the parameters (for example : 4 for a key = "est__max_depth") <br/>
 |      <br/>
 |     df : dict, defaut = None <br/>
 |         Train dictionnary. Must contain keys "train" and "target" with the train dataset (pandas DataFrame) and the <br/>
 |         associated target (pandas Serie with dtype='float' for a regression or dtype='int' for a classification) resp. <br/>     
 | <br/> 
 |      Returns<br/>
 |      -------<br/>
 |      <br/>
 |      score : float.<br/>
 |          The score. The higher the better (positive for a score and negative for a loss)<br/>
 | <br/>
 | get_params(self, deep=True)<br/>
 | <br/>
 | optimise(self, space, df, max_evals=40)<br/> 
 |      Optimises hyper-parameters of the whole Pipeline with a given scoring function. By defaut, estimator used is 'xgboost' <br/> 
 |      and no feature selection is applied. Algorithm used to optimize : Tree Parzen Estimator. <br/> 
 |      IMPORTANT : Try to avoid dependent parameters and to set one feature selection strategy and one estimator strategy <br/>
 |      at a time.<br/>
 |      <br/>
 |      Parameters<br/>
 |      ----------<br/>
 |      <br/>
 |      space : dict, defaut = None.<br/>
 |          Hyper-parameters space.<br/>
 |      <br/>
 |          - The keys must respect the following syntax : "enc__param".<br/>
 |      <br/>
 |          With :<br/>
 |              1/ "enc" = "ne" for na encoder<br/>
 |              2/ "enc" = "ce" for categorical encoder<br/>
 |              3/ "enc" = "fs" for feature selector [OPTIONAL]<br/>
 |              4/ "enc" = "stck"+str(i) to add layer n°i of meta-features (assuming 1 ... i-1 layers are created...) [OPTIONAL]<br/>
 |              5/ "enc" = "est" for the final estimator<br/>
 |      <br/>
 |          And:<br/>
 |              "param" : a correct associated parameter for each step. (for example : "max_depth" for "enc"="est", <br/>
 |              "entity_embedding" for "enc"="ce")<br/>
 |<br/>      
 |          - The values must respect the following syntax : {"search" : strategy, "space" : list}<br/>
 |      <br/>
 |          With:<br/>
 |              "strategy" = "choice" or "uniform". Defaut = "choice"<br/>
 |      <br/>
 |          And:<br/>
 |             list : a list of values to be tested if strategy="choice". If strategy = "uniform", list = [value_min, value_max].<br/>
 |      <br/>
 |      df : dict, defaut = None<br/>
 |          Train dictionnary. Must contain keys "train" and "target" with the train dataset (pandas DataFrame) and the <br/>
 |          associated target (pandas Serie with dtype='float' for a regression or dtype='int' for a classification) resp. <br/>
 |      <br/>
 |      max_evals : int, defaut = 40.<br/>
 |          Number of iterations. For an accurate optimal hyper-parameter, max_evals = 40.<br/>
 |<br/>      
 |      Returns<br/>
 |      -------<br/>
 |      <br/>
 |      best_params : dict.<br/>
 |          The optimal hyper-parameter dictionnary.<br/>
 | <br/>
 | set_params(self, **params)<br/>
<br/>
<br/>

## prediction
<br/>
class Predictor <br/> 
|  Predicts the target on the test dataset. <br/>
| <br/>
|  Parameters <br/>
|  ---------- <br/> 
| <br/> 
|  to_path : str, defaut = "save" <br/>
|      Name of the folder where the feature importances and predictions are saved (.png and .csv format). Must contain target <br/> 
|      encoder object (for classification task only). <br/> 
| <br/> 
|  verbose : bool, defaut = True <br/> 
|      Verbose mode <br/>
|  <br/>
|  Methods defined here: <br/>
|  <br/>
|  __init__(self, to_path='save', verbose=True) <br/>
|  <br/>
|  plot_feature_importances(self, importance, fig_name = "feature_importance.png") <br/>
|      Saves feature importances plot <br/>
|  <br/>
|      Parameters <br/>
|      ---------- <br/>
|  <br/>
|      importance : dict <br/>
|          dictionnary with features (key) and importances (values) <br/>
|  <br/>
|      fig_name : str, defaut = "feature_importance.png" <br/>
|          figure name <br/>
|  <br/>
|      Returns <br/>
|      ------- <br/>
|  <br/>
|      None <br/>
|  <br/>
|  fit_predict(self, params, df) <br/>
|      Fits the model. Then predicts on test dataset and outputs feature importances and the submission file (.png and .csv <br/>
|      format). <br/>
|  <br/>
|      Parameters <br/>
|      ---------- <br/>
|  <br/>
|      params : dict, defaut = None. <br/>
|          Hyper-parameters dictionnary for the whole pipeline. If params = None, defaut configuration is evaluated. <br/>
|  <br/>
|          - The keys must respect the following syntax : "enc__param". <br/>
|  <br/>
|          With : <br/>
|              1/ "enc" = "ne" for na encoder <br/>
|              2/ "enc" = "ce" for categorical encoder <br/>
|              3/ "enc" = "fs" for feature selector [OPTIONAL] <br/>
|              4/ "enc" = "stck"+str(i) to add layer n°i of meta-features (assuming 1 ... i-1 layers are created...) [OPTIONAL] <br/>
|              5/ "enc" = "est" for the final estimator <br/>
|  <br/>
|          And: <br/>
|              "param" : a correct associated parameter for each step. (for example : "max_depth" for "enc"="est", <br/>
|              "entity_embedding" for "enc"="ce") <br/>
|  <br/> 
|          - The values are those of the parameters (for example : 4 for a key = "est__max_depth") <br/>
|  <br/>
|      df : dict, defaut = None <br/>
|          Dataset dictionnary. Must contain keys "train","test" and "target" with the train dataset (pandas DataFrame), the test <br/>
|          dataset (pandas DataFrame) and the associated target (pandas Serie with dtype='float' for a regression or dtype='int' <br/>
|          for a classification) resp. <br/>
|  <br/> 
|      Returns <br/>
|      -------- <br/>
|  <br/>
|      None <br/>
|  <br/>
|  get_params(self, deep=True) <br/>
|  <br/>
|  set_params(self, params) <br/>
<br/>
<br/> 

## preprocessing
<br/>
class Drift_thresholder <br/>
|  Automatically deletes ids and drifting variables between train and test datasets. <br/>
|  Deletes on train and test datasets. The list of drift coefficients is available and saved as "drifts.txt" <br/>
| <br/>
|  Parameters <br/>
|  ---------- <br/>
| <br/>
|  threshold : float (between 0.5 and 1.), defaut = 0.9 <br/>
|      Threshold used to deletes variables and ids. The lower the more you keep non-drifting/stable variables. <br/>
| <br/>
|  inplace : bool, defaut = False <br/>
|      If True, train and test datasets are transformed. Returns self. <br/>
|      Otherwise, train and test datasets are not transformed. Returns a new dictionnary with cleaned datasets. <br/>
| <br/>
|  verbose : bool, defaut = True <br/>
|      Verbose mode <br/>
| <br/>
|  to_path : str, defaut = "save" <br/>
|      Name of the folder where the list of drift coefficients is saved <br/>
| <br/>
|  Methods defined here: <br/>
| <br/>
|  __init__(self, threshold=0.8, inplace=False, verbose=True, to_path='save') <br/>
| <br/>
|  drifts(self) <br/>
|      Returns the univariate drifts for all variables. <br/>
| <br/>
|  fit_transform(self, df) <br/>
|      Automatically deletes ids and drifting variables between train and test datasets. <br/>
|      Deletes on train and test datasets. The list of drift coefficients is available and saved as "drifts.txt" <br/>
| <br/>
|      Parameters <br/>
|      ---------- <br/>
| <br/>
|      df : dict, defaut = None <br/>
|          Dictionnary containing : <br/>
|          'train' : pandas dataframe for train dataset <br/>
|          'test' : pandas dataframe for test dataset <br/>
|          'target' : pandas serie for the target <br/>
| <br/>
|      Returns <br/>
|      ------- <br/>
| <br/>
|      df : dict <br/>
|          Dictionnary containing : <br/>
|          'train' : pandas dataframe for train dataset <br/>
|          'test' : pandas dataframe for test dataset <br/>
|          'target' : pandas serie for the target <br/>
<br/>
<br/>
class Reader <br/>
|  Reads and cleans data <br/>
| <br/>
|  Parameters <br/>
|  ---------- <br/>
| <br/>
|  sep : str, defaut = None <br/>
|       Delimiter to use when reading a csv file. <br/>
| <br/>
|  header : int or None, defaut = 0. <br/>
|      If header=0, the first line is considered as a header. <br/>
|      Otherwise, there is no header. <br/>
|      Useful for csv and xls files. <br/>
| <br/>
|  to_hdf5 : bool, defaut = True <br/>
|      If True, dumps each file to hdf5 format <br/>
| <br/>
|  to_path : str, defaut = "save" <br/>
|      Name of the folder where files and encoders are saved <br/>
| <br/>
|  verbose : bool, defaut = True <br/>
|      Verbose mode <br/>
| <br/>
|  Methods defined here: <br/>
| <br/>
|  __init__(self, sep=None, header=0, to_hdf5=False, to_path='save', verbose=True) <br/>
| <br/>
|  clean(self, path, date_strategy, drop_duplicate) <br/>
|      Reads and cleans data (accepted formats : csv, xls, json and h5) : <br/>
| <br/>
|      - del Unnamed columns <br/>
|      - casts lists into variables <br/>
|      - try to cast variables into float <br/>
|      - cleans dates <br/>
|      - drop duplicates (if drop_duplicate=True) <br/>
| <br/>
|      Parameters <br/>
|      ---------- <br/>
| <br/>
|      filepath_or_buffer : str, pathlib.Path <br/>
|          The string could be a URL. <br/>
| <br/>
|      date_strategy : str, defaut = "complete" <br/>
|          The strategy to encode dates : <br/>
|          - complete : creates timestamp from 01/01/2017, month, day and day_of_week <br/>
|          - to_timestamp : creates timestamp from 01/01/2017 <br/>
| <br/>
|      drop_duplicate : bool, defaut = False <br/>
|          If True, drop duplicates when reading each file. <br/>
| <br/>
|      Returns <br/>
|      ------- <br/>
| <br/>
|      df : pandas dataframe <br/>
| <br/>
|  train_test_split(self, Lpath, target_name) <br/>
|      Given a list of several paths and a target name, automatically creates and cleans train and test datasets. <br/>
|      Also determines the task and encodes the target (classification problem only). <br/>
|      Finally dumps the datasets to hdf5, and eventually the target encoder. <br/>
| <br/>
|      Parameters <br/>
|      ---------- <br/>
| <br/>
|      Lpath : list, defaut = None <br/>
|          List of str paths to load the data <br/>
| <br/>
|      target_name : str, defaut = None <br/>
|          The name of the target. Works for both classification (multiclass or not) and regression. <br/>
| <br/>
|      Returns <br/>
|      ------- <br/>
| <br/>
|      df : dict <br/>
|          Dictionnary containing : <br/>
|          'train' : pandas dataframe for train dataset <br/>
|          'test' : pandas dataframe for test dataset <br/>
|          'target' : pandas serie for the target <br/>
