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


# Documentation [WORK IN PROGRESS]

## encoding

<br/>
> class Categorical_encoder <br/>
>
 |  Encodes categorical features. Several strategies are possible (supervised or not). Works for both classification and regression tasks. <br/>
 |  <br/>
 |  <br/>
 |  Parameters <br/>
 |  ---------- <br/>
 |  <br/>
 |  strategy : string, defaut = "label_encoding" <br/>
 |      The strategy to encode categorical features. <br/>
 |      Available strategies = "label_encoding", "dummification", "random_projection", entity_embedding" <br/>
 |  <br/>
 |  verbose : boolean, defaut = False <br/>
 |      Verbose mode. Useful for entity embedding strategy. <br/>
 |  <br/>
 |  Methods defined here: <br/>
 |  <br/>
 |  __init__(self, strategy='label_encoding', verbose=False) <br/>
 |  <br/>
 |  fit(self, df_train, y_train) <br/>
 |      Fits Categorical Encoder. <br/>
 |      <br/>
 |      Parameters <br/>
 |      ---------- <br/>
 |      <br/>
 |      df_train : pandas dataframe of shape = (n_train, n_features) <br/>
 |      The train dataset with numerical and categorical features. NA values are allowed. <br/>
 |      <br/>
 |      y_train : pandas series of shape = (n_train, ). <br/>
 |      The target for classification or regression tasks. <br/>
 |      <br/>
 |      <br/>
 |      Returns <br/>
 |      ------- <br/>
 |      None <br/>
 |  <br/>
 |  fit_transform(self, df_train, y_train) <br/>
 |      Fits Categorical Encoder and transforms the dataset <br/>
 |      <br/>
 |      Parameters <br/>
 |      ---------- <br/>
 |      <br/>
 |      df_train : pandas dataframe of shape = (n_train, n_features) <br/>
 |      The train dataset with numerical and categorical features. NA values are allowed. <br/>
 |      <br/>
 |      y_train : pandas series of shape = (n_train, ). <br/>
 |      The target for classification or regression tasks. <br/>
 |      <br/>
 |      <br/>
 |      Returns <br/>
 |      ------- <br/>
 |      <br/>
 |      df_train : pandas dataframe of shape = (n_train, n_features) <br/>
 |      The train dataset with numerical and encoded categorical features. <br/>
 |  <br/>
 |  get_params(self, deep=True) <br/>
 |  <br/>
 |  set_params(self, **params) <br/>
 |  <br/>
 |  transform(self, df) <br/>
 |      Transforms the dataset <br/>
 |      <br/>
 |      Parameters <br/>
 |      ---------- <br/>
 |      <br/>
 |      df : pandas dataframe of shape = (n, n_features) <br/>
 |      The dataset with numerical and categorical features. NA values are allowed. <br/> 
 |      <br/>
 |      <br/>
 |      Returns <br/>
 |      ------- <br/>
 |      <br/>
 |      df : pandas dataframe of shape = (n, n_features) <br/>
 |      The dataset with numerical and encoded categorical features. <br/>
<br/>
<br/>
class NA_encoder <br/>
 |  Encodes missing values for both numerical and categorical features. Several strategies are possible in each case. <br/>
 |  <br/>
 |  <br/>
 |  Parameters <br/>
 |  ---------- <br/>
 |  <br/>
 |  numerical_strategy : string or float or int, defaut = "mean" <br/>
 |      The strategy to encode NA for numerical features. <br/>
 |      Available strategies = "mean", "median", "most_frequent" or a float/int value <br/>
 |  <br/>
 |  categorical_strategy : string, defaut = '<NULL>' <br/>
 |      The strategy to encode NA for categorical features. <br/> 
 |      Available strategies = a string or np.NaN <br/>
 |  <br/>
 |  Methods defined here: <br/>
 |  <br/>
 |  __init__(self, numerical_strategy='mean', categorical_strategy='<NULL>') <br/>
 |  <br/>
 |  fit(self, df_train, y_train=None) <br/>
 |      Fits NA Encoder. <br/>
 |      <br/>
 |      Parameters <br/>
 |      ---------- <br/>
 |      <br/>
 |      df_train : pandas dataframe of shape = (n_train, n_features) <br/>
 |      The train dataset with numerical and categorical features. <br/>
 |      <br/>
 |      y_train : [OPTIONAL]. pandas series of shape = (n_train, ). defaut = None <br/>
 |      The target for classification or regression tasks. <br/>
 |      <br/>        
 |      <br/>
 |      Returns <br/>
 |      ------- <br/>
 |      None <br/>
 |  <br/>
 |  fit_transform(self, df_train, y_train=None) <br/>
 |      Fits NA Encoder and transforms the dataset. <br/>
 |      <br/>
 |      Parameters <br/>
 |      ---------- <br/>
 |      <br/>
 |      df_train : pandas dataframe of shape = (n_train, n_features) <br/>
 |      The train dataset with numerical and categorical features. <br/>
 |      <br/>
 |      y_train : [OPTIONAL]. pandas series of shape = (n_train, ). defaut = None <br/>
 |      The target for classification or regression tasks. <br/>
 |      <br/>
 |      <br/>
 |      Returns <br/>
 |      ------- <br/>
 |      <br/>
 |      df_train : pandas dataframe of shape = (n_train, n_features) <br/>
 |      The train dataset with no missing values. <br/>
 |  <br/>
 |  get_params(self, deep=True) <br/>
 |  <br/>
 |  set_params(self, **params) <br/>
 |  <br/>
 |  transform(self, df) <br/>
 |      Transforms the dataset <br/>
 |      <br/>
 |      Parameters <br/>
 |      ---------- <br/>
 |      <br/>
 |      df : pandas dataframe of shape = (n, n_features) <br/>
 |      The dataset with numerical and categorical features. <br/>
 |      <br/>
 |      <br/>
 |      Returns <br/>
 |      ------- <br/>
 |      <br/>
 |      df : pandas dataframe of shape = (n, n_features) <br/>
 |      The dataset with no missing values. <br/>
<br/>
<br/>

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

#### class Drift_thresholder ####
*Automatically deletes ids and drifting variables between train and test datasets. Deletes on train and test datasets. The list of drift coefficients is available and saved as "drifts.txt"*

<br/>

> **Parameters**
> ___
>  
>   ***threshold*** : **float** (between 0.5 and 1.), defaut = 0.8 <br/>
> *Threshold used to deletes variables and ids. The lower the more you keep non-drifting/stable variables.*
>
> ***inplace*** : **bool**, defaut = False <br/>
> *If True, train and test datasets are transformed. Returns self. Otherwise, train and test datasets are not transformed. Returns a new dictionnary with cleaned datasets.*
> 
> ***verbose*** : **bool**, defaut = True <br/>
> *Verbose mode*
> 
> ***to_path*** : **str**, defaut = "save" <br/>
> *Name of the folder where the list of drift coefficients is saved* 

<br/>

> **Methods defined here:**
> ___
>
> <br/>
>
> ***init***(self, threshold=0.8, inplace=False, verbose=True, to_path='save') 
>
> <br/>
> 
> ***drifts***(self) 
>
> *Returns the univariate drifts for all variables.*
>
> <br/>
>
> ***fit_transform***(self, df)
>
> *Automatically deletes ids and drifting variables between train and test datasets. Deletes on train and test datasets. The list of drift coefficients is available and saved as "drifts.txt"*
>
>> **Parameters** 
>> ___ 
>>
>> ***df*** : **dict**, defaut = None <br/>
>> *Dictionnary containing :* <br/>
>> *'train' : pandas dataframe for train dataset* <br/>
>> *'test' : pandas dataframe for test dataset* <br/>
>> *'target' : pandas serie for the target* <br/>
>>
>> <br/>
>>
>> **Returns** 
>> ___ 
>>
>> ***df*** : **dict** <br/>
>> *Dictionnary containing :* <br/>
>> *'train' : pandas dataframe for train dataset* <br/>
>> *'test' : pandas dataframe for test dataset*<br/>
>> *'target' : pandas serie for the target* <br/>

<br/>
<br/>

####  class Reader  ####
*Reads and cleans data.*

<br/>

> **Parameters**
> ___
>  
> ***sep*** : **str**, defaut = None <br/>
> *Delimiter to use when reading a csv file.*
>
> ***header*** : **int or None**, defaut = 0 <br/>
> *If header=0, the first line is considered as a header. Otherwise, there is no header. Useful for csv and xls files.*
> 
> ***to_hdf5*** : **bool**, defaut = True <br/>
> *If True, dumps each file to hdf5 format*
>
> ***to_path*** : **str**, defaut = "save" <br/>
> *Name of the folder where files and encoders are saved*
>
> ***verbose*** : **bool**, defaut = True <br/>
> *Verbose mode* 

<br/>

> **Methods defined here:**
> ___
>
> <br/>
>
> ***init***(self, sep=None, header=0, to_hdf5=False, to_path='save', verbose=True) 
> 
> <br/>
>
> ***clean***(self, path, date_strategy, drop_duplicate) 
>
> *Reads and cleans data (accepted formats : csv, xls, json and h5) :* <br/>
> *- del Unnamed columns* <br/>
> *- casts lists into variables* <br/>
> *- try to cast variables into float* <br/>
> *- cleans dates* <br/>
> *- drop duplicates (if drop_duplicate=True)* <br/>
>
>> **Parameters** 
>> ___ 
>>
>> ***filepath*** : **str**, defaut = None <br/>
>> *filepath* <br/>
>>
>> ***date_strategy*** : **str**, defaut = "complete" <br/>
>> *The strategy to encode dates :* <br/> 
>> *- complete : creates timestamp from 01/01/2017, month, day and day_of_week* <br/>
>> *- to_timestamp : creates timestamp from 01/01/2017* <br/> 
>>
>> ***drop_duplicate*** : **bool**, defaut = False <br/>
>> *If True, drop duplicates when reading each file.*
>>
>> <br/>
>>
>> **Returns** 
>> ___ 
>>
>> ***df*** : **pandas dataframe** 
>
> <br/>
>
> ***train_test_split***(self, Lpath, target_name) 
>
> *Given a list of several paths and a target name, automatically creates and cleans train and test datasets. Also determines the task and encodes the target (classification problem only). Finally dumps the datasets to hdf5, and eventually the target encoder.*
>
>> **Parameters** 
>> ___ 
>> 
>> ***Lpath*** : **list**, defaut = None <br/>
>> *List of str paths to load the data*
>> 
>> ***target_name*** : **str**, defaut = None <br/> 
>> *The name of the target. Works for both classification (multiclass or not) and regression.* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***df*** : **dict**, defaut = None <br/>
>> *Dictionnary containing :* <br/>
>> *'train' : pandas dataframe for train dataset* <br/>
>> *'test' : pandas dataframe for test dataset* <br/>
>> *'target' : pandas serie for the target* <br/>
