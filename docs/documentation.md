Welcome to MLBox's documentation!
======================================

This page is the official documentation for MLBox package. You will learn how to use all the features provided by this tool. [**Here**](https://github.com/AxeldeRomblay/MLBox/blob/master/docs/MLBox.pdf) is an extra-explanation of the different subpackages. 


### Table of Contents

* **[encoding](#encoding)**<br>

* **[model](#model)**<br>
  * [classification](#classification) <br>
  * [regression](#regression) <br>

* **[optimisation](#optimisation)**<br>

* **[prediction](#prediction)**<br>

* **[preprocessing](#preprocessing)**<br>

<br/>

# MLBox's documentation 

<br/>

## encoding

<br/>

####  class Categorical_encoder  ####
*Encodes categorical features. Several strategies are possible (supervised or not). Works for both classification and regression tasks.*

<br/>

> **Parameters**
> ___
>  
> ***strategy*** : **str**, defaut = `"label_encoding"` <br/>
> *The strategy to encode categorical features. Available strategies = `"label_encoding"`, `"dummification"`, `"random_projection"`, `"entity_embedding"`*
>
> ***verbose*** : **bool**, defaut = `False` <br/>
> *Verbose mode. Useful for entity embedding strategy.*

<br/>

> **Methods defined here:**
> ___
>
> <br/>
>
> ***init***(self, strategy="label_encoding", verbose=False) 
> 
> <br/>
>
> ***fit***(self, df_train, y_train) 
>
> *Fits Categorical Encoder.*
>
>> **Parameters** 
>> ___ 
>>
>> ***df_train*** : **pandas dataframe**, shape = (n_train, n_features) <br/>
>> *The train dataset with numerical and categorical features. NA values are allowed.* 
>>
>> ***y_train*** : **pandas series**, shape = (n_train, ) <br/>
>> *The target for classification or regression tasks.* 
>>
>> **Returns** 
>> ___ 
>>
>> ***None*** 
>
> <br/>
>
> ***fit_transform***(self, df_train, y_train) 
>
> *Fits Categorical Encoder and transforms the dataset*
>
>> **Parameters** 
>> ___ 
>> 
>> ***df_train*** : **pandas dataframe**, shape = (n_train, n_features) <br/>
>> *The train dataset with numerical and categorical features. NA values are allowed.* 
>>
>> ***y_train*** : **pandas series**, shape = (n_train, ) <br/>
>> *The target for classification or regression tasks.* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***df_train*** : **pandas dataframe**, shape = (n_train, n_features) <br/>
>> *The train dataset with numerical and encoded categorical features* 
>
> <br/>
>
> ***get_params***(self, deep=True)
>
> <br/>
>
> ***set_params***(self, params)
>
> <br/>
>
> ***transform***(self, df)
>
> *Transforms the dataset*
>
>> **Parameters** 
>> ___ 
>> 
>> ***df*** : **pandas dataframe**, shape = (n, n_features) <br/>
>> *The dataset with numerical and categorical features. NA values are allowed.* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***df*** : **pandas dataframe**, shape = (n, n_features) <br/>
>> *The dataset with numerical and encoded categorical features* 

<br/>
<br/>

####  class NA_encoder  ####
*Encodes missing values for both numerical and categorical features. Several strategies are possible in each case.*

<br/>

> **Parameters**
> ___
>  
> ***numerical_strategy*** : **str or float or int**, defaut = `"mean"` <br/>
> *The strategy to encode NA for numerical features. Available strategies = `"mean"`, `"median"`, `"most_frequent"` or a float/int value*
>
> ***categorical_strategy*** : **str**, defaut = `"<NULL>"` <br/>
> *The strategy to encode NA for categorical features. Available strategies = `np.NaN` or a str*

<br/>

> **Methods defined here:**
> ___
>
> <br/>
>
> ***init***(self, numerical_strategy="mean", categorical_strategy="\<NULL\>") 
> 
> <br/>
>
> ***fit***(self, df_train, y_train=None) 
>
> *Fits NA Encoder.*
>
>> **Parameters** 
>> ___ 
>>
>> ***df_train*** : **pandas dataframe**, shape = (n_train, n_features) <br/>
>> *The train dataset with numerical and categorical features.* 
>>
>> ***y_train*** [OPTIONAL] : **pandas series**, shape = (n_train, ). Defaut = `None` <br/>
>> *The target for classification or regression tasks.* 
>>
>> **Returns** 
>> ___ 
>>
>> ***None*** 
>
> <br/>
>
> ***fit_transform***(self, df_train, y_train=None) 
>
> *Fits NA Encoder and transforms the dataset*
>
>> **Parameters** 
>> ___ 
>> 
>> ***df_train*** : **pandas dataframe**, shape = (n_train, n_features) <br/>
>> *The train dataset with numerical and categorical features.* 
>>
>> ***y_train*** [OPTIONAL] : **pandas series**, shape = (n_train, ). Defaut = `None` <br/>
>> *The target for classification or regression tasks.* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***df_train*** : **pandas dataframe**, shape = (n_train, n_features) <br/>
>> *The train dataset with no missing values* 
>
> <br/>
>
> ***get_params***(self, deep=True)
>
> <br/>
>
> ***set_params***(self, params)
>
> <br/>
>
> ***transform***(self, df)
>
> *Transforms the dataset*
>
>> **Parameters** 
>> ___ 
>> 
>> ***df*** : **pandas dataframe**, shape = (n, n_features) <br/>
>> *The dataset with numerical and categorical features.* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***df*** : **pandas dataframe**, shape = (n, n_features) <br/>
>> *The dataset with no missing values* 

<br/>
<br/>

## model

<br/>

### classification

<br/>

### regression

<br/>
<br/>

## optimisation

<br/>

####  class Optimiser  ####
*Optimises hyper-parameters of the whole Pipeline:* <br/>

*1. NA encoder (missing values encoder)*<br/> 
*2. CA encoder (categorical features encoder)*<br/> 
*3. Feature selector [OPTIONAL]*<br/> 
*4. Stacking estimator - feature engineer [OPTIONAL]*<br/> 
*5. Estimator (classifier or regressor)*<br/> 

*Works for both regression and classification (multiclass or binary) tasks.* <br/>

<br/>

> **Parameters**
> ___
>  
> ***scoring*** : **str, callable or None**, defaut = `None` <br/>
> *The scoring function used to optimise hyper-parameters. Compatible with sklearn metrics and scorer callable objects. If None, `"log_loss"` is used for classification and `"mean_squarred_error"` for regression.* <br/>
> * *Available scorings for classification: `"accuracy"`, `"roc_auc"`, `"f1"`, `"log_loss"`, `"precision"`, `"recall"`.* <br/>
> * *Available scorings for regression: `"mean_absolute_error"`, `"mean_squarred_error"`, `"median_absolute_error"`, `"r2"`.*
>
> ***n_folds*** : **int**, defaut = `2` <br/>
> *The number of folds for cross validation (stratified for classification)*
>
> ***random_state*** : **int**, defaut = `1` <br/>
> *pseudo-random number generator state used for shuffling*
>
> ***verbose*** : **bool**, defaut = `True` <br/>
> *Verbose mode.*

<br/>

> **Methods defined here:**
> ___
>
> <br/>
>
> ***init***(self, scoring=None, n_folds=2, random_state=1, verbose=True) 
> 
> <br/>
>
> ***evaluate***(self, params, df) 
>
> *Evaluates the scoring function with given hyper-parameters of the whole Pipeline. If no parameters are set, defaut configuration for each step is evaluated : no feature selection is applied and no meta features are created.*
>
>> **Parameters** 
>> ___ 
>>
>> ***params*** : **dict**, defaut = `None` <br/>
>> *Hyper-parameters dictionnary for the whole pipeline. If `params = None`, defaut configuration is evaluated.* <br/>
>>
>> * *The keys must respect the following syntax : `"enc__param"`.* <br/>
>>
>>    * *With:* <br/>
>>      *1. `"enc" = "ne"` for NA encoder* <br/>
>>      *2. `"enc" = "ce"` for categorical encoder* <br/>
>>      *3. `"enc" = "fs"` for feature selector [OPTIONAL]* <br/>
>>      *4. `"enc" = "stck"+str(i)` to add layer n°i of meta-features (assuming i-1 layers are created) [OPTIONAL]* <br/>
>>      *5. `"enc" = "est"` for the final estimator* <br/>
>>    * *And:* <br/>
>>      *`"param"` : a correct associated parameter for each step. (for example : `"max_depth"` for `"enc"="est"`, `"entity_embedding"` for `"enc"="ce"`)* <br/>
>> 
>> * *The values are those of the parameters (for ex: 4 for a key=`"est__max_depth"`).* <br/>
>> 
>> ***df*** : **dict**, defaut = `None` <br/>
>> *Dataset dictionnary. Must contain keys "train","test" and "target" with the train dataset (pandas DataFrame), the test dataset (pandas DataFrame) and the associated target (pandas Serie with `dtype='float'` for a regression or `dtype='int'` for a classification) resp.* 
>>
>> <br/>
>>
>> **Returns** 
>> ___ 
>>
>> ***score*** : **float** <br/>
>> *The score. The higher the better (positive for a score and negative for a loss).*
>
> <br/>
>
> ***optimise***(self, space, df, max_evals=40) 
>
> *Optimises hyper-parameters of the whole Pipeline with a given scoring function. Algorithm used to optimise : Tree Parzen Estimator.* <br/>
> *IMPORTANT : Try to avoid dependent parameters and to set one feature selection strategy and one estimator strategy at a time.*
>
>> **Parameters** 
>> ___ 
>> 
>> ***space*** : **dict**, defaut = `None` <br/>
>> *Hyper-parameters space* <br/>
>>
>> * *The keys must respect the following syntax : `"enc__param"`.* <br/>
>>   * *With:* <br/>
>>      *1. `"enc" = "ne"` for NA encoder* <br/>
>>      *2. `"enc" = "ce"` for categorical encoder* <br/>
>>      *3. `"enc" = "fs"` for feature selector [OPTIONAL]* <br/>
>>      *4. `"enc" = "stck"+str(i)` to add layer n°i of meta-features (assuming i-1 layers are created...) [OPTIONAL]* <br/>
>>      *5. `"enc" = "est"` for the final estimator* <br/>
>>   * *And:* <br/>
>>       *`"param"` : a correct associated parameter for each step. (for example : `"max_depth"` for `"enc"="est"`, `"entity_embedding"` for `"enc"="ce"`)* <br/>
>> 
>> * *The values must respect the following syntax : `{"search" : strategy, "space" : list}`* <br/>
>>   * *With `strategy = "choice"` or `"uniform"`. Defaut = `"choice"`* <br/>
>>   * *And `list` : a list of values to be tested if `strategy = "choice"`. If `strategy = "uniform"`, `list = [value_min, value_max]`.* <br/>
>> 
>> ***df*** : **dict**, defaut = `None` <br/>
>> *Train dictionnary. Must contain keys "train" and "target" with the train dataset (pandas DataFrame) and the associated target (pandas Serie with `dtype='float'` for a regression or `dtype='int'` for a classification) resp.* 
>>
>> ***max_evals*** : **int**, defaut = `40`. <br/>
>> *Number of iterations. For an accurate optimal hyper-parameter, `max_evals = 40`.*
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***best_params*** : **dict** <br/>
>> *The optimal hyper-parameter dictionnary.*
>
> <br/>
>
> ***get_params***(self, deep=True)
>
> <br/>
>
> ***set_params***(self, params)

<br/>
<br/>

## prediction

<br/>

####  class Predictor  ####
*Predicts the target on the test dataset.*

<br/>

> **Parameters**
> ___
>  
> ***to_path*** : **str**, defaut = `"save"` <br/>
> *Name of the folder where the feature importances and predictions are saved (.png and .csv format). Must contain target encoder object (for classification task only).*
>
> ***verbose*** : **bool**, defaut = `True` <br/>
> *Verbose mode.*

<br/>

> **Methods defined here:**
> ___
>
> <br/>
>
> ***init***(self, to_path='save', verbose=True) 
> 
> <br/>
>
> ***plot_feature_importances***(self, importance, fig_name = "feature_importance.png") 
>
> *Saves feature importances plot*
>
>> **Parameters** 
>> ___ 
>>
>> ***importance*** : **dict** <br/>
>> *Dictionnary with features (key) and importances (values).* 
>>
>> ***fig_name*** : **str**, defaut = `"feature_importance.png"` <br/>
>> *Figure name.* 
>>
>> **Returns** 
>> ___ 
>>
>> ***None*** 
>
> <br/>
>
> ***fit_predict***(self, params, df) 
>
> *Fits the model. Then predicts on test dataset and outputs feature importances and the submission file (.png and .csv formats).*
>
>> **Parameters** 
>> ___ 
>> 
>> ***params*** : **dict**, defaut = `None` <br/>
>> *Hyper-parameters dictionnary for the whole pipeline. If `params = None`, defaut configuration is evaluated.* <br/>
>>
>> * *The keys must respect the following syntax : `"enc__param"`.* <br/>
>>   * *With:* <br/>
>>       *1. `"enc" = "ne"` for NA encoder* <br/>
>>       *2. `"enc" = "ce"` for categorical encoder* <br/>
>>       *3. `"enc" = "fs"` for feature selector [OPTIONAL]* <br/>
>>       *4. `"enc" = "stck"+str(i)` to add layer n°i of meta-features (assuming i-1 layers are created) [OPTIONAL]* <br/>
>>       *5. `"enc" = "est"` for the final estimator* <br/>
>>   * *And:* <br/>
>>       *`"param"` : a correct associated parameter for each step. (for example : `"max_depth"` for `"enc"="est"`, `"entity_embedding"` for `"enc"="ce"`)* <br/>
>> 
>> * *The values are those of the parameters (for ex: `4` for a key = `"est__max_depth"`).* <br/>
>> 
>> ***df*** : **dict**, defaut = `None` <br/>
>> *Dataset dictionnary. Must contain keys "train","test" and "target" with the train dataset (pandas DataFrame), the test dataset (pandas DataFrame) and the associated target (pandas Serie with dtype='float' for a regression or dtype='int' for a classification) resp.* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***None*** 
>
> <br/>
>
> ***get_params***(self, deep=True)
>
> <br/>
>
> ***set_params***(self, params)


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
>   ***threshold*** : **float** (between 0.5 and 1.), defaut = `0.8` <br/>
> *Threshold used to deletes variables and ids. The lower the more you keep non-drifting/stable variables.*
>
> ***inplace*** : **bool**, defaut = `False` <br/>
> *If True, train and test datasets are transformed. Returns self. Otherwise, train and test datasets are not transformed. Returns a new dictionnary with cleaned datasets.*
> 
> ***verbose*** : **bool**, defaut = `True` <br/>
> *Verbose mode*
> 
> ***to_path*** : **str**, defaut = `"save"` <br/>
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
>> ***df*** : **dict**, defaut = `None` <br/>
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
> ***sep*** : **str**, defaut = `None` <br/>
> *Delimiter to use when reading a csv file.*
>
> ***header*** : **int or None**, defaut = `0` <br/>
> *If header=0, the first line is considered as a header. Otherwise, there is no header. Useful for csv and xls files.*
> 
> ***to_hdf5*** : **bool**, defaut = `True` <br/>
> *If True, dumps each file to hdf5 format*
>
> ***to_path*** : **str**, defaut = `"save"` <br/>
> *Name of the folder where files and encoders are saved*
>
> ***verbose*** : **bool**, defaut = `True` <br/>
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
>> ***filepath*** : **str**, defaut = `None` <br/>
>> *filepath* <br/>
>>
>> ***date_strategy*** : **str**, defaut = `"complete"` <br/>
>> *The strategy to encode dates :* <br/> 
>> *- complete : creates timestamp from 01/01/2017, month, day and day_of_week* <br/>
>> *- to_timestamp : creates timestamp from 01/01/2017* <br/> 
>>
>> ***drop_duplicate*** : **bool**, defaut = `False` <br/>
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
>> ***Lpath*** : **list**, defaut = `None` <br/>
>> *List of str paths to load the data*
>> 
>> ***target_name*** : **str**, defaut = `None` <br/> 
>> *The name of the target. Works for both classification (multiclass or not) and regression.* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***df*** : **dict** <br/>
>> *Dictionnary containing :* <br/>
>> *'train' : pandas dataframe for train dataset* <br/>
>> *'test' : pandas dataframe for test dataset* <br/>
>> *'target' : pandas serie for the target* <br/>
