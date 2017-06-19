#### class Classifier ####
*Wraps scikitlearn classifiers.* <br/>

<br/>

> **Parameters**
> ___
>  
> ***strategy*** : **str**, defaut = `"LightGBM"` (if installed else `"XGBoost"`) <br/>
> *The choice for the classifier.* <br/>
> *Available strategies = `"LightGBM"` (if installed), `"XGBoost"`, `"RandomForest"`, `"ExtraTrees"`, `"Tree"`, `"Bagging"`, `"AdaBoost"` or `"Linear"`.* 
>
> ***\*\*params*** <br/>
> *Parameters of the corresponding classifier. Ex: `n_estimators`, `max_depth`, ...*

<br/>

> **Methods defined here:**
> ___
>
> <br/>
>
> ***init***(self, strategy='l1', threshold=0.3) 
> 
> <br/>
>
> ***fit***(self, df_train, y_train) 
>
> *Fits Clf_feature_selector.*
>
>> **Parameters** 
>> ___ 
>>
>> ***df_train*** : **pandas dataframe**, shape = (n_train, n_features) <br/>
>> *The train dataset with numerical features and no NA* 
>>
>> ***y_train*** : **pandas series**, shape = (n_train, ) <br/>
>> *The target for classification task. Must be encoded.* 
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
> *Fits Clf_feature_selector and transforms the dataset*
>
>> **Parameters** 
>> ___ 
>> 
>> ***df_train*** : **pandas dataframe**, shape = (n_train, n_features) <br/>
>> *The train dataset with numerical features and no NA* 
>>
>> ***y_train*** : **pandas series**, shape = (n_train, ) <br/>
>> *The target for classification task. Must be encoded.* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***df_train*** : **pandas dataframe**, shape = (n_train, n_features*(1-threshold)) <br/>
>> *The train dataset with relevant features* 
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
>> *The dataset with numerical features and no NA* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***df*** : **pandas dataframe**, shape = (n, n_features*(1-threshold)) <br/>
>> *The train dataset with relevant features.* 
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
