#### class Regressor ####
*Wraps scikitlearn regressors.* <br/>

<br/>

> **Parameters**
> ___
>  
> ***strategy*** : **str**, defaut = `"LightGBM"` (if installed else `"XGBoost"`) <br/>
> *The choice for the regressor.* <br/>
> *Available strategies = `"LightGBM"` (if installed), `"XGBoost"`, `"RandomForest"`, `"ExtraTrees"`, `"Tree"`, `"Bagging"`, `"AdaBoost"` or `"Linear"`.* 
>
> ***\*\*params*** <br/>
> *Parameters of the corresponding regressor. Ex: `n_estimators`, `max_depth`, ...*

<br/>

> **Methods defined here:**
> ___
>
> <br/>
>
> ***init***(self, strategy='LightGBM', \*\*params) 
> 
> <br/>
>
> ***feature_importances***(self) 
>
> *Computes feature importances. Regressor must be fitted before.*
>
>> **Parameters** 
>> ___ 
>>
>> ***None*** 
>>
>> <br/>
>>
>> **Returns** 
>> ___ 
>>
>> ***importance*** : **dict** <br/>
>> *Dictionnary containing a measure of feature importance (value) for each feature (key).*
>
> <br/>
>
> ***fit***(self, df_train, y_train) 
>
> *Fits Regressor.*
>
>> **Parameters** 
>> ___ 
>> 
>> ***df_train*** : **pandas dataframe**, shape = (n_train, n_features) <br/>
>> *The train dataset with numerical features and no NA* 
>>
>> ***y_train*** : **pandas series**, shape = (n_train, ) <br/>
>> *The target for regression task.* 
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
> ***predict***(self, df) 
>
> *Predicts the target.*
>
>> **Parameters** 
>> ___ 
>> 
>> ***df*** : **pandas dataframe**, shape = (n, n_features) <br/>
>> *The dataset with numerical features.* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***y*** : **array**, shape = (n, ) <br/>
>> *The target to be predicted.* 
>
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
>> *The dataset with numerical features.* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***df_transform*** : **pandas dataframe**, shape = (n, n_selected_features) <br/>
>> *The transformed dataset with its most important features.* 
>
> <br/>
>
> ***score***(self, df, y , sample_weight=None)
>
> *Returns the coefficient of determination R^2 of the prediction.*
>
>> **Parameters** 
>> ___ 
>> 
>> ***df*** : **pandas dataframe**, shape = (n, n_features) <br/>
>> *The dataset with numerical features.* 
>>
>> ***y*** : **pandas series**, shape = (n,) <br/>
>> *The target for regression task.*
>>
>> ***sample_weight*** : **array**, shape = (n,) [OPTIONAL]<br/>
>> *Sample weights*
>>
>> <br/>
>>
>> **Returns** 
>> ___ 
>>
>> ***score*** : **float** <br/>
>> *R^2 of self.predict(X) wrt. y.*
>
> <br/>
>
> ***get_estimator***(self)
>
> *Returns sklearn regressor.*
>
>> **Parameters** 
>> ___ 
>>
>> ***None*** 
>>
>> <br/>
>>
>> **Returns** 
>> ___ 
>>
>> ***estimator*** : **sklearn regressor** <br/>
>> *Sklearn estimator.*
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
