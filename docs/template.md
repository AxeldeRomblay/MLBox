#### class Clf_feature_selector ####
*Selects useful features. Several strategies are possible (filter and wrapper methods). Works for classification problems only (multiclass or binary).* <br/>

<br/>

> **Parameters**
> ___
>  
> ***strategy*** : **str**, defaut = `"l1"` <br/>
> *The strategy to select features.* <br/>
> *Available strategies = `"variance"`, `"l1"` or `"rf_feature_importance"`. 
>
> ***threshold*** : **float**, defaut = `0.3` <br/>
> *The percentage of variables to discard according to the strategy. Must be between 0. and 1.*

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
> ***get_params***(self, deep=True)
>
> <br/>
>
> ***set_params***(self, params)

<br/>
<br/>
