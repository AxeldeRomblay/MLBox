####  class Categorical_encoder  ####
*Encodes categorical features. Several strategies are possible (supervised or not). Works for both classification and regression tasks.*

<br/>

> **Parameters**
> ___
>  
> ***strategy*** : **str**, defaut = "label_encoding" <br/>
> *The strategy to encode categorical features. Available strategies = "label_encoding", "dummification", "random_projection", entity_embedding"*
>
> ***verbose*** : **bool**, defaut = False <br/>
> *Verbose mode. Useful for entity embedding strategy.*

<br/>

> **Methods defined here:**
> ___
>
> <br/>
>
> ***init***(self, strategy='label_encoding', verbose=False) 
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
