####  class NA_encoder  ####
*Encodes missing values for both numerical and categorical features. Several strategies are possible in each case.*

<br/>

> **Parameters**
> ___
>  
> ***numerical_strategy*** : **str or float or int**, defaut = "mean" <br/>
> *The strategy to encode NA for numerical features. Available strategies = "mean", "median", "most_frequent" or a float/int value*
>
> ***categorical_strategy*** : **str**, defaut = "\<NULL\>" <br/>
> *The strategy to encode NA for categorical features. Available strategies = np.NaN or a str*

<br/>

> **Methods defined here:**
> ___
>
> <br/>
>
> ***init***(self, numerical_strategy='mean', categorical_strategy='\<NULL\>') 
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
>> ***y_train*** [OPTIONAL] : **pandas series**, shape = (n_train, ). Defaut = None <br/>
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
>> ***y_train*** [OPTIONAL] : **pandas series**, shape = (n_train, ). Defaut = None <br/>
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
