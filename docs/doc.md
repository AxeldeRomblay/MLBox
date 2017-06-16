**class Drift_thresholder** <br/>
*Automatically deletes ids and drifting variables between train and test datasets.Deletes on train and test datasets. The list of drift coefficients is available and saved as "drifts.txt"*


> **Parameters**
> ___
>  
>   ***threshold*** : **float** (between 0.5 and 1.), defaut = 0.9 
> *Threshold used to deletes variables and ids. The lower the more you keep non-drifting/stable variables.*
>
> ***inplace*** : **bool**, defaut = False 
> *If True, train and test datasets are transformed. Returns self. Otherwise, train and test datasets are not transformed. Returns a new dictionnary with cleaned datasets.*
> 
> ***verbose*** : **bool**, defaut = True 
> *Verbose mode*
> 
> ***to_path*** : **str**, defaut = "save" 
> *Name of the folder where the list of drift coefficients is saved* 

<br/>
<br/>

> **Methods defined here:**
> ___
>
> ***init***(self, threshold=0.8, inplace=False, verbose=True, to_path='save') 
>
> ***drifts***(self) 
> *Returns the univariate drifts for all variables.*
>
> ***fit_transform***(self, df)
>
> *Automatically deletes ids and drifting variables between train and test datasets. Deletes on train and test datasets. The list of drift coefficients is available and saved as "drifts.txt"*
>
>> **Parameters** 
>> ___ 
>>
>> ***df*** : **dict**, defaut = None 
>> *Dictionnary containing :*
>> *'train' : pandas dataframe for train dataset*
>> *'test' : pandas dataframe for test dataset* 
>> *'target' : pandas serie for the target* 
>>
>> **Returns** 
>> ___ 
>>
>> ***df*** : **dict** 
>> *Dictionnary containing :* 
>> *'train' : pandas dataframe for train dataset* 
>> *'test' : pandas dataframe for test dataset*
>> *'target' : pandas serie for the target*

