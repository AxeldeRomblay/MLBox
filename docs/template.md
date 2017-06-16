#### class Drift_thresholder ####
*Automatically deletes ids and drifting variables between train and test datasets. Deletes on train and test datasets. The list of drift coefficients is available and saved as "drifts.txt"*


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
> ***init***(self, threshold=0.8, inplace=False, verbose=True, to_path='save') 
>
> ***drifts***(self) 
>
> *Returns the univariate drifts for all variables.*
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
>> **Returns** 
>> ___ 
>>
>> ***df*** : **dict** <br/>
>> *Dictionnary containing :* <br/>
>> *'train' : pandas dataframe for train dataset* <br/>
>> *'test' : pandas dataframe for test dataset*<br/>
>> *'target' : pandas serie for the target* <br/>

