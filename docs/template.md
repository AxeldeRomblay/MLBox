####  class Reader  ####
*Reads and cleans data.*

> **Parameters**
> ___
>  
> ***sep*** : **str**, defaut = None <br/>
> *Delimiter to use when reading a csv file. *
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

