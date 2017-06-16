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
> <br/>
> ***init***(self, sep=None, header=0, to_hdf5=False, to_path='save', verbose=True) 
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
>> **Returns** 
>> ___ 
>>
>> ***df*** : **dict**, defaut = None <br/>
>> *Dictionnary containing :* <br/>
>> *'train' : pandas dataframe for train dataset* <br/>
>> *'test' : pandas dataframe for test dataset* <br/>
>> *'target' : pandas serie for the target* <br/>
