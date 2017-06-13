Welcome to MLBox's documentation!
======================================

This page is the official documentation for MLBox package. You will learn how to use all the features provided by this tool. 


### Table of Contents

* **[encoding](#encoding)**<br>

* **[model](#model)**<br>
  * [classification](#classification) <br>
  * [regression](#regression) <br>

* **[optimisation](#optimisation)**<br>

* **[prediction](#prediction)**<br>

* **[preprocessing](#preprocessing)**<br>


## Documentation

### encoding



### model

#### classification

#### regression

### optimisation

### prediction

### preprocessing

class Reader <br/>
|  Reads and cleans data <br/>
| <br/>
|  Parameters <br/>
|  ---------- <br/> 
| <br/>
|  sep : str, defaut = None <br/>
|       Delimiter to use when reading a csv file. <br/>
| <br/>
|  header : int or None, defaut = 0. <br/>
|      If header=0, the first line is considered as a header. <br/>
|      Otherwise, there is no header. <br/>
|      Useful for csv and xls files. <br/>
| <br/>
|  to_hdf5 : bool, defaut = True <br/>
|      If True, dumps each file to hdf5 format <br/>
| <br/>
|  to_path : str, defaut = "save" <br/>
|      Name of the folder where files and encoders are saved <br/>
| <br/>
|  verbose : bool, defaut = True <br/>
|      Verbose mode <br/>
| <br/>
|  Methods defined here: <br/>
| <br/>
|  __init__(self, sep=None, header=0, to_hdf5=False, to_path='save', verbose=True) 
|   
|  clean(self, path, date_strategy, drop_duplicate) 
|      Reads and cleans data (accepted formats : csv, xls, json and h5) : 
|       
|      - del Unnamed columns 
|      - casts lists into variables 
|      - try to cast variables into float 
|      - cleans dates 
|      - drop duplicates (if drop_duplicate=True) 
|       
|      Parameters 
|      ---------- 
|       
|      filepath_or_buffer : str, pathlib.Path
|          The string could be a URL. 
|
|      date_strategy : str, defaut = "complete" 
|          The strategy to encode dates : 
|          - complete : creates timestamp from 01/01/2017, month, day and day_of_week 
|          - to_timestamp : creates timestamp from 01/01/2017 
|       
|      drop_duplicate : bool, defaut = False 
|          If True, drop duplicates when reading each file. 
|       
|      Returns 
|      ------- 
|       
|      df : pandas dataframe 
|   
|
|  train_test_split(self, Lpath, target_name) 
|      Given a list of several paths and a target name, automatically creates and cleans train and test datasets. 
|      Also determines the task and encodes the target (classification problem only). 
|      Finally dumps the datasets to hdf5, and eventually the target encoder. 
|       
|      Parameters 
|      ---------- 
|       
|      Lpath : list, defaut = None 
|          List of str paths to load the data 
|       
|      target_name : str, defaut = None 
|          The name of the target. Works for both classification (multiclass or not) and regression. 
|            
|      Returns 
|      ------- 
|       
|      df : dict 
|          Dictionnary containing : 
|          'train' : pandas dataframe for train dataset 
|          'test' : pandas dataframe for test dataset 
|          'target' : pandas serie for the target
 



