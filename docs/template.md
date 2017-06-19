####  class Optimiser  ####
*Optimises hyper-parameters of the whole Pipeline:* <br/>

*1. NA encoder (missing values encoder)*<br/> 
*2. CA encoder (categorical features encoder)*<br/> 
*3. Feature selector [OPTIONAL]*<br/> 
*4. Stacking estimator - feature engineer [OPTIONAL]*<br/> 
*5. Estimator (classifier or regressor)*<br/> 

*Works for both regression and classification (multiclass or binary) tasks.* <br/>

<br/>

> **Parameters**
> ___
>  
> ***scoring*** : **str, callable or None**, defaut = None <br/>
> *The scoring function used to optimise hyper-parameters. Compatible with sklearn metrics and scorer callable objects. If None, "log_loss" is used for classification and "mean_squarred_error" for regression.* <br/>
> * *Available scorings for classification: "accuracy", "roc_auc", "f1", "log_loss", "precision", "recall".* <br/>
> * *Available scorings for regression: "mean_absolute_error", "mean_squarred_error", "median_absolute_error", "r2".*
>
> ***n_folds*** : **int**, defaut = 2 <br/>
> *The number of folds for cross validation (stratified for classification)*
>
> ***random_state*** : **int**, defaut = 1 <br/>
> *pseudo-random number generator state used for shuffling*
>
> ***verbose*** : **bool**, defaut = True <br/>
> *Verbose mode.*

<br/>

> **Methods defined here:**
> ___
>
> <br/>
>
> ***init***(self, scoring=None, n_folds=2, random_state=1, verbose=True) 
> 
> <br/>
>
> ***evaluate***(self, params, df) 
>
> *Evaluates the scoring function with given hyper-parameters of the whole Pipeline. If no parameters are set, defaut configuration for each step is evaluated : no feature selection is applied and no meta features are created.*
>
>> **Parameters** 
>> ___ 
>>
>> ***params*** : **dict**, defaut = None <br/>
>> *Hyper-parameters dictionnary for the whole pipeline. If params = None, defaut configuration is evaluated.* <br/>
>>
>> * *The keys must respect the following syntax : "enc\_\_param".* <br/>
>>   * *With:* <br/>
>>       *1. "enc" = "ne" for NA encoder* <br/>
>>       *2. "enc" = "ce" for categorical encoder* <br/>
>>       *3. "enc" = "fs" for feature selector [OPTIONAL]* <br/>
>>       *4. "enc" = "stck"+str(i) to add layer n°i of meta-features (assuming 1 ... i-1 layers are created...) [OPTIONAL]* <br/>
>>       *5. "enc" = "est" for the final estimator* <br/>
>>   * *And:* <br/>
>>       *"param" : a correct associated parameter for each step. (for example : "max_depth" for "enc"="est", "entity_embedding" for "enc"="ce")* <br/>
>> >> * *The values are those of the parameters (for ex: 4 for a key="est\_\_max_depth").* <br/>
>> >> ***df*** : **dict**, defaut = None <br/>
>> *Dataset dictionnary. Must contain keys "train","test" and "target" with the train dataset (pandas DataFrame), the test dataset (pandas DataFrame) and the associated target (pandas Serie with dtype='float' for a regression or dtype='int' for a classification) resp.* 
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
> ***fit_predict***(self, params, df) 
>
> *Fits the model. Then predicts on test dataset and outputs feature importances and the submission file (.png and .csv formats).*
>
>> **Parameters** 
>> ___ 
>> 
>> ***params*** : **dict**, defaut = None <br/>
>> *Hyper-parameters dictionnary for the whole pipeline. If params = None, defaut configuration is evaluated.* <br/>
>>
>> * *The keys must respect the following syntax : "enc\_\_param".* <br/>
>>   * *With:* <br/>
>>       *1. "enc" = "ne" for NA encoder* <br/>
>>       *2. "enc" = "ce" for categorical encoder* <br/>
>>       *3. "enc" = "fs" for feature selector [OPTIONAL]* <br/>
>>       *4. "enc" = "stck"+str(i) to add layer n°i of meta-features (assuming 1 ... i-1 layers are created...) [OPTIONAL]* <br/>
>>       *5. "enc" = "est" for the final estimator* <br/>
>>   * *And:* <br/>
>>       *"param" : a correct associated parameter for each step. (for example : "max_depth" for "enc"="est", "entity_embedding" for "enc"="ce")* <br/>
>> 
>> * *The values are those of the parameters (for ex: 4 for a key="est\_\_max_depth").* <br/>
>> 
>> ***df*** : **dict**, defaut = None <br/>
>> *Dataset dictionnary. Must contain keys "train","test" and "target" with the train dataset (pandas DataFrame), the test dataset (pandas DataFrame) and the associated target (pandas Serie with dtype='float' for a regression or dtype='int' for a classification) resp.* 
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
> ***get_params***(self, deep=True)
>
> <br/>
>
> ***set_params***(self, params)
>
> <br/>
>
