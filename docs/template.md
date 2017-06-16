####  class Predictor  ####
*Predicts the target on the test dataset.*

<br/>

> **Parameters**
> ___
>  
> ***to_path*** : **str**, defaut = "save" <br/>
> *Name of the folder where the feature importances and predictions are saved (.png and .csv format). Must contain target encoder object (for classification task only).*
>
> ***verbose*** : **bool**, defaut = True <br/>
> *Verbose mode.*

<br/>

> **Methods defined here:**
> ___
>
> <br/>
>
> ***init***(self, to_path='save', verbose=True) 
> 
> <br/>
>
> ***plot_feature_importances***(self, importance, fig_name = "feature_importance.png") 
>
> *Saves feature importances plot*
>
>> **Parameters** 
>> ___ 
>>
>> ***importance*** : **dict** <br/>
>> *Dictionnary with features (key) and importances (values).* 
>>
>> ***fig_name*** : **str**, defaut = "feature_importance.png" <br/>
>> *Figure name.* 
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
>> <br/>
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
