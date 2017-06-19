#### class Classifier ####
*Wraps scikitlearn classifiers.* <br/>

<br/>

> **Parameters**
> ___
>  
> ***strategy*** : **str**, defaut = `"LightGBM"` (if installed else `"XGBoost"`) <br/>
> *The choice for the classifier.* <br/>
> *Available strategies = `"LightGBM"` (if installed), `"XGBoost"`, `"RandomForest"`, `"ExtraTrees"`, `"Tree"`, `"Bagging"`, `"AdaBoost"` or `"Linear"`.* 
>
> ***\*\*params*** <br/>
> *Parameters of the corresponding classifier. Ex: `n_estimators`, `max_depth`, ...*

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
> ***feature_importances***(self) 
>
> *Computes feature importances. Classifier must be fitted before.*
>
>> **Parameters** 
>> ___ 
>>
>> ***None*** 
>>
>> <br/>
>>
>> **Returns** 
>> ___ 
>>
>> ***importance*** : **dict** <br/>
>> *Dictionnary containing a measure of feature importance (value) for each feature (key).*
>
> <br/>
>
> ***fit***(self, df_train, y_train) 
>
> *Fits Classifier.*
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
>> ***None*** 
>
> <br/>
>
> ***predict***(self, df) 
>
> *Predicts the target.*
>
>> **Parameters** 
>> ___ 
>> 
>> ***df*** : **pandas dataframe**, shape = (n, n_features) <br/>
>> *The dataset with numerical features.* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***y*** : **array**, shape = (n, ) <br/>
>> *The encoded classes to be predicted.* 
>
> <br/>
>
> ***predict_log_proba***(self, df) 
>
> *Predicts class log-probabilities for df.*
>
>> **Parameters** 
>> ___ 
>> 
>> ***df*** : **pandas dataframe**, shape = (n, n_features) <br/>
>> *The dataset with numerical features.* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***y*** : **array**, shape = (n, n_classes) <br/>
>> *The log-probabilities for each class* 
>
> <br/>
>
> ***predict_proba***(self, df) 
>
> *Predicts class probabilities for df.*
>
>> **Parameters** 
>> ___ 
>> 
>> ***df*** : **pandas dataframe**, shape = (n, n_features) <br/>
>> *The dataset with numerical features.* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***y*** : **array**, shape = (n, n_classes) <br/>
>> *The probabilities for each class* 
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
>> *The dataset with numerical features.* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***df_transform*** : **pandas dataframe**, shape = (n, n_selected_features) <br/>
>> *The transformed dataset with its most important features.* 
>
> <br/>
>
> ***score***(self, df, y , sample_weight=None)
>
> *Returns the mean accuracy.*
>
>> **Parameters** 
>> ___ 
>> 
>> ***df*** : **pandas dataframe**, shape = (n, n_features) <br/>
>> *The dataset with numerical features.* 
>>
>> ***y*** : **pandas series**, shape = (n,) <br/>
>> *The numerical encoded target for classification tasks.*
>>
>> ***sample_weight*** : **array**, shape = (n,) [OPTIONAL]<br/>
>> *Sample weights*
>>
>> <br/>
>>
>> **Returns** 
>> ___ 
>>
>> ***score*** : **float** <br/>
>> *Mean accuracy of self.predict(df) wrt. y.*
>
> <br/>
>
> ***get_estimator***(self)
>
> *Returns sklearn classifier.*
>
>> **Parameters** 
>> ___ 
>>
>> ***None*** 
>>
>> <br/>
>>
>> **Returns** 
>> ___ 
>>
>> ***estimator*** : **sklearn classifier** <br/>
>> *Sklearn estimator.*
>
> <br/>
>
> ***get_params***(self, deep=True)
>
> <br/>
>
> ***set_params***(self, params)

<br/>
<br/>
