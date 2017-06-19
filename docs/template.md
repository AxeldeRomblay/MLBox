#### class StackingClassifier ####
*A Stacking classifier is a classifier that uses the predictions of several first layer estimators (generated with a cross validation method) for a second layer estimator.* <br/>

<br/>

> **Parameters**
> ___
>  
> ***base_estimators*** : **list**, defaut = `[Classifier(strategy = "XGBoost"), Classifier(strategy = "RandomForest"), Classifier(strategy = "ExtraTrees")]` <br/>
> *List of estimators to fit in the first level using a cross validation.* 
>
> ***level_estimator*** : **object**, defaut = `LogisticRegression()` <br/>
> *The estimator used in second and last level.*
>
> ***n_folds*** : **int**, defaut = `5` [OPTIONAL] <br/>
> *Number of folds used to generate the meta features for the training set.*
>
> ***copy*** : **bool**, defaut = `False` [OPTIONAL] <br/>
> *If true, meta features are added to the original dataset.*
>
> ***drop_first*** : **bool**, defaut = `True` [OPTIONAL] <br/>
> *If True, each estimator output n_classes-1 probabilities.*
>
> ***random_state*** : **None, int or RandomState**, defaut = `1` [OPTIONAL] <br/>
> *Pseudo-random number generator state used for shuffling. If None, use default numpy RNG for shuffling.*
>
> ***verbose*** : **bool**, defaut = `True` [OPTIONAL] <br/>
> *Verbose mode.*

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
> ***fit***(self, df_train, y_train) 
>
> *Fits Reg_feature_selector.*
>
>> **Parameters** 
>> ___ 
>>
>> ***df_train*** : **pandas dataframe**, shape = (n_train, n_features) <br/>
>> *The train dataset with numerical features and no NA* 
>>
>> ***y_train*** : **pandas series**, shape = (n_train, ) <br/>
>> *The target for regression task.* 
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
> *Fits Reg_feature_selector and transforms the dataset*
>
>> **Parameters** 
>> ___ 
>> 
>> ***df_train*** : **pandas dataframe**, shape = (n_train, n_features) <br/>
>> *The train dataset with numerical features and no NA* 
>>
>> ***y_train*** : **pandas series**, shape = (n_train, ) <br/>
>> *The target for regression task.* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***df_train_transform*** : **pandas dataframe**, shape = (n_train, n_features*(1-threshold)) <br/>
>> *The train dataset with relevant features* 
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
>> *The dataset with numerical features and no NA* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***df_transform*** : **pandas dataframe**, shape = (n, n_features*(1-threshold)) <br/>
>> *The train dataset with relevant features.* 
>
> <br/>
>
> ***get_params***(self, deep=True)
>
> <br/>
>
> ***set_params***(self, params)
