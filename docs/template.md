#### class StackingRegressor ####
*A Stacking regressor is a regressor that uses the predictions of several first layer estimators (generated with a cross validation method) for a second layer estimator.* <br/>

<br/>

> **Parameters**
> ___
>  
> ***base_estimators*** : **list**, defaut = `[Regressor(strategy = "XGBoost"), Regressor(strategy = "RandomForest"), Regressor(strategy = "ExtraTrees")]` <br/>
> *List of estimators to fit in the first level using a cross validation.* 
>
> ***level_estimator*** : **object**, defaut = `LinearRegression()` <br/>
> *The estimator used in second and last level.*
>
> ***n_folds*** : **int**, defaut = `5` [OPTIONAL] <br/>
> *Number of folds used to generate the meta features for the training set.*
>
> ***copy*** : **bool**, defaut = `False` [OPTIONAL] <br/>
> *If true, meta features are added to the original dataset.*
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
> ***init***(self, base_estimators=[Regressor(strategy = "XGBoost"), Regressor(strategy = "RandomForest"), Regressor(strategy = "ExtraTrees")], level_estimator=LinearRegression(), n_folds=5, copy=False, random_state=1, verbose=True) 
> 
> <br/>
>
> ***fit***(self, df_train, y_train) 
>
> *Fits the first level estimators and the second level estimator on train dataset.*
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
> *Create meta-features for the training dataset.*
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
>> ***df_train_transform*** : **pandas dataframe**, shape = (n_train, n_features*int(copy)+len(base_estimators)) <br/>
>> *The transformed train dataset with meta features.* 
>
> <br/>
>
> ***transform***(self, df_test)
>
> *Transforms and creates meta features for the test dataset only. If you want to transform the train dataset, you have to use `fit_transform` function.*
>
>> **Parameters** 
>> ___ 
>> 
>> ***df_test*** : **pandas dataframe**, shape = (n_test, n_features) <br/>
>> *The test dataset with numerical features and no NA* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***df_test_transform*** : **pandas dataframe**, shape = (n_test, n_features*int(copy)+len(base_estimators)) <br/>
>> *The transformed test dataset with meta features.* 
>
> <br/>
>
> ***predict***(self, df_test)
>
> *Predict target on test dataset using the meta-features.*
>
>> **Parameters** 
>> ___ 
>> 
>> ***df_test*** : **pandas dataframe**, shape = (n_test, n_features) <br/>
>> *The test dataset with numerical features and no NA* 
>>
>> <br/>
>> 
>> **Returns** 
>> ___ 
>>
>> ***y*** : **array**, shape = (n_test, ) <br/>
>> *The predicted target.* 
>
> <br/>
>
> ***get_params***(self, deep=True)
>
> <br/>
>
> ***set_params***(self, params)
