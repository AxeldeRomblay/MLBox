Help on class NA_encoder in module mlbox.encoding.na_encoder:

class NA_encoder
 |  Encodes missing values for both numerical and categorical features. Several strategies are possible in each case.
 |  
 |  
 |  Parameters
 |  ----------
 |  
 |  numerical_strategy : string or float or int, defaut = "mean"
 |      The strategy to encode NA for numerical features. 
 |      Available strategies = "mean", "median", "most_frequent" or a float/int value
 |  
 |  categorical_strategy : string, defaut = '<NULL>'
 |      The strategy to encode NA for categorical features. 
 |      Available strategies = a string or np.NaN
 |  
 |  Methods defined here:
 |  
 |  __init__(self, numerical_strategy='mean', categorical_strategy='<NULL>')
 |  
 |  fit(self, df_train, y_train=None)
 |      Fits NA Encoder.
 |      
 |      Parameters
 |      ----------
 |      
 |      df_train : pandas dataframe of shape = (n_train, n_features)
 |      The train dataset with numerical and categorical features. 
 |      
 |      y_train : [OPTIONAL]. pandas series of shape = (n_train, ). defaut = None
 |      The target for classification or regression tasks.
 |              
 |      
 |      Returns
 |      -------
 |      None
 |  
 |  fit_transform(self, df_train, y_train=None)
 |      Fits NA Encoder and transforms the dataset.
 |      
 |      Parameters
 |      ----------
 |      
 |      df_train : pandas dataframe of shape = (n_train, n_features)
 |      The train dataset with numerical and categorical features. 
 |      
 |      y_train : [OPTIONAL]. pandas series of shape = (n_train, ). defaut = None
 |      The target for classification or regression tasks.
 |              
 |      
 |      Returns
 |      -------
 |      
 |      df_train : pandas dataframe of shape = (n_train, n_features)
 |      The train dataset with no missing values.
 |  
 |  get_params(self, deep=True)
 |  
 |  set_params(self, **params)
 |  
 |  transform(self, df)
 |      Transforms the dataset
 |      
 |      Parameters
 |      ----------
 |      
 |      df : pandas dataframe of shape = (n, n_features)
 |      The dataset with numerical and categorical features. 
 |              
 |      
 |      Returns
 |      -------
 |      
 |      df : pandas dataframe of shape = (n, n_features)
 |      The dataset with no missing values.

class Categorical_encoder
 |  Encodes categorical features. Several strategies are possible (supervised or not). Works for both classification and regression tasks.
 |  
 |  
 |  Parameters
 |  ----------
 |  
 |  strategy : string, defaut = "label_encoding"
 |      The strategy to encode categorical features.
 |      Available strategies = "label_encoding", "dummification", "random_projection", entity_embedding"
 |  
 |  verbose : boolean, defaut = False
 |      Verbose mode. Useful for entity embedding strategy.
 |  
 |  Methods defined here:
 |  
 |  __init__(self, strategy='label_encoding', verbose=False)
 |  
 |  fit(self, df_train, y_train)
 |      Fits Categorical Encoder.
 |      
 |      Parameters
 |      ----------
 |      
 |      df_train : pandas dataframe of shape = (n_train, n_features)
 |      The train dataset with numerical and categorical features. NA values are allowed.
 |      
 |      y_train : pandas series of shape = (n_train, ).
 |      The target for classification or regression tasks.
 |      
 |      
 |      Returns
 |      -------
 |      None
 |  
 |  fit_transform(self, df_train, y_train)
 |      Fits Categorical Encoder and transforms the dataset
 |      
 |      Parameters
 |      ----------
 |      
 |      df_train : pandas dataframe of shape = (n_train, n_features)
 |      The train dataset with numerical and categorical features. NA values are allowed.
 |      
 |      y_train : pandas series of shape = (n_train, ).
 |      The target for classification or regression tasks.
 |      
 |      
 |      Returns
 |      -------
 |      
 |      df_train : pandas dataframe of shape = (n_train, n_features)
 |      The train dataset with numerical and encoded categorical features.
 |  
 |  get_params(self, deep=True)
 |  
 |  set_params(self, **params)
 |  
 |  transform(self, df)
 |      Transforms the dataset
 |      
 |      Parameters
 |      ----------
 |      
 |      df : pandas dataframe of shape = (n, n_features)
 |      The dataset with numerical and categorical features. NA values are allowed.
 |      
 |      
 |      Returns
 |      -------
 |      
 |      df : pandas dataframe of shape = (n, n_features)
 |      The dataset with numerical and encoded categorical features.
