**class Drift_thresholder** 
*Automatically deletes ids and drifting variables between train and test datasets.*
*Deletes on train and test datasets. The list of drift coefficients is available and saved as "drifts.txt"*

> **Parameters**
> ___
>  
>   ***threshold*** : **float** (between 0.5 and 1.), defaut = 0.9 
> *Threshold used to deletes variables and ids. The lower the more you keep non-drifting/stable variables.*

