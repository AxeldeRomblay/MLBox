"""A classification example using mlbox."""
from mlbox.preprocessing import Reader
from mlbox.preprocessing import Drift_thresholder
from mlbox.optimisation import Optimiser
from mlbox.prediction import Predictor

# Paths to the train set and the test set.
paths = ["train_classification.csv", "test_classification.csv"]
# Name of the feature to predict.
# This columns should only be present in the train set.
target_name = "Survived"

# Reading and cleaning all files
# Declare a reader for csv files
rd = Reader(sep=',')
# Return a dictionnary containing three entries
# dict["train"] contains training samples withtout target columns
# dict["test"] contains testing elements withtout target columns
# dict["target"] contains target columns for training samples.
dict = rd.train_test_split(paths, target_name)

dft = Drift_thresholder()
dict = dft.fit_transform(dict)

# Tuning
# Declare an optimiser. Scoring possibilities for classification lie in :
# {"accuracy", "roc_auc", "f1", "neg_log_loss", "precision", "recall"}
opt = Optimiser(scoring='accuracy', n_folds=3)
opt.evaluate(None, dict)

# Space of hyperparameters
# The keys must respect the following syntax : "enc__param".
#   "enc" = "ne" for na encoder
#   "enc" = "ce" for categorical encoder
#   "enc" = "fs" for feature selector [OPTIONAL]
#   "enc" = "stck"+str(i) to add layer n°i of meta-features [OPTIONAL]
#   "enc" = "est" for the final estimator
#   "param" : a correct associated parameter for each step.
#   Ex: "max_depth" for "enc"="est", ...
# The values must respect the syntax: {"search":strategy,"space":list}
#   "strategy" = "choice" or "uniform". Default = "choice"
#   list : a list of values to be tested if strategy="choice".
#   Else, list = [value_min, value_max].
# Available strategies for ne_numerical_strategy are either an integer, a float
#   or in {'mean', 'median', "most_frequent"}
# Available strategies for ce_strategy are:
#   {"label_encoding", "dummification", "random_projection", entity_embedding"}
space = {'ne__numerical_strategy': {"search": "choice", "space": [0]},
         'ce__strategy': {"search": "choice",
                          "space": ["label_encoding",
                                    "random_projection",
                                    "entity_embedding"]},
         'fs__threshold': {"search": "uniform",
                           "space": [0.01, 0.3]},
         'est__max_depth': {"search": "choice",
                            "space": [3, 4, 5, 6, 7]}

         }

# Optimises hyper-parameters of the whole Pipeline with a given scoring
# function. Algorithm used to optimize : Tree Parzen Estimator.
#
# IMPORTANT : Try to avoid dependent parameters and to set one feature
# selection strategy and one estimator strategy at a time.
best = opt.optimise(space, dict, 15)

# Make prediction and save the results in save folder.
prd = Predictor()
prd.fit_predict(best, dict)
