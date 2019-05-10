from mlbox.preprocessing import Reader
from mlbox.preprocessing import Drift_thresholder
from mlbox.optimisation import Optimiser
from mlbox.prediction import Predictor

paths = ["train.csv", "test.csv"]
target_name = "Survived"

# Reading and cleaning all files
rd = Reader(sep=',')
df = rd.train_test_split(paths, target_name)

dft = Drift_thresholder()
df = dft.fit_transform(df)

# Tuning
opt = Optimiser(scoring='accuracy', n_folds=3)
opt.evaluate(None, df)

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

best = opt.optimise(space, df, 15)

prd = Predictor()
prd.fit_predict(best, df)
