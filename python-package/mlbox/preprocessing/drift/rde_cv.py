# coding: utf-8
# Authors: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
#          Alexis BONDU <alexis.bondu@gmail.com>
# License: BSD 3 clause

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from .drift_threshold import DriftThreshold
from ipywidgets import IntProgress, HTML, VBox
from IPython.display import display


def log_progress(sequence, every=None, size=None):

    """
    Returns a log progress bar for computations

    """

    is_iterator = False

    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = size / 200  # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)

    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{index} / ?'.format(index=index)
                else:
                    progress.value = index
                    label.value = u'{index} / {size}'.format(
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = str(index or '?')


class RDECV():

    def __init__(self,
                 estimator=None,
                 scoring=None,
                 delta_score=0.1,
                 cv=None,
                 drifts=None,
                 max_features=1.,
                 verboseMode=True):

        '''Recursive Drift Elimination

        Recursive Drift Elimination algorithm that performs a robust feature selection using a cross-validation.
        Variables kept are those that optimize the mean of the scoring function across the folds and that are stables (low drift)


        Parameters
        ----------
        estimator : estimator, defaut = None
            The estimator which will be deteriorated in order to reduce the drift between train and test datasets (classifier or regressor !)

        scoring : sklearn metric function, defaut = None

        delta_score : float, default = 0.1
            The allowed decrease of the quality of the estimator.
            Must be between 0. and 1.

        cv : cv object, default = None
            If none two folds are used

        drifts : dict, defaut = None
            The dictionnary of the drifts coefs for each variables.
            If None, the univariate drifts are estimated with default parameters of the class DriftEstimator

        max_features : int or float, default = 1.
            The proportion / or the number of features to process

        verboseMode : bool, default = True
            If true, messages are displayed during the algorithm
        '''

        # For the fit method

        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.delta_score = np.abs(delta_score)
        self.drifts = drifts
        self.max_features = max_features
        self.verboseMode = verboseMode
        # Scores obtained over the winning iterations
        # Scores higher than the fixed limit
        self.__scores = None
        self.__dropList = None  # variables to drop
        self.__keepList = None  # variables to keep
        self.__fitOK = False

    def get_params(self):

        return {'estimator': self.estimator,
                'cv': self.cv,
                'scoring': self.scoring,
                'delta_score': self.delta_score,
                'drifts': self.drifts,
                'max_features': self.max_features,
                'verboseMode': self.verboseMode}

    def set_params(self, **params):

        if('estimator' in params.keys()):
            self.estimator = params['estimator']
        if('cv' in params.keys()):
            self.cv = params['cv']
        if('scoring' in params.keys()):
            self.scoring = params['scoring']
        if('delta_score' in params.keys()):
            self.delta_score = params['delta_score']
        if('drifts' in params.keys()):
            self.drifts = params['drifts']
        if('max_features' in params.keys()):
            self.max_features = params['max_features']
        if('verboseMode' in params.keys()):
            self.verboseMode = params['verboseMode']

    def fit(self, df_train, df_test, y_train):

        '''Fits RDECV algorithm.

        Parameters
        ----------
        df_train : pandas dataframe of shape = (n_train, p)
            Train set

        df_test : pandas dataframe of shape = (n_test, p)
            Test set

        y_train : target
            Target for train set

        Returns
        -------
        self : object
            Returns self.
        '''

        if (self.estimator is None):
            raise ValueError('No estimator defined !')

        if (self.scoring is None):
            raise ValueError('You must specify the scoring function !')

        if (self.cv is None):
            self.cv = KFold(n_splits=2, shuffle=True, random_state=1)
            print('Warning : cv is not defined. 2 folds are used by default.')

        # If drift levels are not indicated
        if (self.drifts is None):
            print('Warning : drift coeffs are not defined. '
                  'Let\'s compute them ...')

            de = DriftThreshold()
            de.fit(df_train, df_test)
            self.drifts = de.drifts()
            del de

        # Compute initial score
        print("")
        print('Computing initial score :')
        print('-------------------------')
        print ("")

        self.__scores = []
        self.__scores.append(np.mean(cross_val_score(estimator=self.estimator,
                                                     X=df_train,
                                                     y=y_train,
                                                     scoring=self.scoring,
                                                     cv=self.cv)))

        # Init
        sorted_drifts = sorted(self.drifts.items(),
                               key=lambda x: x[1],
                               reverse=True)

        col_drifts = []  # Contains column names with decreasing drift order
        for i in range(len(sorted_drifts)):
            col_drifts.append(sorted_drifts[i][0])

        if (type(self.max_features) == int):
            # Number maximum of features to deal with
            idEndLoop = min(self.max_features, len(col_drifts))
        else:
            idEndLoop = min(int(self.max_features * len(col_drifts)),
                            len(col_drifts))

        # Verbose

        if self.verboseMode:
            print('initial score (with all variables) : '+str(self.__scores[0])
                  )

            if self.__scores[0] > 0:
                limitScore = self.__scores[0] * (1. - self.delta_score)
            else:
                limitScore = self.__scores[0] * (1. + self.delta_score)

            print('limit score : ' + str(limitScore))

            print("")
            print('RDECV algorithm is starting : (' + str(idEndLoop) + ' '
                  'variables processed )')
            print('----------------------------')
            print("")

        # Run Algorithm

        self.__dropList = []
        self.__keepList = []

        countRemoveVar = 0

        for col in log_progress(col_drifts[:idEndLoop]):

            self.__dropList.append(col)
            self.__keepList.append(col)

            X_tmp = df_train.drop(self.__dropList, axis=1)

            currentScore = np.mean(cross_val_score(estimator=self.estimator,
                                                   X=X_tmp,
                                                   y=y_train,
                                                   scoring=self.scoring,
                                                   cv=self.cv)
                                   )

            # Degradation Budget (limitScore) of the model reached?

            if currentScore < limitScore:
                del self.__dropList[-1]  # Put back the column in the dataset

                if self.verboseMode:
                    print('    CAN\'T BE REMOVED  - '
                          'Removing variable ' + '\'' + str(col) + '\'' + ''
                          'deteriorates the model with '
                          'the too bad score : ' + str(currentScore))

            else:
                self.__scores.append(currentScore)  # Column not kept
                del self.__keepList[-1]
                countRemoveVar += 1

                if self.verboseMode:
                    print(str(countRemoveVar) + ' removed variables - '
                          'Removing variable' + '\'' + str(col) + '\' is OK '
                          'with the score : ' + str(currentScore))

        self.__keepList = self.__keepList + col_drifts[idEndLoop:]
        self.__fitOK = True

        return self

    def transform(self, df):

        """Select the features with low drift and high predictive information.

        Parameters
        ----------
        df : pandas dataframe
            A DataFrame with the same features

        Returns
        -------
        pandas DataFrame
            The transformed dataset
        """

        if self.__fitOK:

            return df[self.get_support()]

        else:
            raise ValueError('Call the fit function before !')

    def get_loss(self, absolute=False):

        """Final score obtained at the end of the RDECV algorithm

        The decrease of the initial score is limited by the parameter \'delta_score\'

        Parameters
        ----------
        absolute : bool, default = False
            If False, the relative decrease is returned

        Returns
        -------
        float
            The final score
        """


        if self.__fitOK:

            if(absolute):

                return self.__scores[-1]

            else:
                numerator = np.abs(self.__scores[0] - self.__scores[-1])
                denominator = (np.abs(self.__scores[0]) + 1e-15)
                return numerator / denominator

        else:
            raise ValueError('Call the fit function before !')

    def get_support(self, complement = False):

        """Returns the variables kept or dropped.

        Parameters
        ----------
        complement : bool, default = True
            If True, returns the features to drop
            If False, returns the features to keep

        Returns
        -------
        list
            The list of features to keep or to drop.
        """

        if self.__fitOK:

            if(complement):
                return self.__dropList

            else:
                return self.__keepList

        else:
            raise ValueError('Call the fit function before !')

    def residual_drifts(self, complement = False):

        '''
        Returns the univariate drifts for the variables kept or dropped.

        Parameters
        ----------
        complement : bool, default = False
            If True, returns the drifts for the features to drop
            If False, returns the drifts for the features to keep

        Returns
        -------
        dict
            The dictionnary of drifts for the kept or dropped features.

        '''

        if self.__fitOK:

            results = dict()

            if(complement):
                for col in self.__dropList:
                    results[col] = self.drifts[col]

            else:
                for col in self.__keepList:
                    results[col] = self.drifts[col]

            return sorted(results.items(), key=lambda x: x[1], reverse=True)

        else:
            raise ValueError('Call the fit function before !')
