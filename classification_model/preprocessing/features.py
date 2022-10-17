# to handle datasets
import pandas as pd
import numpy as np

import re

from sklearn.base import BaseEstimator, TransformerMixin


class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    # Extract fist letter of variable

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables        

    def fit(self, X, y=None):
        """needs fit statement for sklearn pipeline """
        return self

    def transform(self, X, y=None):
        """Keeps only the first letter of the value"""
        def strip_letter(row):
            try:
                return row[0]
            except TypeError:
                return np.nan

        for var in self.variables:
            X[var] = X[var].apply(strip_letter)
        
        return X


class OneHotEncoding(BaseEstimator, TransformerMixin):
    # One hot encode categorical variables

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables

    def fit(self, X, y=None):
        """needs fit statement for sklearn pipeline """

        return self

    def transform(self, X, y=None):
        """ Performs one hot encoding and drop the variable column"""
        def add_cat_column(X, var):
            X = X.copy()
            for label in X[var].unique():
                X[label] = np.where(X[var] == label, 1, 0)
            
            X.drop([var], axis=1, inplace=True)

            return X
        
        for var in self.variables:
            X = add_cat_column(X, var)
        
        return X


def configuring_data(*, data: pd.DataFrame) -> pd.DataFrame:
    """ Performs necessary transformations to dataset"""

    def get_first_cabin(row):
        """ retain only the first cabin if more than
        1 are available per passenger"""

        try:
            return row[0]
        except TypeError:
            return np.nan

    def get_title(passenger):
        """ Extracts the title (Mr, Ms, etc) from the name variable"""
        line = passenger
        if re.search('Mrs', line):
            return 'Mrs'
        elif re.search('Mr', line):
            return 'Mr'
        elif re.search('Miss', line):
            return 'Miss'
        elif re.search('Master', line):
            return 'Master'
        else:
            return 'Other'

    # replace interrogation marks by NaN values
    data = data.replace('?', np.nan)

    data['cabin'] = data['cabin'].astype('string')
    data['cabin'] = data['cabin'].apply(get_first_cabin)

    data['title'] = data['name'].apply(get_title)

    # cast numerical variables as floats
    data['fare'] = data['fare'].astype('float')
    data['age'] = data['age'].astype('float')

    # drop unnecessary variables
    data.drop(labels=['name', 'ticket', 'boat', 'body', 'home.dest'], axis=1, inplace=True)

    return data

