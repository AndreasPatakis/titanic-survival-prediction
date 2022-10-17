#pipeline
from sklearn.pipeline import Pipeline

# feature scaling
from sklearn.preprocessing import StandardScaler

# to build the models
from sklearn.linear_model import LogisticRegression

# for imputation
from feature_engine.imputation import (
    AddMissingIndicator,
    MeanMedianImputer,
    CategoricalImputer
)

from feature_engine.encoding import RareLabelEncoder

# import preprocessing features
from preprocessing import features

# import configuration
from classification_model.config.core import config

# set up the pipeline
titanic_pipe = Pipeline([

    # ===== IMPUTATION =====
    # impute categorical variables with string 'missing'
    ('categorical_imputation', CategoricalImputer(imputation_method='missing', variables=config.model_config.categorical_variables)),

    # add missing indicator to numerical variables
    ('missing_indicator', AddMissingIndicator(variables=config.model_config.numerical_variables)),

    # impute numerical variables with the median
    ('median_imputation', MeanMedianImputer(imputation_method='mean', variables=config.model_config.numerical_variables)),


    # Extract first letter from cabin
    ('extract_letter', features.ExtractLetterTransformer(variables=config.model_config.cabin)),


    # == CATEGORICAL ENCODING ======
    # remove categories present in less than 5% of the observations (0.05)
    # group them in one category called 'Rare'
    ('rare_label_encoder', RareLabelEncoder(tol=0.1, n_categories=1, variables=config.model_config.categorical_variables)),


    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder', features.OneHotEncoding(variables=config.model_config.categorical_variables)),

    # scale using standardization
    ('scaler', StandardScaler()),

    # logistic regression (use C=0.0005 and random_state=0)
    ('Logistic', LogisticRegression(C=0.0005, random_state=0)),
])