import pandas as pd

from classification_model.config.core import config
from classification_model.preprocessing import features


def test_extract_letter_transformer(test_data) -> None:
    """ Testing letter extraction transformer"""

    var = config.model_config.cabin
    test_data = test_data[~(test_data[var].isnull())
                          & ~(test_data[var] == '?')]
    transformer = features.ExtractLetterTransformer(variables=var)
    processed_data = transformer.fit_transform(test_data)
    assert isinstance(processed_data, pd.DataFrame)
    assert processed_data[processed_data[var[0]].str.len() > 1].empty


def test_configuring_data(test_data) -> None:
    """ Testing the configuring data function"""

    test_data = features.configuring_data(data=test_data)
    # Check return type
    assert isinstance(test_data, pd.DataFrame)
    # Check if question marks are replaced
    for var in test_data.columns:
        assert test_data[test_data[var] == '?'].empty
    # Check cabin data type
    assert test_data['cabin'].dtype == 'O'
    # Check if only the first cabin is kept
    test_data = test_data[test_data['cabin'].notna()]
    assert test_data[test_data['cabin'].str.contains(pat=' ')].empty
    # Check if 'title' feature exists
    assert 'title' in test_data.columns
    # Check data types
    assert test_data['fare'].dtype == 'float'
    assert test_data['fare'].dtype == 'float'
    # Check if only our selected features are left
    assert all(
        [False for feature in test_data.columns
         if feature not in config.model_config.features]
    )


def test_one_hot_encoding(test_data) -> None:
    """ Testing categorical one hot encoding function"""

    test_data = features.configuring_data(data=test_data)
    test_data.dropna(inplace=True)
    cat_vars = config.model_config.categorical_variables
    all_cat_labels = [label for var in cat_vars
                      for label in test_data[var].unique()]
    # Checks that the our final categorical variables exist at the dataframe
    assert all([False for var in cat_vars if var not in test_data.columns])
    transformer = features.OneHotEncoding(cat_vars)
    test_data = transformer.fit_transform(test_data)
    # Check that previous categorical varibales are dropped
    assert all([False for var in test_data.columns if var in cat_vars])
    # Check that all catgeorical labels have become columns
    assert all(
        [False for var in all_cat_labels
         if var not in test_data.columns]
    )
