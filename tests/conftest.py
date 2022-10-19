import pandas as pd

from classification_model.config.core import config, DATASET_PATH
from classification_model.preprocessing.data_manager import load_dataset

from sklearn.model_selection import train_test_split

import pytest


@pytest.fixture
def test_data() -> pd.DataFrame:
    """ Returns a test dataframe"""

    data_df = load_dataset(
                    path=DATASET_PATH/config.app_config.training_data_file
                )

    _, X_test, _, _ = train_test_split(
        data_df.drop(columns=config.model_config.target),
        data_df[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state
    )

    return X_test
