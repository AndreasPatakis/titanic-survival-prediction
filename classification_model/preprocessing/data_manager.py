from typing import List
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

import joblib

from classification_model.config.core import config, DATASET_PATH, PACKAGE_ROOT, TRAINED_MODEL_DIR
from classification_model import __version__ as _version

def split_dataset(df: pd.DataFrame, test_set_perc: float = 0.1) -> pd.DataFrame:
    """ Splitting dataset to train and test sets"""

    df_size = len(df)
    test = df[:int(df_size*test_set_perc)]
    train = df[int(df_size*test_set_perc)+1:]

    return train, test


def path_exists(path: Path) -> bool:
    """Checks where the path given exists"""

    if path.exists():
        return True
    else:
        raise Exception("Datasets path was not found at: {}".format(path))


def clear_folder(path: Path = DATASET_PATH, to_keep: List = []) -> None:
    """Deleting all the unecessary files from folder"""

    if path_exists(path):
        to_keep.append('__init__.py')
        for file in path.iterdir():
            if file.name not in to_keep:
                file.unlink()


def dir_is_empty(path: Path = DATASET_PATH, ignore: List = []) -> bool:
    """ Checks whether a direcotry is empty"""

    ignore.append('__init__.py')
    if path_exists(path):
        files = [file for file in path.iterdir() if (file.is_file() and file.name not in ignore)]

    if not files:
        return True

    return False


def get_datasets(
    replace_existing: bool = config.app_config.replace_existing_datasets,
    source: str = config.app_config.dataset_source
    ) -> None:    
    """ Gets dataset from source and saves it to folder """

    if path_exists(DATASET_PATH):
        if replace_existing or dir_is_empty(DATASET_PATH):
            # clearing folder
            clear_folder()

            # loading data from source
            df = pd.DataFrame()
            df = pd.read_csv(source)
            train, test = split_dataset(df)
            
            df.to_csv(DATASET_PATH/config.app_config.raw_data_file, header=True, index=False)
            train.to_csv(DATASET_PATH/config.app_config.training_data_file, header=True, index=False)
            test.to_csv(DATASET_PATH/config.app_config.testing_data_file, header=True, index=False)


def load_dataset(*, path: Path = DATASET_PATH/config.app_config.training_data_file) -> pd.DataFrame:
    """ Loads dataset csv and returns it as a dataframe"""

    # Making sure that we have our datasets
    get_datasets()
    data = pd.read_csv(path)

    return data


def save_pipeline(*, pipeline_to_save: Pipeline) -> None:
    """Saves trained pipeline while replacing the old one"""

    file_name = "{}{}.pkl".format(config.app_config.pipeline_save, _version)
    save_path = TRAINED_MODEL_DIR/file_name

    # Clears trained_model folder
    clear_folder(path=PACKAGE_ROOT/'trained_model')
    joblib.dump(pipeline_to_save, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """ Loads the saved pipeline"""

    path = TRAINED_MODEL_DIR/file_name
    _pipeline = joblib.load(filename=path)
    return _pipeline


