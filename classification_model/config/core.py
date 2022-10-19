from pathlib import Path
from sklearn.pipeline import Pipeline
from pydantic import BaseModel

from strictyaml import YAML, load

from typing import List


import classification_model

# # Directories
PACKAGE_ROOT = Path(classification_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
DATASET_PATH = PACKAGE_ROOT / 'datasets'
CONFIG_FILE_PATH = PACKAGE_ROOT / 'config.yml'
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_model'


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    raw_data_file: str
    training_data_file: str
    testing_data_file: str
    replace_existing_datasets: bool
    dataset_source: str
    pipeline_name: str
    pipeline_save: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    numerical_variables: List[str]
    categorical_variables: List[str]
    cabin: List[str]
    target: str
    features: List[str]
    test_size: float
    random_state: int
    cat_stats: dict


class Config(BaseModel):
    """ Master config object"""

    app_config: AppConfig
    model_config: ModelConfig


def get_yaml_config_file() -> Path:
    """ Return config file path or raises error"""

    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    else:
        raise Exception('Config not found at: '.format(CONFIG_FILE_PATH))


def parse_yaml_config(yaml_config_dir: str = None) -> YAML:
    """Parses yaml config file"""

    if not yaml_config_dir:
        yaml_config_dir = get_yaml_config_file()
    
    if yaml_config_dir:
        with open(yaml_config_dir, 'r') as config_file:
            parsed_config = load(config_file.read())
            return parsed_config

    else:
        raise OSError(
            'Config file was not found at: {}'.format(yaml_config_dir)
            )


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Validates config data and return config object"""

    if not parsed_config:
        parsed_config = parse_yaml_config()
    
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data)
    )

    return _config


config = create_and_validate_config()