import numpy as np

from sklearn.metrics import accuracy_score

from classification_model.predict import make_prediction
from classification_model.config.core import config, TRAINED_MODEL_DIR, DATASET_PATH
from classification_model.preprocessing.data_manager import load_dataset

from classification_model import __version__ as _version


def test_trained_pipeline_exists() -> None:
    """ Checks that the pickle file of the trained modle exists"""

    files = [f.name for f in TRAINED_MODEL_DIR.iterdir() if f.is_file()]
    pipeline_file = f'{config.app_config.pipeline_save}{_version}.pkl'
    assert pipeline_file in files


def test_prediciton() -> None:
    """ Checks validity and accuracy of results"""

    test_data = load_dataset(path=DATASET_PATH/config.app_config.raw_data_file)
    test_data = test_data.sample(frac=1)[:int(0.1*len(test_data))]
    expected_no_predicitons = len(test_data)
    results = make_prediction(input_data=test_data)
    # Check that the result dict returned 3 keys
    assert len(results.keys()) == 3
    # Check that there where no errors
    assert results.get('errors') == None
    # Check number and type of results are correct
    predictions = results.get('predictions')
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.int64)
    assert len(predictions) == expected_no_predicitons
    # Check accuracy
    y_pred = list(predictions)
    y_true = test_data[config.model_config.target]
    score = accuracy_score(y_true=y_true, y_pred=y_pred)
    assert score > 0.65
    

