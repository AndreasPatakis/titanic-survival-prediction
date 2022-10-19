import typing as t

import pandas as pd
from classification_model.preprocessing.validations import validate_inputs
from classification_model.preprocessing.data_manager import load_pipeline
from classification_model.config.core import config
from classification_model import __version__ as _version

from sklearn.metrics import accuracy_score


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict]) -> dict:

    pipeline_name = "{}{}.pkl".format(config.app_config.pipeline_save, _version)
    _titanic_pipe = load_pipeline(file_name=pipeline_name)
    
    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input=data)
    results = {
        'predictions': None,
        'errors': errors,
        'version': _version
    }
    if not errors:
        predictions = _titanic_pipe.predict(validated_data[config.model_config.features])
        results['predictions'] = predictions

    return results
