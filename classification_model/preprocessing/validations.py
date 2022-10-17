from classification_model.config.core import config
from classification_model.preprocessing.features import configuring_data

from pydantic import BaseModel, ValidationError
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np


class InputModelSchema(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    cabin: str
    embarked: str
    title: str


class MultipleModelInputs(BaseModel):
    inputs: List[InputModelSchema]


def validate_inputs(*, input: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:

    processed_data = configuring_data(data=input)
    validated_data = processed_data[config.model_config.features].copy()
    errors = None
    try:
        MultipleModelInputs(
            inputs=validated_data.replace(np.nan, None).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json
    
    return validated_data, errors
