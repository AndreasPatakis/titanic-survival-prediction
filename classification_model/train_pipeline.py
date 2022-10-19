# to divide train and test set
from email import header
from sklearn.model_selection import train_test_split

# import preprocessing functionalities
from preprocessing.data_manager import load_dataset, save_pipeline

# import preprocessing features
from preprocessing.features import configuring_data 

# import configuration
from classification_model.config.core import config, DATASET_PATH

# import pipeline
from pipeline import titanic_pipe


def train_model() -> None:
    """Trainning the model"""

    data = load_dataset(path=DATASET_PATH/config.app_config.raw_data_file)
    data = configuring_data(data=data)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],  # target
        test_size=config.model_config.test_size,  # percentage of obs in test set
        random_state=config.model_config.random_state  # seed to ensure reproducibility
    )

    # train the pipeline
    titanic_pipe.fit(X_train, y_train)    

    # save model
    save_pipeline(pipeline_to_save=titanic_pipe)


if __name__ == '__main__':
    train_model()
