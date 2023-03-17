from typing import List, Optional, Dict
from fastapi import APIRouter, HTTPException, status
from scripts.models import train_binary_cls_model

models_router = APIRouter()

# GET
@models_router.get(
    "/models/all",
    tags=["Models"],
    status_code=status.HTTP_200_OK,
    summary="Get all models available for consumption.",
)
def get_all_models():

    try:
        return "Hola mundo!"
    except HTTPException as H:
        raise H
    except Exception as E:
        raise E


# POST
@models_router.post(
    "/models/train_binary_classification_model",
    tags=["Models"],
    status_code=status.HTTP_200_OK,
    summary="Train a model with some config level.",
)
def train_binary_classification_model(
    input_data_filename_path: Optional[str] = None,
    features_filter: Optional[List] = None,
    model_name: Optional[str] = None,
    destination_storage_name: Optional[str] = None,
    model_custom_params: Optional[Dict] = None,
    grid_search_cv_params: Optional[Dict] = None,
    grid_search_cv: Optional[bool] = False,
    train_test_split_data: Optional[bool] = True,
    random_state: Optional[int] = None,
    shuffle_data: Optional[bool] = True,
    endpoint_test_mode: Optional[bool] = False,
):
    """Train a model with certain config, input data and a desired path to save it as a .pkl for its consumption.

    Args:

        input_data_filename_path (Optional[str]): Name of file to work with. Must include file extension. For example: input/my_own_file.csv. If None, is setted to /output/dataset_SCL_complete.csv.

        features_filter (Optional[List]): List of features existing into the input file to filter before training model. For example ['my_feature1', 'my_feature2']. If None, all file's features are going to be considered.

        model_name (Optional[str]): Name of the model to train. Must be one of the following: 'log-reg' (for LogisticRegression), 'xgb' (for XGBoost Classifier). If None, is setted to 'xgb'.

        destination_storage_name (Optional[str]): Output filename to save into src/models/ folder. If None, is setted to model_name + current_datetime in format '%Y-%m-%d-%H%M'.

        model_custom_params (Optional[Dict]): _description_

        grid_search_cv_params (Optional[Dict]): _description_

        grid_search_cv (Optional[bool], optional): _description_. Defaults to False.

        train_test_split_data (Optional[bool], optional): _description_. Defaults to True.

        random_state (Optional[int]): _description_. Defaults to None.

        shuffle_data (Optional[bool], optional): _description_. Defaults to True.

        endpoint_test_mode (Optional[bool], optional): If True, just take the first 1000 records in input file to train model. Defaults to False.
    """
    try:
        created = train_binary_cls_model(
            input_data_filename_path,
            features_filter,
            model_name,
            destination_storage_name,
            model_custom_params,
            grid_search_cv_params,
            grid_search_cv,
            train_test_split_data,
            random_state,
            shuffle_data,
            endpoint_test_mode,
        )
        return created
    except HTTPException as H:
        raise H
    except Exception as E:
        raise E
