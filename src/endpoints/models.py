from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from schemas.models import PredictionSchema, TrainBinaryClassificationSchema
from scripts.models import (
    get_model_filenames,
    make_predictions,
    train_binary_cls_model,
    model_classification_report,
)

models_router = APIRouter()

# GET
@models_router.get(
    "/models/all_models",
    tags=["Models"],
    status_code=status.HTTP_200_OK,
    summary="Get all models available for consumption.",
)
def all_models():
    """Given the models directory: src/models, this endpoint returns a list of model filenames available for use.

    Returns:
        List: model filenames. For example: ["my_model1.pkl", "my_model2.pkl"]
    """
    try:
        files = get_model_filenames()
        return files
    except HTTPException as H:
        raise H
    except Exception as E:
        raise E


# GET
@models_router.get(
    "/models/classification_report",
    tags=["Models"],
    status_code=status.HTTP_200_OK,
    summary="Get all models available for consumption.",
)
def classification_report(y_real_filename: str, y_predicted_filename: str):
    """Given a model of directory 'src/models', this endpoint returns some metrics evaluating that model with a given y_train and y_test batches from 'src/files/output'.

    Returns:
        List: model filenames. For example: ["my_model1.pkl", "my_model2.pkl"]
    """
    try:
        report = model_classification_report(y_real_filename, y_predicted_filename)
        return JSONResponse(status_code=status.HTTP_200_OK, content=report)
    except HTTPException as H:
        raise H
    except Exception as E:
        raise E


# POST
@models_router.post(
    "/models/predict",
    tags=["Models"],
    status_code=status.HTTP_200_OK,
    summary="Given a pre-trained model, use it for classify each line of the input data and predict if the flight is probable to be delayed or not.",
    response_model=PredictionSchema,
)
def predicti(request_body: PredictionSchema):
    """Given one of the models in the directory src/models, this endpoint writes a .csv file with predictions as src/files/output/{current_datetime}--{model_filename}--predictions.csv.

    Args:
        model_filename (str): One of the available models in directory src/models/.

        input_file_name (Optional[str], optional): For multiple records prediction purposes, pass here the file name to take as input for making predictions. It must exist in folder 'src/files/input/'. Just one of input_path_file or input_row must be passed. For example: 'test_batch.csv'. Defaults to None.

        input_row (Optional[Dict], optional): For individual prediction purposes, pass here the dictionary as input for making predictions. Just one of input_path_file or input_row must be passed. For example: {"OPERA_Aerolineas Argentinas": 0, "OPERA_Aeromexico": 0, "OPERA_Air Canada": 0, "OPERA_Air France": 0, "OPERA_Alitalia": 0, "OPERA_American Airlines": 0, "OPERA_Austral": 0, "OPERA_Avianca": 0, "OPERA_British Airways": 0, "OPERA_Copa Air": 0, "OPERA_Delta Air": 0, "OPERA_Gol Trans": 0, "OPERA_Grupo LATAM": 1, "OPERA_Iberia": 0, "OPERA_JetSmart SPA": 0, "OPERA_K.L.M.": 0, "OPERA_Lacsa": 0, "OPERA_Latin American Wings": 0, "OPERA_Oceanair Linhas Aereas": 0, "OPERA_Plus Ultra Lineas Aereas": 0, "OPERA_Qantas Airways": 0, "OPERA_Sky Airline": 0, "OPERA_United Airlines": 0, "TIPOVUELO_I": 1, "TIPOVUELO_N": 0, "MES_1": 0, "MES_2": 0, "MES_3": 0, "MES_4": 0, "MES_5": 0, "MES_6": 0, "MES_7": 0, "MES_8": 0, "MES_9": 1, "MES_10": 0, "MES_11": 0, "MES_12": 0}. Defaults to None.

    Returns:
       status: 201 if the ...predictions.csv file was created succesfully into src/files/output/.
    """
    try:
        predictions = make_predictions(
            request_body.model_filename,
            request_body.X_test_filename,
            request_body.categorical_features,
            request_body.numerical_features,
            request_body.minmax_scaler_numerical_f,
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=predictions)
    except HTTPException as H:
        raise H
    except Exception as E:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{E}")


# POST
@models_router.post(
    "/models/train_binary_classification_model",
    tags=["Models"],
    status_code=status.HTTP_200_OK,
    summary="Train a model with some config level.",
    response_model=TrainBinaryClassificationSchema,
)
def train_binary_classification_model(request_body: TrainBinaryClassificationSchema):
    """Train a model with certain config, input data and a desired path to save it as a .pkl for its consumption.

    Args:

        input_data_filename_path (Optional[str]): Name of file to work with. Must include file extension. For example: input/my_own_file.csv. If None, is setted to /output/dataset_SCL_complete.csv.

        features_filter (Optional[List]): List of features existing into the input file to filter before training model.
        For example ['my_feature1', 'my_feature2']. If None, all file's features are going to be considered.

        categorical_features (Optional[List]) .Default to None.
        numerical_features (Optional[List]) .Default to None.
        minmax_scaler_numerical_f (Optional[bool]) .Default to False.
        label (Optional[bool]) .Default to None.

        model_name (Optional[str]): Name of the model to train. Must be one of the following: 'log-reg' (for LogisticRegression),
        'xgb' (for XGBoost Classifier). If None, is setted to 'xgb'.

        destination_storage_name (Optional[str]): Output filename to save into src/models/ folder.
        For example 'XGB_random_state10_learning_rate1-4'. If None, is setted to model_name + current_datetime in format '%Y-%m-%d-%H%M'.

        model_custom_params (Optional[Dict]): Custom parameters for selected model_name.
        For example for 'log-reg' model: {"penalty":"l2", "dual": True}. If None, model is going to train with its own default parameters.

        grid_search_cv (Optional[bool], optional): If True, selected model is going to be optimized by GridSearchCV function of
        scikit-learn. Likewise, grid_search_cv_params must be given if grid_search_cv is True. Defaults to False.

        grid_search_cv_params (Optional[Dict]): Dictionary of settings for GridSearchCV hiper-parameter optimization.If grid_search_cv is True, this field is mandatory.
        For example: {param_grid: {"learning_rate": [0.01,0.05, 0.1],"n_estimators": [50, 100, 150],"subsample": [0.5, 0.9]},"cv": 2,"n_jobs": -1,"verbose": 1}. Defaults to None.

        train_test_split_data (Optional[bool], optional): If input data must be splitted in train-test batches (66% - 33% respectively)
        for model training or if not.Defaults to True.

        random_state (Optional[int]): For reproducible runs. Defaults to None.

        shuffle_data (Optional[bool], optional): If input data as pd.DataFrame must be shuffled (sklearn.utils.shuffle: Shuffle
        arrays or sparse matrices in a consistent way.). Defaults to True.

        shuffle_features (Optional[List]) .Default to None.

        endpoint_test_mode (Optional[bool], optional): If True, just take the first 1000 records in input file to train model.
        This option helps for faster tests of the endpoint. Defaults to False.

    Returns:
       status: 201 if the model.pkl file was created succesfully into src/models/.
    """
    try:
        created = train_binary_cls_model(
            request_body.X_train_filename,
            request_body.y_train_filename,
            request_body.model_name,
            request_body.destination_model_name,
            dict(request_body.model_custom_params),
            request_body.grid_search_cv,
            dict(request_body.grid_search_cv_params),
            request_body.random_state,
        )
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=created)
    except HTTPException as H:
        raise H
    except Exception as E:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{E}")
