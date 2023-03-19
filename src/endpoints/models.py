from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from schemas.models import PredictionSchema, TrainBinaryClassificationSchema
from scripts.models import (
    get_model_filenames,
    delete_modelname,
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
    """Given the models directory: 'src/models', this endpoint returns a list of model filenames available for use.

    **Returns:**
        List: Model filenames. For example: ["my_model1.pkl", "my_model2.pkl"]
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
    summary="Given a real label and a predicted label, get som metrics evaluating performance of the predictions.",
)
def classification_report(y_real_filename: str, y_predicted_filename: str):
    """Returns some metrics evaluating a given y_train and y_test stored in 'src/files/output'.

    **Args:**
        y_real_filename (str): File name stored in 'src/files/output' with label real values.
        y_predicted_filename (str): File name stored in 'src/files/output' with predicte predicted values.

    **Returns:**
        JSONResponse: With Keys: 'accuracy', 'recall', 'f1_score', 'roc_auc_score' and 'data'.
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
    "/models/train_binary_classification_model",
    tags=["Models"],
    status_code=status.HTTP_200_OK,
    summary="Train a model with some config level and data.",
    response_model=TrainBinaryClassificationSchema,
)
def train_binary_classification_model(request_body: TrainBinaryClassificationSchema):
    """Train a model with certain config, input data and a desired path to save it as a .pkl for its consumption.

    **Args:**
        X_train_filename (str): File name with train features stored in 'src/files/output/'.
        y_train_filename (str): File name with train label stored in 'src/files/output/'.
        model_name (str, optional): Name of the model to train. Must be one of the following: 'log-reg' (for LogisticRegression), 'xgb' (for XGBoost Classifier). If None, is setted to 'xgb' by default.
        destination_model_name (str, optional): How is going to be saved the model into 'src/models/' folder.
            For example 'My_XGB_Classifier_model'. Default to {model_name}_{YYYY-MM-DD-HHMM}.
        model_custom_params (Dict, optional): Custom parameters for selected model_name.
            For example for 'log-reg' model: {"penalty":"l2", "dual": True}. If None, model is going to train with default parameters.
        grid_search_cv (bool, optional): If True, selected model is going to be optimized by GridSearchCV function of
            scikit-learn. Likewise, grid_search_cv_params must be given if grid_search_cv is True. Defaults to False.
        grid_search_cv_params (Dict, optional): Dictionary of settings for GridSearchCV hiper-parameter optimization. If grid_search_cv is True, this field is mandatory.
            For example: {param_grid: {"learning_rate": [0.01,0.05, 0.1],"n_estimators": [50, 100, 150],"subsample": [0.5, 0.9]},"cv": 2,"n_jobs": -1,"verbose": 1}. Defaults to None.
        random_state (int): Random state for code replication. If None, results are expected to be random. Defaults to None.

    **Returns:**
       JSONResponse: With message of success creation of model.
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


# POST
@models_router.post(
    "/models/predict",
    tags=["Models"],
    status_code=status.HTTP_200_OK,
    summary="Given a pre-trained model stored in 'src/models/' and data for make predictions, use it for classify each line of the input data and predict if the flight is probable to be delayed or not.",
    response_model=PredictionSchema,
)
def predict(request_body: PredictionSchema):
    """Given one of the models in the directory 'src/models', this endpoint writes a .csv file with predictions as 'src/files/output/{model_filename}-predictions.csv'.

    **Args:**
        model_filename (str): One of the available models in directory 'src/models/'.
        X_test_filename (str): File name with features that could be used for making predictions with the selected model. Must be stored in 'src/files/output/'. For example: 'dataset_SCL_complete-X_test.csv'.
        categorical_features (List, optional): List of features to be treated as categorical ones and do **One Hot Encoding** with it. Defaults to None.
        numerical_features (List, optional): List of features to be treated as numerical ones and do **MinMax Scaling** with it ONLY if arg minmax_scaler_numerical_f is True. Defaults to None.
        minmax_scaler_numerical_f (bool, optional): For numerical_features, apply a MinMax Scaling. Defaults to False.

    **Returns:**
       JSONResponse: Of predictions, for example: [0,1,1,0].
    """
    try:
        predictions = make_predictions(
            request_body.model_filename,
            request_body.X_test_filename,
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=predictions)
    except HTTPException as H:
        raise H
    except Exception as E:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{E}")


# DELETE
@models_router.delete(
    "/models/delete_by_name",
    tags=["Models"],
    status_code=status.HTTP_200_OK,
    summary="Delete a given model of the 'src/models/' repo's directory.",
)
def delete_by_name(path: str):
    """Given a model path to delete into the 'src/models/' folder in repository, this functions delete that model.

    **Args:**
        path (str): Desired model path to delete. For example 'My_XGB_Classifier_model.pkl'.

    **Returns:**
        str: Message informing if the process was made succesfully or not.
    """
    try:
        response = delete_modelname(path)
        return JSONResponse(status_code=status.HTTP_200_OK, content=response)
    except HTTPException as H:
        raise H
    except Exception as E:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{E}")
