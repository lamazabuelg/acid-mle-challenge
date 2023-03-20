import os
import pandas as pd
import pickle
import logging
import warnings
import settings
from datetime import datetime
from typing import Optional, Dict
from fastapi import HTTPException, status
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from utils.functions import folder_inspection, delete_by_path

warnings.filterwarnings("ignore")
path_base = settings.PATH_BASE
files_path = os.path.join(path_base, "files")
if not os.path.isdir(files_path):
    os.mkdir(files_path)
models_path = os.path.join(path_base, "models")
if not os.path.isdir(models_path):
    os.mkdir(models_path)


def get_model_filenames():
    result = folder_inspection(models_path)
    return result


def model_classification_report(y_test_filename, y_predicted_filename):
    try:
        # Read Data Files
        y_test = pd.read_csv(os.path.join(files_path, "output", y_test_filename))
        y_pred = pd.read_csv(os.path.join(files_path, "output", y_predicted_filename))

        df = pd.DataFrame(pd.concat([y_test, y_pred], axis=1))
        df.columns = ["real", "predicted"]
        # Making Report
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        recall = recall_score(y_true=y_test, y_pred=y_pred)
        f1score = f1_score(y_true=y_test, y_pred=y_pred)
        rocauc_score = roc_auc_score(y_true=y_test, y_score=y_pred)
        result = {
            "accuracy": accuracy,
            "recall": recall,
            "f1_score": f1score,
            "roc_auc_score": rocauc_score,
            "data": df.to_dict(),
        }
        return result
    except HTTPException as H:
        raise H
    except Exception as E:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{E}")


def make_predictions(model_filename, X_test_filename):
    try:
        # Load Model
        with open(
            os.path.join(models_path, f"{model_filename}.pkl"), mode="rb"
        ) as file:
            model = pickle.load(file)

        # Read Input Data file
        X_test = pd.read_csv(os.path.join(files_path, "output", X_test_filename))

        # Predict
        predictions = pd.DataFrame(model.predict(X_test))
        filename_output = os.path.join(
            files_path, "output", f"{model_filename}-predictions.csv"
        )
        predictions.to_csv(filename_output, index=False)
        logging.info(
            f"'{filename_output}.pkl' succesfully created in 'src/files/output/'."
        )
        return list(predictions[0])
    except HTTPException as H:
        raise H
    except Exception as E:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{E}")


def train_binary_cls_model(
    X_train_filename: str,
    y_train_filename: str,
    model_name: Optional[str],
    destination_model_name: Optional[str],
    model_custom_params: Optional[Dict],
    grid_search_cv: Optional[bool],
    grid_search_cv_params: Optional[Dict],
    random_state: Optional[int],
    balancing_methodology: Optional[str],
):
    try:
        # Review if args are coherent
        ## GridSearchCV
        if grid_search_cv and (
            grid_search_cv_params is None or len(grid_search_cv_params) == 0
        ):
            logging.error(
                "When grid_search_cv is setted to True, grid_search_cv_params is required!"
            )
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

        ## model_name
        if model_name is not None and model_name not in settings.MODELS_ALLOWED:
            logging.error(
                f"model_name got unexpected value. Possible models are: {settings.MODELS_ALLOWED}"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"model_name got unexpected value. Possible models are: {settings.MODELS_ALLOWED}",
            )

        ## balancing_methodology
        if balancing_methodology is not None and balancing_methodology not in [
            "balanced",
            "under",
            "over",
        ]:
            logging.error(
                "balancing_methodology got unexpected value. Possible methodologies are: 'balanced','under-sampling','over-sampling'."
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="balancing_methodology got unexpected value. Possible methodologies are: 'balanced','under-sampling','over-sampling'.",
            )

        logging.info(
            f"Starting session, working with random_state = {random_state} for reproducible results."
        )

        # X and y
        X_train = pd.read_csv(os.path.join(files_path, "output", X_train_filename))
        y_train = pd.read_csv(os.path.join(files_path, "output", y_train_filename))

        # Start model config
        if model_name is None:
            model_name = "xgb"
        logging.info(f"Configuring model {model_name}...")
        logging.info(f"Custom Params for model: {model_custom_params}.")

        ## Balancing method
        if balancing_methodology == "under":
            random_undersampler = RandomUnderSampler(random_state=random_state)
            X_train, y_train = random_undersampler.fit_resample(X_train, y_train)
        if balancing_methodology == "over":
            random_oversampler = RandomOverSampler(random_state=random_state)
            X_train, y_train = random_oversampler.fit_resample(X_train, y_train)
        if balancing_methodology == "balanced":
            if model_name == "log-reg":
                model = (
                    LogisticRegression(class_weight="balanced")
                    if model_custom_params is None and len(model_custom_params) == 0
                    else LogisticRegression(
                        class_weight="balanced", **model_custom_params
                    )
                )
                model_fit = model.fit(X_train, y_train)
            elif model_name == "xgb":
                counter = Counter(y_train["atraso_15"].tolist())
                scale_pos_weight_value = counter[0] / counter[1]
                model = (
                    XGBClassifier(
                        scale_pos_weight=scale_pos_weight_value,
                        random_state=random_state,
                    )
                    if model_custom_params is None and len(model_custom_params) == 0
                    else XGBClassifier(
                        scale_pos_weight=scale_pos_weight_value,
                        random_state=random_state,
                        **model_custom_params,
                    )
                )
                model_fit = model.fit(X_train, y_train)
        else:
            if model_name == "log-reg":
                model = (
                    LogisticRegression()
                    if model_custom_params is None and len(model_custom_params) == 0
                    else LogisticRegression(**model_custom_params)
                )
                model_fit = model.fit(X_train, y_train)
            elif model_name == "xgb":
                model = (
                    XGBClassifier(
                        random_state=random_state,
                    )
                    if model_custom_params is None and len(model_custom_params) == 0
                    else XGBClassifier(
                        random_state=random_state,
                        **model_custom_params,
                    )
                )
                model_fit = model.fit(X_train, y_train)

        # GridSearchCV
        logging.info(f"GridSearchCV for model: {grid_search_cv}")
        logging.info(f"GridSearchCV Params for model: {grid_search_cv_params}")
        if grid_search_cv:
            param_grid = grid_search_cv_params.get("param_grid")
            cv = grid_search_cv_params.get("cv", None)
            n_jobs = grid_search_cv_params.get("n_jobs", None)
            verbose = grid_search_cv_params.get("verbose", None)
            model_fit = GridSearchCV(
                model_fit, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose
            ).fit(X_train, y_train)

        # Save Files
        ## Model.pkl
        if destination_model_name is None:
            destination_model_name = (
                f"{model_name}_{datetime.now().strftime(format='%Y-%m-%d-%H%M')}"
            )
        logging.info(
            f"Saving trained model as {destination_model_name}.pkl into src/models/."
        )
        with open(
            os.path.join(models_path, f"{destination_model_name}.pkl"), mode="wb"
        ) as file:
            pickle.dump(model_fit, file)
            logging.info(
                f"Model '{destination_model_name}.pkl' succesfully created in 'src/models/'."
            )
        return (
            f"Success! Model '{destination_model_name}.pkl' created in 'src/models/'."
        )
    except HTTPException as H:
        raise H
    except Exception as E:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{E}")
