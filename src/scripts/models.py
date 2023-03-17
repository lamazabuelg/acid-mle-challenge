import os
import pandas as pd
import pickle
import logging
import warnings
import settings
from datetime import datetime
from typing import List, Optional, Dict
from fastapi import APIRouter, HTTPException, status
from utils.functions import folder_inspection
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
path_base = settings.PATH_BASE
files_path = f"{path_base}\\src\\files"
models_path = f"{path_base}\\src\\models"


def train_binary_cls_model(
    input_data_filename_path: Optional[str],
    features_filter: Optional[List],
    model_name: Optional[str],
    destination_storage_name: Optional[str],
    model_custom_params: Optional[Dict],
    grid_search_cv: Optional[bool],
    grid_search_cv_params: Optional[Dict],
    train_test_split_data: Optional[bool],
    random_state: Optional[int],
    shuffle_data: Optional[bool],
    endpoint_test_mode: Optional[bool],
):
    try:
        # Review if args are coherent
        # GridSearchCV
        if grid_search_cv and (
            grid_search_cv_params is None or len(grid_search_cv_params) == 0
        ):
            logging.error(
                "When grid_search_cv is setted to True, grid_search_cv_params is required!"
            )
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

        # Input file
        if input_data_filename_path is None:
            input_data_filename_path = f"{files_path}\\output\\dataset_SCL_complete.csv"
        else:
            input_data_filename_path = input_data_filename_path.replace("/", "\\")
            input_data_filename_path = f"{path_base}\\{input_data_filename_path}"
        input_filename = input_data_filename_path.split("\\")[-1]
        input_foldername = input_data_filename_path.replace(input_filename, "")
        try:
            folder_inspection(input_foldername)
        except:
            logging.error(f"Folder {input_foldername} couldn't be found.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

        # model_name
        if model_name not in settings.MODELS_ALLOWED:
            logging.error(
                f"model_name got unexpected value. Possible models are: {settings.MODELS_ALLOWED}"
            )
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

        logging.info(
            f"Starting session, working with random_state = {random_state} for reproducible results."
        )

        # X-y Split
        logging.info("Reading input file...")
        data = (
            pd.read_csv(input_data_filename_path, nrows=1000)
            if endpoint_test_mode
            else pd.read_csv(input_data_filename_path)
        )
        logging.info(
            f"Read file {input_filename} succesfully! It has {len(data)} rows."
        )

        ## Shuffle
        if shuffle_data:
            data = shuffle(
                data[["OPERA", "MES", "TIPOVUELO", "SIGLADES", "DIANOM", "atraso_15"]],
                random_state=random_state,
            )
            logging.info("Data shuffled data succesfully!")
        X = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )
        y = data["atraso_15"]

        # Features selection
        if features_filter is not None and len(features_filter) > 0:
            logging.info(f"Train model only with features: [{features_filter}].")
            X[features_filter]

        # Train-test Split
        if train_test_split_data:
            logging.info(
                f"Working with train_test_split config: Just the 66% of data is going to be taken for training tasks into the {random_state} random_state."
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=random_state
            )
        else:
            logging.info(
                f"Working without train_test_split config: All data is going to be taken for training tasks into the {random_state} random_state."
            )
            X_train, X_test, y_train, y_test = X, None, y, None

        # Start model config
        if model_name is None:
            model_name = "xgb"
        logging.info(f"Configuring model {model_name}...")
        logging.info(f"Custom Params for model: {model_custom_params}.")
        if model_name == "log-reg":
            model = (
                LogisticRegression()
                if model_custom_params is None and len(model_custom_params) == 0
                else LogisticRegression(**model_custom_params)
            )
            model_fit = model.fit(X_train, y_train)
        elif model_name == "xgb":
            model = (
                XGBClassifier(random_state=random_state)
                if model_custom_params is None and len(model_custom_params) == 0
                else XGBClassifier(random_state=random_state, **model_custom_params)
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

        # Save File
        if destination_storage_name is None:
            destination_storage_name = (
                f"{model_name}_{datetime.now().strftime(format='%Y-%m-%d-%H%M')}"
            )
        logging.info(
            f"Saving trained model as {destination_storage_name}.pkl into src/models/."
        )
        if not os.path.isdir(models_path):
            os.mkdir(models_path)
        with open(f"{models_path}\\{destination_storage_name}.pkl", mode="wb") as file:
            pickle.dump(model_fit, file)
        return status.HTTP_201_CREATED
    except HTTPException as H:
        raise H
    except Exception as E:
        raise E
