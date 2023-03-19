import os
import re
import pandas as pd
import numpy as np
import logging
import settings
from typing import Optional, List
from fastapi import HTTPException, status
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from utils.functions import (
    folder_inspection,
    download_by_path,
    temporada_alta,
    dif_min,
    get_periodo_dia,
    get_features_from_df,
)

path_base = settings.PATH_BASE
files_path = os.path.join(path_base, "src", "files")
original_dataset_path = os.path.join(files_path, "input", "dataset_SCL.csv")
if not os.path.isdir(files_path):
    os.mkdir(files_path)


def get_all_file_names(
    path: Optional[str],
):
    if path is None:
        path = files_path
    else:
        path = os.path.join(files_path, path)
    result = folder_inspection(path)
    return result


def download_from_filename(
    path: Optional[str],
):
    get_all_file_names(path)
    path = os.path.join(files_path, path)
    file_to_download = download_by_path(path)
    return file_to_download


def upload_new_file(file):
    try:
        contents = file.file.read()
        with open(os.path.join(files_path, "input", file.filename), "wb") as f:
            f.write(contents)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="There was an error uploading the file",
        )
    finally:
        file.file.close()

    return f"Successfully uploaded {file.filename}"


def create_features_from_base(
    generate_both_files: Optional[bool],
    files_to_generate: Optional[List],
    test_mode: Optional[bool],
    test_size: Optional[int],
    test_random_state: Optional[int],
):
    # Review if args are coherent
    if not generate_both_files and (
        len(files_to_generate) == 0 or files_to_generate is None
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"When generate_both_files is False. One value between 'complete' or 'new_features' must be selected here. Received {files_to_generate} instead.",
        )

    # Set destination path
    destination_storage_folder = os.path.join(files_path, "output")

    # load base file
    base = pd.read_csv(original_dataset_path)
    if test_mode:
        base = base.sample(test_size, random_state=test_random_state)

    # create new columns
    base["temporada_alta"] = base["Fecha-I"].apply(temporada_alta)
    base["dif_min"] = base.apply(dif_min, axis=1)
    base["atraso_15"] = np.where(base["dif_min"] > 15, 1, 0)
    base["periodo_dia"] = base["Fecha-I"].apply(get_periodo_dia)
    new_features = base[["temporada_alta", "dif_min", "atraso_15", "periodo_dia"]]
    # load files
    if generate_both_files or "complete" in files_to_generate:
        base.to_csv(
            os.path.join(destination_storage_folder, "dataset_SCL_complete.csv"),
            index=False,
        )
    if generate_both_files or "new_features" in files_to_generate:
        new_features.to_csv(
            os.path.join(destination_storage_folder, "synthetic_features.csv"),
            index=False,
        )
    base_nan = base[base.isin([np.nan, np.inf, -np.inf]).any(1)]
    base = base[~base.isin([np.nan, np.inf, -np.inf]).any(1)]
    response = {
        "values with nan": base_nan.fillna("").to_dict(orient="records"),
        "values without nan": base.to_dict(orient="records"),
    }
    return response


def split_train_test(
    data_filename: Optional[str],
    features_filter: Optional[List],
    categorical_features: Optional[List],
    numerical_features: Optional[List],
    minmax_scaler_numerical_f: Optional[bool],
    label: Optional[str],
    shuffle_data: Optional[bool],
    shuffle_features: Optional[List],
    sample_data: Optional[int],
    random_state: Optional[int],
):
    try:
        # Review if args are coherent

        # Input file
        if data_filename is None:
            data_filename = "dataset_SCL_complete.csv"
        data_filename = os.path.join(files_path, data_filename)
        input_filename = re.sub(r"\..*", "", os.path.basename(data_filename))
        input_foldername = os.path.dirname(data_filename)
        try:
            folder_inspection(input_foldername)
        except:
            logging.error(f"Folder {input_foldername} couldn't be found.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Folder {input_foldername} couldn't be found.",
            )

        # X-y Split
        logging.info("Reading input file...")
        data = pd.read_csv(data_filename)
        logging.info(
            f"Read file {input_filename} succesfully! It has {len(data)} rows."
        )

        ## Sampling
        if sample_data is not None:
            data = data.sample(n=sample_data, random_state=random_state)
            logging.info("Data resampled succesfully!")

        ## Shuffle
        if shuffle_data:
            data = shuffle(
                data[shuffle_features],
                random_state=random_state,
            )
            logging.info("Data shuffled succesfully!")
        X = get_features_from_df(
            df=data,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            minmax_scaler=minmax_scaler_numerical_f,
        )
        y = data[label]

        # Features selection
        if features_filter is not None and len(features_filter) > 0:
            logging.info(f"Train model only with features: [{features_filter}].")
            X = X[features_filter]

        # Train-test Split
        logging.info(
            f"Working with train_test_split config: Just the 66% of data is going to be taken for training tasks into the {random_state} random_state."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=random_state
        )
        # src/files/output/
        X_train.to_csv(
            os.path.join(files_path, "output", f"{input_filename}-X_train.csv"),
            index=False,
        )
        X_test.to_csv(
            os.path.join(files_path, "output", f"{input_filename}-X_test.csv"),
            index=False,
        )
        y_train.to_csv(
            os.path.join(files_path, "output", f"{input_filename}-y_train.csv"),
            index=False,
        )
        y_test.to_csv(
            os.path.join(files_path, "output", f"{input_filename}-y_test.csv"),
            index=False,
        )
        return {
            "X_train": X_train.to_dict("records"),
            "X_test": X_test.to_dict("records"),
            "y_train": list(y_train),
            "y_test": list(y_test),
        }
    except HTTPException as H:
        raise H
    except Exception as E:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{E}")
