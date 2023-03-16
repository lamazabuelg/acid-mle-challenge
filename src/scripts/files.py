import pandas as pd
import numpy as np
from fastapi import HTTPException, status
import settings
from utils.functions import (
    folder_inspection,
    download_by_path,
    temporada_alta,
    dif_min,
    get_periodo_dia,
)

path_base = settings.PATH_BASE
files_path = f"{path_base}\\src\\files"
original_dataset_path = f"{files_path}\\input\\dataset_SCL.csv"


def get_all_file_names(path: str):
    if path is None:
        path = files_path
    else:
        path = path.replace("/", "\\")
        path = f"{path_base}\\{path}"
    result = folder_inspection(path)
    return result


def download_from_filename(path: str):
    if path is None:
        path = original_dataset_path
    else:
        path = path.replace("/", "\\")
        path = f"{path_base}\\{path}"
    file_to_download = download_by_path(path)
    return file_to_download


def create_features_from_base(
    destination_storage_folder: str,
    generate_both_files: bool,
    files_to_generate: list,
    test_mode: bool,
    test_size: int,
    test_random_state: int,
):
    # Review if args are coherent
    if not generate_both_files and (
        len(files_to_generate) == 0 or files_to_generate is None
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    # Set path for windows
    if destination_storage_folder is None:
        destination_storage_folder = f"{files_path}\\output"
    else:
        destination_storage_folder = destination_storage_folder.replace("/", "\\")
        destination_storage_folder = f"{path_base}\\{destination_storage_folder}"

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
    base.to_csv(
        f"{destination_storage_folder}\\dataset_SCL_complete.csv", index=False
    ) if generate_both_files or "complete" in files_to_generate else None
    new_features.to_csv(
        f"{destination_storage_folder}\\synthetic_features.csv", index=False
    ) if generate_both_files or "new_features" in files_to_generate else None
