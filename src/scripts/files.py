import os
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
files_path = os.path.join(path_base, "src", "files")
original_dataset_path = os.path.join(files_path, "input", "dataset_SCL.csv")
if not os.path.isdir(files_path):
    os.mkdir(files_path)


def get_all_file_names(path: str):
    if path is None:
        path = files_path
    else:
        path = os.path.join(files_path, path)
    result = folder_inspection(path)
    return result


def download_from_filename(path: str):
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
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="There was an error uploading the file",
        )
    finally:
        file.file.close()

    return {
        "status_code": status.HTTP_201_CREATED,
        "detail": f"Successfully uploaded {file.filename}",
    }


def create_features_from_base(
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
    return base.to_dict()
