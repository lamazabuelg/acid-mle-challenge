from typing import Optional
from fastapi import APIRouter, HTTPException, status, File, UploadFile
from fastapi.responses import JSONResponse
from schemas.files import CreateFeaturesSchema, TrainTestSplitSchema
from scripts.files import (
    get_all_file_names,
    download_from_filename,
    delete_filename,
    create_features_from_base,
    upload_new_file,
    split_train_test,
)

files_router = APIRouter()

# GET
@files_router.get(
    "/files/all_files",
    tags=["Files"],
    status_code=status.HTTP_200_OK,
    summary="Get the directory info of a selected path into directory 'src/files/' in the repository.",
)
def all_file_names(path: Optional[str] = None):
    """Given a folder to search into the 'src/files/' folder in repository, this functions returns the structure of files inside that folder path.

    **Args:**
        path (Optional[str], optional): Desired folder to watch. Defaults to '/src/files/' folder in repo.

    **Returns:**
        Dict: key:values where keys are folders and values are files with any extension.
    """
    try:
        files = get_all_file_names(path)
        return files
    except HTTPException as H:
        raise H
    except Exception as E:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{E}")


# GET
@files_router.get(
    "/files/download_by_name",
    tags=["Files"],
    status_code=status.HTTP_200_OK,
    summary="Download a given file of the 'src/files/' repo's directory.",
)
def download_file_by_name(path: str):
    """Given a path to search into the 'src/files/' folder in repository, this functions returns the file for its downloading.

    **Args:**
        path (str): Desired folder to watch. For example 'input/myfile.csv'.

    **Returns:**
        file: File for download locally.
    """
    try:
        file_to_download = download_from_filename(path)
        return file_to_download
    except HTTPException as H:
        raise H
    except Exception as E:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{E}")


# POST
@files_router.post(
    "/files/upload_file",
    tags=["Files"],
    status_code=status.HTTP_200_OK,
    summary="Upload a file into 'src/files/input/' directory.",
)
def upload_file(file: UploadFile = File(...)):
    """Upload a file into 'src/files/input/' directory and let it be useful for latest use.

    **Args:**
        file: File for upload in repo's directory '/src/files/'.
    """
    try:
        response = upload_new_file(file)
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=response)
    except HTTPException as H:
        raise H
    except Exception as E:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{E}")


# POST
@files_router.post(
    "/files/create_additional_features",
    tags=["Files"],
    status_code=status.HTTP_200_OK,
    summary="With the 'src/files/input/dataset_SCL.csv' file, generate 'src/files/output/dataset_SCL_complete.csv' and 'src/files/output/synthetic_features.csv' files in repo.",
    response_model=CreateFeaturesSchema,
)
def create_additional_features(request_body: CreateFeaturesSchema):
    """This Endpoint takes the **dataset_SCL.csv** stored by default in path 'src/files/input/dataset_SCL.csv' and generate the following column features into 'src/files/output/synthetic_features.csv':
        **temporada_alta:**
            1 if *Fecha-I* is between Dec-15 and Mar-03,
                or between Jul-15 and Jul-31,
                or Sep-11 and Sep-30;
            0 otherwise.
        **dif_min:** difference in minutes between *Fecha-O* and *Fecha-I*.
        **atraso_15:**
            1 if **dif_min** > 15,
            0 otherwise.
        **periodo_dia:** Given *Fecha-I*,
            "mañana" if between 5:00 and 11:59,
            "tarde" if between 12:00 and 18:59,
            "noche" if between 19:00 and 4:59.
        After generated, **synthetic_features.csv** is going to contain just these columns. While **dataset_SCL_complete.csv** is going to contain base **dataset_SCL.csv columns + synthetic_features.csv** columns.

    **Args:**
        generate_both_files (bool, optional): If user wants to generate both **dataset_SCL_complete.csv** and **synthetic_features.csv**. Defaults to True.
        generate_files (str, optional): If generate_both_files is False. Select 'complete' for just having **dataset_SCL_complete.csv** or 'new_features' for just having **synthetic_features.csv** file. Defaults to None.
        test_mode (bool, optional): If user wants to run this endpoint in test mode. Test mode implies that process will be take only the number of test_size records of .csv files. Defaults to False.
        test_size (int, optional): Number of records to filter in test_mode. Default 100.
        test_random_state (int, optional): Random state number to reproduce code in several executions. Default None.

    **Returns:**
        str: Message informing if the process was made succesfully or not.
    """
    try:
        base_complete = create_features_from_base(
            request_body.generate_both_files,
            request_body.generate_files,
            request_body.test_mode,
            request_body.test_size,
            request_body.test_random_state,
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=base_complete)
    except HTTPException as H:
        raise H
    except Exception as E:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{E}")


# POST
@files_router.post(
    "/files/train_test_split",
    tags=["Files"],
    status_code=status.HTTP_200_OK,
    summary="Generate 'X_train', 'X_test', 'y_train' and 'y_test' files into 'src/files/output/'.",
    response_model=TrainTestSplitSchema,
)
def train_test_split(request_body: TrainTestSplitSchema):
    """Given a structured data_filename.csv file into 'src/files/', returns a dictionary with four keys: 'X_train', 'X_test', 'y_train' and 'y_test'.
    Likewise it save each part of returned dictionary as a .csv file into 'src/files/output/' directory.

    **Args:**
        data_filename: str. Filename to split into 'src/files/' directory. For example 'output/dataset_SCL_complete.csv'.
        features_filter (List, optional): List of features existing into the input file to filter X_train and X_test batches.
            For example ['my_feature1', 'my_feature2']. If None, all file's features are going to be considered.
        categorical_features (List, optional): List of features to be treated as categorical ones and do **One Hot Encoding** with it. Defaults to None.
        numerical_features (List, optional): List of features to be treated as numerical ones and do **MinMax Scaling** with it ONLY if arg minmax_scaler_numerical_f is True. Defaults to None.
        minmax_scaler_numerical_f (bool, optional): For numerical_features, apply a MinMax Scaling. Defaults to False.
        label (str, optional): Label column in data to split in y_train and y_test batches. Defaults to "atraso_15".
        shuffle_data (bool, optional): If data must be shuffled. Defaults to True.
        shuffle_features (List, optional): List of features to shuffle. Defaults to ["OPERA", "MES", "TIPOVUELO", "SIGLADES", "DIANOM", "atraso_15"].
        sample_data (int, optional): If not None, data is going to be sampled by this max number of records. Defaults to None.
        random_state (int, optional): Random state for code replication, it applies to shuffle and sample args. If None, results are expected to be random. Defaults to None.

    **Returns:**
        str: Message informing if the process was made succesfully or not.
    """
    try:
        response = split_train_test(
            request_body.data_filename,
            request_body.features_filter,
            request_body.categorical_features,
            request_body.numerical_features,
            request_body.minmax_scaler_numerical_f,
            request_body.label,
            request_body.shuffle_data,
            request_body.shuffle_features,
            request_body.sample_data,
            request_body.random_state,
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=response)
    except HTTPException as H:
        raise H
    except Exception as E:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{E}")


# DELETE
@files_router.delete(
    "/files/delete_by_name",
    tags=["Files"],
    status_code=status.HTTP_200_OK,
    summary="Delete a given file of the 'src/files/' repo's directory.",
)
def delete_by_name(path: str):
    """Given a file path to delete into the 'src/files/' folder in repository, this functions delete that file.

    **Args:**
        path (str): Desired file path to delete. For example 'input/myfile.csv'.

    **Returns:**
        str: Message informing if the process was made succesfully or not.
    """
    try:
        response = delete_filename(path)
        return JSONResponse(status_code=status.HTTP_200_OK, content=response)
    except HTTPException as H:
        raise H
    except Exception as E:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{E}")
