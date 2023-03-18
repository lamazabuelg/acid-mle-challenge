from typing import Optional
from fastapi import APIRouter, HTTPException, status, File, UploadFile
from fastapi.responses import JSONResponse
from schemas.files import CreateFeaturesSchema
from scripts.files import (
    get_all_file_names,
    download_from_filename,
    create_features_from_base,
    upload_new_file,
)

files_router = APIRouter()

# GET
@files_router.get(
    "/files/all",
    tags=["Files"],
    status_code=status.HTTP_200_OK,
    summary="Get the directory info of path ../files in the repository.",
)
def all_file_names(path: Optional[str] = None):
    """Given a folder to search into the ../files folder in repository, this functions returns the structure of files inside that folder path.

    Args:
        path (Optional[str], optional): Desired folder to watch. Defaults to /src/files folder in repo.

    Returns:
        Dict: key:values where keys are folders and values are files with any extension.
    """
    try:
        files = get_all_file_names(path)
        return files
    except HTTPException as H:
        raise H
    except:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
        )


# GET
@files_router.get(
    "/files/download_by_name",
    tags=["Files"],
    status_code=status.HTTP_200_OK,
    summary="Download a given file of the 'src/files/' repo's directory.",
)
def download_file_by_name(path: str):
    """Given a folder to search into the ../files folder in repository, this functions returns the file for its downloading.

    Args:
        path (Optional[str], optional): Desired folder to watch. For example 'input/myfile.csv'.
    """
    try:
        file_to_download = download_from_filename(path)
        return file_to_download
    except HTTPException as H:
        raise H
    except:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
        )


# POST
@files_router.post(
    "/files/upload_file",
    tags=["Files"],
    status_code=status.HTTP_200_OK,
    summary="Upload a file into 'src/files/input/' directory.",
)
def upload_file(file: UploadFile = File(...)):
    """Given a path to search into the repository, this function download the selected file.

    Args:
        path (Optional[str], optional): Desired Pathfile to download. Defaults to /src/files/input/dataset_SCL.csv folder in repo.
    """
    try:
        response = upload_new_file(file)
        return response
    except HTTPException as H:
        raise H
    except:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
        )


# POST
@files_router.post(
    "/files/create_additional_features",
    tags=["Files"],
    status_code=status.HTTP_200_OK,
    summary="With the dataset_SCL.csv file, generate dataset_SCL_complete.csv and synthetic_features.csv files in a directory path.",
    response_model=CreateFeaturesSchema,
)
def create_additional_features(request_body: CreateFeaturesSchema):
    """This Endpoint takes the **dataset_SCL.csv** stored by default in path **src/files/input/dataset_SCL.csv** and generate the following column features into **/src/files/output**:

        temporada_alta:
            - 1 if Fecha-I is between Dec-15 and Mar-03,
                or between Jul-15 and Jul-31,
                or Sep-11 and Sep-30;
            - 0 otherwise.

        dif_min: difference in minutes between Fecha-O and Fecha-I.

        atraso_15:
            - 1 if dif_min > 15,
            - 0 otherwise.

        periodo_dia: Given Fecha-I
            - "mañana" if between 5:00 and 11:59
            - "tarde" if between 12:00 and 18:59
            - "noche" if between 19:00 and 4:59

        After generated, synthetic_features.csv is going to contain just these columns. While dataset_SCL_complete.csv is going to contain base dataset_SCL.csv columns + synthetic_features.csv columns.

    Args:

        generate_both_files (bool, optional): If user wants to generate both dataset_SCL_complete.csv and synthetic_features.csv. Defaults to True.

        generate_files (Optional[List], optional): If generate_both_files is False. Select 'complete' for just having dataset_SCL_complete.csv or 'new_features' for just having synthetic_features.csv file. Defaults to None.

        test_mode (bool, optional): If user wants to run this endpoint in test mode. Test mode implies that it will be taken only the number of test_size records of .csv files. Defaults to False.

        test_size (int, optional): Number of records to filter in test_mode. Default 100.

        test_random_state (int, optional): Random state number to reproduce code in several executions. Default None.

    Returns:

        Dict: dataset_SCL_complete.csv content data.
    """
    try:
        base_complete = create_features_from_base(
            request_body.generate_both_files,
            request_body.generate_files,
            request_body.test_mode,
            request_body.test_size,
            request_body.test_random_state,
        )
        return JSONResponse(base_complete)
    except HTTPException as H:
        raise H
    except Exception as E:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{E}")
