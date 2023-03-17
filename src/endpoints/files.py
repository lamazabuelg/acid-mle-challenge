# public libraries
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status

# own libraries
from scripts.files import (
    get_all_file_names,
    download_from_filename,
    create_features_from_base,
)

files_router = APIRouter()

# GET
@files_router.get(
    "/files/all",
    tags=["Files"],
    status_code=status.HTTP_200_OK,
    summary="Get the structure tree of a given path directory in the repository.",
)
def all_file_names(path: Optional[str] = None):
    """Given a path to search into the repository, this functions returns the structure of files inside that path.

    Args:
        path (Optional[str], optional): Desired Path to search. Defaults to /src/files folder in repo.

    Returns:
        python dict: key:values where keys are folders and values are files with any extension.
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
    summary="Download a given pathfile of the repo's directory.",
)
def download_file_by_name(path: Optional[str] = None):
    """Given a path to search into the repository, this function download the selected file.

    Args:
        path (Optional[str], optional): Desired Pathfile to download. Defaults to /src/files/input/dataset_SCL.csv folder in repo.
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
    "/files/create_additional_features",
    tags=["Files"],
    status_code=status.HTTP_200_OK,
    summary="With the dataset_SCL.csv file, generate dataset_SCL_complete.csv and synthetic_features.csv files in a directory path.",
)
def create_additional_features(
    destination_storage_name: Optional[str] = None,
    generate_both_files: bool = True,
    generate_files: list = None,
    test_mode: Optional[bool] = False,
    test_size: Optional[int] = 100,
    test_random_state: Optional[int] = None,
):
    """This Endpoint takes the **dataset_SCL.csv** stored by default in path **src/files/input/dataset_SCL.csv** and generate the following column features:

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

        destination_storage_name (Optional[str]): Folder path to save files. Defaults to /src/files/output.

        generate_both_files (bool, optional): If user wants to generate both dataset_SCL_complete.csv and synthetic_features.csv. Defaults to True.

        generate_files (Optional[List], optional): If generate_both_files is False. Select 'complete' for just having dataset_SCL_complete.csv or 'new_features' for just having synthetic_features.csv file. Defaults to None.

        test_mode (bool, optional): If user wants to run this endpoint in test mode. Test mode implies that it will be taken only the number of test_size records of .csv files. Defaults to False.

        test_size (int, optional): Number of records to filter in test_mode. Default 100.

        test_random_state (int, optional): Random state number to reproduce code in several executions. Default None.

    Returns:

        HTTPResponse
    """
    try:
        create_features_from_base(
            destination_storage_name,
            generate_both_files,
            generate_files,
            test_mode,
            test_size,
            test_random_state,
        )
        return status.HTTP_200_OK
    except HTTPException as H:
        raise H
    except:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
        )
