# public libraries
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status

# own libraries
from scripts.files import get_all_file_names, download_from_filename

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
    except HTTPException:
        return HTTPException(
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
    except HTTPException:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
        )
