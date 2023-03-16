import settings
from utils.functions import folder_inspection, download_by_path

path_base = settings.PATH_BASE
files_path = f"{path_base}\\src\\files"


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
        path = f"{files_path}\\input\\dataset_SCL.csv"
    else:
        path = path.replace("/", "\\")
        path = f"{path_base}\\{path}"
    file_to_download = download_by_path(path)
    return file_to_download
