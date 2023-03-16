import os
from fastapi.responses import FileResponse


def folder_inspection(path: str):
    scan = [x for x in os.scandir(path)]
    result = []
    for obj in scan:
        if str(obj).find(".") >= 0:
            result.append(obj.name)
        elif str(obj).find("_") < 0:
            result.append({obj.name: folder_inspection(obj.path)})
    return result


def download_by_path(path: str):
    filename = path.split("\\")[-1]
    file_to_download = FileResponse(path=path, filename=filename)
    return file_to_download
