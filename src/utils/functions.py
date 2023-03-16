import os
from datetime import datetime
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


def temporada_alta(fecha):
    fecha_año = int(fecha.split("-")[0])
    fecha = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S")
    range1_min = datetime.strptime("15-Dec", "%d-%b").replace(year=fecha_año)
    range1_max = datetime.strptime("31-Dec", "%d-%b").replace(year=fecha_año)
    range2_min = datetime.strptime("1-Jan", "%d-%b").replace(year=fecha_año)
    range2_max = datetime.strptime("3-Mar", "%d-%b").replace(year=fecha_año)
    range3_min = datetime.strptime("15-Jul", "%d-%b").replace(year=fecha_año)
    range3_max = datetime.strptime("31-Jul", "%d-%b").replace(year=fecha_año)
    range4_min = datetime.strptime("11-Sep", "%d-%b").replace(year=fecha_año)
    range4_max = datetime.strptime("30-Sep", "%d-%b").replace(year=fecha_año)

    if (
        (fecha >= range1_min and fecha <= range1_max)
        or (fecha >= range2_min and fecha <= range2_max)
        or (fecha >= range3_min and fecha <= range3_max)
        or (fecha >= range4_min and fecha <= range4_max)
    ):
        return 1
    else:
        return 0


def dif_min(data):
    fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
    fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
    dif_min = ((fecha_o - fecha_i).total_seconds()) / 60
    return dif_min


def get_periodo_dia(fecha):
    fecha_time = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S").time()
    mañana_min = datetime.strptime("05:00", "%H:%M").time()
    mañana_max = datetime.strptime("11:59", "%H:%M").time()
    tarde_min = datetime.strptime("12:00", "%H:%M").time()
    tarde_max = datetime.strptime("18:59", "%H:%M").time()
    noche_min1 = datetime.strptime("19:00", "%H:%M").time()
    noche_max1 = datetime.strptime("23:59", "%H:%M").time()
    noche_min2 = datetime.strptime("00:00", "%H:%M").time()
    noche_max2 = datetime.strptime("4:59", "%H:%M").time()

    if fecha_time > mañana_min and fecha_time < mañana_max:
        return "mañana"
    elif fecha_time > tarde_min and fecha_time < tarde_max:
        return "tarde"
    elif (fecha_time > noche_min1 and fecha_time < noche_max1) or (
        fecha_time > noche_min2 and fecha_time < noche_max2
    ):
        return "noche"
