import os
import re
import pandas as pd
import logging
from datetime import datetime
from fastapi import HTTPException, status
from fastapi.responses import FileResponse
from sklearn.preprocessing import MinMaxScaler


def folder_inspection(path: str):
    if os.path.isfile(path):
        return HTTPException(
            status_code=status.HTTP_204_NO_CONTENT,
            detail=f"{os.path.basename(path)} isn't a folder path, but a file path. Nothing to inspect.",
        )
    try:
        scan = [x for x in os.scandir(path)]
        result = []
        for obj in scan:
            if str(obj).find(".") >= 0:
                result.append(obj.name)
            elif str(obj).find("_") < 0:
                result.append({obj.name: folder_inspection(obj.path)})
        return result
    except:
        logging.error(f"Folder {os.path.basename(path)} couldn't be found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Path '{os.path.basename(path)}' wasn't found into 'src/files/' directory.",
        )


def download_by_path(path: str):
    try:
        filename = os.path.basename(path)
        file_to_download = FileResponse(path=path, filename=filename)
        return file_to_download
    except:
        logging.error(f"File {os.path.basename(path)} couldn't be found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Path '{os.path.basename(path)}' wasn't found into 'src/files/' directory.",
        )


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


def get_features_from_df(
    df,
    categorical_features: list = None,
    numerical_features: list = None,
    minmax_scaler: bool = False,
):
    categorical_output = pd.DataFrame([])
    numerical_output = pd.DataFrame([])
    if categorical_features is not None or len(categorical_features) > 0:
        # categorical_feature exist in DataFrame
        for feature in categorical_features:
            if feature not in list(
                set([re.sub(r"_.*", "", col) for col in df.columns])
            ):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Feature {feature} isn't part of the df.columns {df.columns}",
                )
        # One Hot Encoding for Categorical features
        categorical_output = pd.concat(
            [
                pd.get_dummies(df[feature], prefix=feature)
                for feature in categorical_features
            ],
            axis=1,
        )
    if numerical_features is not None or len(numerical_features) > 0:
        minmax_scaler = MinMaxScaler()
        for feature in numerical_features:
            if feature not in df.columns:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Feature {feature} isn't part of the df.columns {df.columns}",
                )
            if minmax_scaler:
                # Min Max Scaler for Numerical features
                numerical_output = pd.concat(
                    [
                        pd.DataFrame(
                            minmax_scaler.fit_transform(
                                df[feature].values.reshape(-1, 1)
                            ),
                            columns=[feature],
                        )
                        for feature in numerical_features
                    ],
                    axis=1,
                )
            else:
                numerical_output = pd.concat(
                    [df[feature] for feature in numerical_features],
                    axis=1,
                )
    return pd.concat([categorical_output, numerical_output], axis=1)
