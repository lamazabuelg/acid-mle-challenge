from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


# class InputRowSchema(BaseModel):
#     OPERA: str
#     MES: int
#     TIPOVUELO: str
#     SIGLADES: str
#     DIANOM: str


class PredictionSchema(BaseModel):
    model_filename: str = Field(..., min_length=1)
    input_path_file: Optional[str] = Field()
    input_row: Optional[List[Dict]] = Field()
    categorical_features: Optional[List] = Field()
    numerical_features: Optional[List] = Field()
    minmax_scaler_numerical_f: Optional[bool] = Field()
    label_test: Optional[str] = Field()

    class Config:
        schema_extra = {
            "example": {
                "model_filename": "XGBoost_gridsearch_v1",
                "input_path_file": None,
                "input_row": [
                    {
                        "OPERA": "Grupo LATAM",
                        "MES": 9,
                        "TIPOVUELO": "N",
                        "SIGLADES": "Calama",
                        "DIANOM": "Sabado",
                        "atraso_15": 1,
                    }
                ],
                "categorical_features": ["OPERA", "MES", "TIPOVUELO"],
                "numerical_features": [],
                "minmax_scaler_numerical_f": False,
                "label_test": "atraso_15",
            },
            "example": {
                "model_filename": "XGBoost_gridsearch_v1",
                "input_path_file": "src/files/input/test_batch.csv",
                "input_row": None,
                "categorical_features": ["OPERA", "MES", "TIPOVUELO"],
                "numerical_features": [],
                "minmax_scaler_numerical_f": False,
                "label_test": "atraso_15",
            },
        }


# class ModelCustomParamsSchema(BaseModel):
#     a = {"learning_rate": 0.01, "subsample": 1, "max_depth": 10}


# class ParamGridSchema(BaseModel):
#     a = {
#         "learning_rate": [0.01, 0.05, 0.1],
#         "n_estimators": [50, 100, 150],
#         "subsample": [0.5, 0.9],
#     }


class GridSearchCVParamsSchema(BaseModel):
    param_grid: Dict = Field()
    cv: int = Field()
    n_jobs: int = Field()
    verbose: int = Field()
    # a = {
    #     "param_grid": {
    #         "learning_rate": [0.01, 0.05, 0.1],
    #         "n_estimators": [50, 100, 150],
    #         "subsample": [0.5, 0.9],
    #     },
    #     "cv": 2,
    #     "n_jobs": -1,
    #     "verbose": 1,
    # }


class TrainBinaryClassificationSchema(BaseModel):
    input_data_filename_path: Optional[str] = Field(
        default="/output/dataset_SCL_complete.csv"
    )
    features_filter: Optional[List] = Field(default=None)
    categorical_features: Optional[List] = Field(default=["OPERA", "MES", "TIPOVUELO"])
    numerical_features: Optional[List] = Field(default=None)
    minmax_scaler_numerical_f: Optional[bool] = Field(default=False)
    label: Optional[str] = Field(default="atraso_15")
    model_name: str = Field(default="xgb")
    destination_storage_name: Optional[str] = Field(
        default=f"{model_name}_{datetime.now().strftime(format='%Y-%m-%d-%H%M')}"
    )
    model_custom_params: Optional[Dict] = Field(default=None)
    grid_search_cv: Optional[bool] = Field(default=False)
    grid_search_cv_params: Optional[GridSearchCVParamsSchema] = Field(default=None)
    train_test_split_data: Optional[bool] = Field(default=True)
    random_state: Optional[int] = Field(default=None)
    shuffle_data: Optional[bool] = Field(default=True)
    shuffle_features: Optional[List] = Field(
        default=["OPERA", "MES", "TIPOVUELO", "SIGLADES", "DIANOM", "atraso_15"]
    )
    endpoint_test_mode: Optional[bool] = Field(default=False)

    class Config:
        schema_extra = {
            "example": {
                "input_data_filename_path": "/output/dataset_SCL_complete.csv",
                "features_filter": [
                    "MES_7",
                    "TIPOVUELO_I",
                    "OPERA_Copa Air",
                    "OPERA_Latin American Wings",
                    "MES_12",
                    "OPERA_Grupo LATAM",
                    "MES_10",
                    "OPERA_JetSmart SPA",
                    "OPERA_Air Canada",
                    "MES_9",
                    "OPERA_American Airlines",
                ],
                "categorical_features": ["OPERA", "MES", "TIPOVUELO"],
                "numerical_features": [],
                "minmax_scaler_numerical_f": True,
                "label": "atraso_15",
                "model_name": "xgb",
                "destination_storage_name": "My_XGB_Classifier_model",
                "model_custom_params": {
                    "learning_rate": 0.01,
                    "subsample": 1,
                    "max_depth": 10,
                },
                "grid_search_cv": True,
                "grid_search_cv_params": {
                    "param_grid": {
                        "learning_rate": [0.01, 0.05, 0.1],
                        "n_estimators": [50, 100, 150],
                        "subsample": [0.5, 0.9],
                    },
                    "cv": 2,
                    "n_jobs": -1,
                    "verbose": 1,
                },
                "train_test_split_data": True,
                "random_state": 10,
                "shuffle_data": True,
                "shuffle_features": [
                    "OPERA",
                    "MES",
                    "TIPOVUELO",
                    "SIGLADES",
                    "DIANOM",
                    "atraso_15",
                ],
                "endpoint_test_mode": False,
            }
        }
