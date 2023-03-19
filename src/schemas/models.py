from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class PredictionSchema(BaseModel):
    model_filename: str = Field()
    X_test_filename: str = Field()

    class Config:
        schema_extra = {
            "example": {
                "model_filename": "My_XGB_Classifier_model",
                "X_test_filename": "dataset_SCL_complete-X_test.csv",
            },
        }


class GridSearchCVParamsSchema(BaseModel):
    param_grid: Dict = Field()
    cv: int = Field()
    n_jobs: int = Field()
    verbose: int = Field()


class TrainBinaryClassificationSchema(BaseModel):
    X_train_filename: str = Field()
    y_train_filename: str = Field()
    model_name: str = Field(default="xgb")
    destination_model_name: Optional[str] = Field(
        default=f"{model_name}_{datetime.now().strftime(format='%Y-%m-%d-%H%M')}"
    )
    model_custom_params: Optional[Dict] = Field(default=None)
    grid_search_cv: Optional[bool] = Field(default=False)
    grid_search_cv_params: Optional[GridSearchCVParamsSchema] = Field(default=None)
    random_state: Optional[int] = Field(default=None)

    class Config:
        schema_extra = {
            "example": {
                "X_train_filename": "dataset_SCL_complete-X_train.csv",
                "y_train_filename": "dataset_SCL_complete-y_train.csv",
                "model_name": "xgb",
                "destination_model_name": "My_XGB_Classifier_model",
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
                "random_state": 10,
            }
        }
