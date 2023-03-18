from typing import Optional, List
from pydantic import BaseModel, Field


class CreateFeaturesSchema(BaseModel):
    generate_both_files: bool = Field(default=True)
    generate_files: Optional[List] = Field()
    test_mode: Optional[bool] = Field(default=False)
    test_size: Optional[int] = Field(default=100)
    test_random_state: Optional[int] = Field()

    class config:
        schema_extra = {
            "generate_both_files": True,
            "generate_files": None,
            "test_mode": False,
            "test_size": 100,
            "test_random_state": 10,
        }


class TrainTestSplitSchema(BaseModel):
    data_filename: str = Field()
    features_filter: Optional[List] = Field(default=None)
    categorical_features: Optional[List] = Field(default=["OPERA", "MES", "TIPOVUELO"])
    numerical_features: Optional[List] = Field(default=None)
    minmax_scaler_numerical_f: Optional[bool] = Field(default=False)
    label: Optional[str] = Field(default="atraso_15")
    shuffle_data: Optional[bool] = Field(default=True)
    shuffle_features: Optional[List] = Field(
        default=["OPERA", "MES", "TIPOVUELO", "SIGLADES", "DIANOM", "atraso_15"]
    )
    sample_data: Optional[int] = Field(default=None)
    random_state: Optional[int] = Field(default=None)

    class Config:
        schema_extra = {
            "example": {
                "data_filename": "output/dataset_SCL_complete.csv",
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
                "shuffle_data": True,
                "shuffle_features": [
                    "OPERA",
                    "MES",
                    "TIPOVUELO",
                    "SIGLADES",
                    "DIANOM",
                    "atraso_15",
                ],
                "sample_data": None,
                "random_state": 10,
            }
        }
