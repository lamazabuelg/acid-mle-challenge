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
