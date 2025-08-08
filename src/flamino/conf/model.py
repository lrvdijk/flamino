from typing import Any

from pydantic import BaseModel, ConfigDict


class Model(BaseModel):
    model_config = ConfigDict(extra='allow')
    __pydantic_extra__: dict[str, Any]

    name: str
