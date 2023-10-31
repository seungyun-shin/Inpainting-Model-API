from typing import List, Tuple
from pydantic import BaseModel

class InpaintModel(BaseModel):
    img : str
    coord : List[List[Tuple[int, int]]]

class InpaintResponse(BaseModel):
    status: str
    result_path: str