from enum import Enum
from pydantic import BaseModel


# --- Pydantic Models for Request/Response ---
class Gender(str, Enum):
    female = "female"
    male = "male"

    def __str__(self) -> str:
        return self.value


class TTSRequest(BaseModel):
    text: str
    gender: Gender  # Using the Enum here will auto-validate the input


class STTResponse(BaseModel):
    text: str