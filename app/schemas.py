from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from .utils import sanitize_text

class LanguageEnum(str, Enum):
    NL = "NL"
    EN = "EN"
    
class StudentInput(BaseModel):
    description: str = Field(..., max_length=1000)
    preferred_location: str | None = None
    current_ects: int | None = None
    tags: list[str] = []

    def sanitize(self):
        self.description = sanitize_text(self.description)
        self.tags = [sanitize_text(t) for t in self.tags]
        if self.preferred_location:
            self.preferred_location = sanitize_text(self.preferred_location)

class ModuleDetails(BaseModel):
    ects: int
    location: str

class RecommendationEntry(BaseModel):
    ID: str
    Module_Name: str
    Description: str
    Score: float
    AI_Reason: str
    Details: ModuleDetails

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationEntry]
    language: LanguageEnum
