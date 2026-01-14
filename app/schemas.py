from pydantic import BaseModel, Field, field_validator
from typing import List
from enum import Enum
from .utils import sanitize_text

class LanguageEnum(str, Enum):
    NL = "NL"
    EN = "EN"
    
class StudentInput(BaseModel):
    description: str = Field(..., min_length=10, max_length=1000)
    preferred_location: str | None = None
    current_ects: int = Field(...)
    tags: list[str] = []
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v):
        if not (1 <= len(v) <= 3):
            raise ValueError('You must provide between 1 and 3 tags')
        return v
    
    @field_validator('current_ects')
    @classmethod
    def validate_ects(cls, v):
        if v not in [15, 30]:
            raise ValueError('The ecs must be 15 or 30')
        return v

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
