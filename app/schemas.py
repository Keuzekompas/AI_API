from pydantic import BaseModel
from .utils import sanitize_text

class StudentInput(BaseModel):
    description: str         
    preferred_location: str | None = None
    current_ects: int | None = None
    tags: list[str] = []

    def sanitize(self):
        self.description = sanitize_text(self.description)
        self.tags = [sanitize_text(t) for t in self.tags]
        if self.preferred_location:
            self.preferred_location = sanitize_text(self.preferred_location)
