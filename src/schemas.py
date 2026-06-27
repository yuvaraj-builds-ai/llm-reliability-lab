from pydantic import BaseModel, ConfigDict, model_validator

class Chunk(BaseModel):
    id: str
    text: str
    source: str
    heading: str

class RetrievedChunk(Chunk):
    score: float

class AnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    citations: list[str]
    refused: bool

    @model_validator(mode="after")
    def check_refusal(self):
        if not self.answer.strip():
            raise ValueError("Answer cannot be empty")
        if self.refused and self.citations:
            raise ValueError("Refusals cannot have citations")
        return self
