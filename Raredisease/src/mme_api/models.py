from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class GenomicVariant(BaseModel):
    assembly: str = Field(..., example="GRCh38")
    chr: str = Field(..., example="17")
    start: int = Field(..., example=41244966)
    end: int = Field(..., example=41244968)
    reference: str = Field(..., example="G")
    alternate: str = Field(..., example="C")

class Gene(BaseModel):
    id: str = Field(..., example="HGNC:12345")

class GenomicFeature(BaseModel):
    gene: Gene
    variant: Optional[GenomicVariant] = None

class PhenotypicFeature(BaseModel):
    type: dict = Field(..., example={"id": "HP:0001250"})
    severity: Optional[dict] = Field(None, example={"id": "HP:0012828"})
    onset: Optional[datetime] = None

class Patient(BaseModel):
    id: str = Field(..., example="MME-2025-ABC")
    genomicFeatures: List[GenomicFeature]
    phenotypicFeatures: List[PhenotypicFeature]

class MatchRequest(BaseModel):
    patient: Patient

class MatchResponse(BaseModel):
    results: List[Patient]
    score: float = Field(..., ge=0.0, le=1.0)
    created: datetime = Field(default_factory=datetime.now) 