from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from typing import List
import hashlib
import time
from datetime import datetime, timedelta

from .models import MatchRequest, MatchResponse, Patient
from .matching import calculate_phenotype_similarity, filter_genomic_matches

app = FastAPI(
    title="MME API v2.3",
    description="Matchmaker Exchange API implementation for rare disease matching",
    version="2.3.0"
)

# Authentication setup
API_KEY_HEADER = APIKeyHeader(name="X-Auth-Token", auto_error=True)
VALID_TOKENS = {}  # In production, use a proper token store

def verify_token(token: str = Security(API_KEY_HEADER)):
    if token not in VALID_TOKENS:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return token

def generate_token(institution_id: str) -> str:
    """Generate SHA-1 based authentication token"""
    timestamp = str(int(time.time()))
    token = hashlib.sha1(f"{institution_id}:{timestamp}".encode()).hexdigest()
    VALID_TOKENS[token] = {
        "institution": institution_id,
        "created": datetime.now()
    }
    return token

@app.post("/match", response_model=List[MatchResponse])
async def match_patient(
    request: MatchRequest,
    token: str = Depends(verify_token)
):
    """
    Match a patient against the database using phenotype and genomic features
    """
    try:
        # Calculate phenotype similarity using Wu-Palmer distance
        phenotype_matches = calculate_phenotype_similarity(
            request.patient.phenotypicFeatures,
            threshold=0.65
        )
        
        # Filter matches based on genomic features
        genomic_matches = filter_genomic_matches(
            request.patient.genomicFeatures,
            phenotype_matches,
            min_score=0.78
        )
        
        return [
            MatchResponse(
                results=[match],
                score=score,
                created=datetime.now()
            )
            for match, score in genomic_matches
        ]
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing match request: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.3.0"}

# Token management endpoints
@app.post("/token/{institution_id}")
async def create_token(institution_id: str):
    """Create a new authentication token for an institution"""
    token = generate_token(institution_id)
    return {"token": token}

@app.delete("/token/{token}")
async def revoke_token(token: str):
    """Revoke an authentication token"""
    if token in VALID_TOKENS:
        del VALID_TOKENS[token]
        return {"status": "token revoked"}
    raise HTTPException(
        status_code=404,
        detail="Token not found"
    ) 