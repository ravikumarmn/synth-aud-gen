"""
FastAPI service for Audience Characteristics Generation.

Provides REST API endpoints to generate detailed audience member characteristics
using persona templates and screener questions as input to Azure OpenAI LLM.
"""

import json
import time
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from generate_audience_characteristics import (
    _create_azure_client,
    generate_audience_characteristics,
)

app = FastAPI(
    title="Audience Characteristics Generator API",
    description="Generate detailed audience member characteristics using Azure OpenAI",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---


class ScreenerQuestion(BaseModel):
    """Screener question and answer pair."""

    question: str
    answer: str


class Persona(BaseModel):
    """Base persona information."""

    id: str | None = None
    personaName: str = ""
    personaType: str = ""
    gender: str = ""
    age: int | None = None
    location: str = ""
    ethnicity: str = ""
    about: str = ""
    goalsAndMotivations: str = ""
    frustrations: str = ""
    needState: str = ""
    occasions: str = ""


class AudienceInput(BaseModel):
    """Single audience input with persona and screener questions."""

    persona: Persona
    screenerQuestions: list[ScreenerQuestion] = Field(default_factory=list)
    sampleSize: int = Field(default=1, ge=1, le=100)


class GenerationRequest(BaseModel):
    """Request body for audience characteristics generation."""

    projectName: str | None = None
    projectDescription: str | None = None
    projectId: str | None = None
    userId: str | None = None
    requestId: str | None = None
    audiences: list[AudienceInput]
    maxConcurrent: int = Field(default=10, ge=1, le=50)


class GeneratedMember(BaseModel):
    """Generated audience member."""

    member_id: str
    about: str | None = None
    goals_and_motivations: list[str] | None = None
    frustrations: list[str] | None = None
    need_state: str | None = None
    occasions: str | None = None
    generation_error: str | None = None


class GenerationStats(BaseModel):
    """Statistics for generation process."""

    total_members: int
    successfully_generated: int
    failed: int


class AudienceMetadata(BaseModel):
    """Metadata for a generated audience."""

    audience_index: int
    sample_size: int
    persona: dict[str, Any]
    screener_questions: list[dict[str, Any]]
    generation_stats: GenerationStats


class GeneratedAudience(BaseModel):
    """Generated audience with members and metadata."""

    generated_audience: list[GeneratedMember]
    metadata: AudienceMetadata


class GenerationResponse(BaseModel):
    """Response body for audience characteristics generation."""

    project_name: str | None = None
    project_description: str | None = None
    project_id: str | None = None
    user_id: str | None = None
    request_id: str | None = None
    generation_model: str
    provider: str
    total_audiences: int
    total_members_processed: int
    total_successfully_generated: int
    total_failed: int
    processing_time_seconds: float
    audiences: list[GeneratedAudience]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str


# --- API Endpoints ---


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="audience-characteristics-generator",
        version="1.0.0",
    )


@app.post(
    "/generate",
    response_model=GenerationResponse,
    tags=["Generation"],
    summary="Generate audience characteristics",
    description="Generate detailed characteristics for audience members based on persona templates and screener questions.",
)
async def generate_characteristics(
    request: GenerationRequest,
) -> GenerationResponse:
    """
    Generate audience characteristics for all provided audiences.

    This endpoint accepts audience data with personas and screener questions,
    then generates detailed characteristics for each audience member using Azure OpenAI.
    """
    start_time = time.time()

    # Initialize Azure OpenAI client
    try:
        client, deployment = _create_azure_client()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Azure OpenAI not configured: {str(e)}",
        )

    try:
        # Convert request to dict format expected by generation functions
        enriched_audiences: list[dict[str, Any]] = []

        for idx, audience in enumerate(request.audiences):
            audience_data = {
                "persona": audience.persona.model_dump(),
                "screenerQuestions": [
                    q.model_dump() for q in audience.screenerQuestions
                ],
                "sampleSize": audience.sampleSize,
            }

            result = await generate_audience_characteristics(
                client=client,
                deployment=deployment,
                audience_data=audience_data,
                audience_index=idx,
                max_concurrent=request.maxConcurrent,
            )
            enriched_audiences.append(result)

        # Calculate totals
        total_generated = sum(
            aud["metadata"]["generation_stats"]["successfully_generated"]
            for aud in enriched_audiences
        )
        total_failed = sum(
            aud["metadata"]["generation_stats"]["failed"] for aud in enriched_audiences
        )

        processing_time = time.time() - start_time

        # Close client
        await client.close()

        return GenerationResponse(
            project_name=request.projectName,
            project_description=request.projectDescription,
            project_id=request.projectId,
            user_id=request.userId,
            request_id=request.requestId,
            generation_model=f"azure:{deployment}",
            provider="azure",
            total_audiences=len(enriched_audiences),
            total_members_processed=total_generated + total_failed,
            total_successfully_generated=total_generated,
            total_failed=total_failed,
            processing_time_seconds=round(processing_time, 2),
            audiences=enriched_audiences,
        )

    except Exception as e:
        await client.close()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )


@app.post(
    "/generate/audience",
    response_model=GeneratedAudience,
    tags=["Generation"],
    summary="Generate audience",
    description="Generate audience members based on persona and screener questions.",
)
async def generate_audience(
    audience: AudienceInput,
    max_concurrent: int = 10,
) -> GeneratedAudience:
    """
    Generate audience members.

    Endpoint for generating audience members for one audience.
    """
    # Initialize Azure OpenAI client
    try:
        client, deployment = _create_azure_client()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Azure OpenAI not configured: {str(e)}",
        )

    try:
        audience_data = {
            "persona": audience.persona.model_dump(),
            "screenerQuestions": [q.model_dump() for q in audience.screenerQuestions],
            "sampleSize": audience.sampleSize,
        }

        result = await generate_audience_characteristics(
            client=client,
            deployment=deployment,
            audience_data=audience_data,
            audience_index=0,
            max_concurrent=max_concurrent,
        )

        await client.close()
        return result

    except Exception as e:
        await client.close()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )


@app.post(
    "/generate/file",
    response_model=GenerationResponse,
    tags=["Generation"],
    summary="Generate audience from file",
    description="Upload a JSON file with persona input data and generate audience characteristics.",
)
async def generate_from_file(
    file: UploadFile = File(..., description="JSON file with persona input data"),
    max_concurrent: int = 10,
) -> GenerationResponse:
    """
    Generate audience characteristics from an uploaded JSON file.

    The file should follow the persona_input.json format with:
    - projectName, projectDescription, projectId, userId, requestId (optional metadata)
    - audiences[]: Array of audience data with persona, screenerQuestions, sampleSize
    """
    # Validate file type
    if not file.filename or not file.filename.endswith(".json"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a JSON file (.json extension)",
        )

    # Read and parse file content
    try:
        content = await file.read()
        data = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON file: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {str(e)}",
        )

    # Validate required structure
    audiences = data.get("audiences", [])
    if not audiences:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must contain 'audiences' array with at least one audience",
        )

    start_time = time.time()

    # Initialize Azure OpenAI client
    try:
        client, deployment = _create_azure_client()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Azure OpenAI not configured: {str(e)}",
        )

    try:
        enriched_audiences: list[dict[str, Any]] = []

        for idx, audience_data in enumerate(audiences):
            # Extract and normalize audience data from file format
            persona = audience_data.get("persona", {})
            screener_questions = audience_data.get("screenerQuestions", [])
            sample_size = audience_data.get("sampleSize", 1)

            normalized_audience = {
                "persona": persona,
                "screenerQuestions": screener_questions,
                "sampleSize": sample_size,
            }

            result = await generate_audience_characteristics(
                client=client,
                deployment=deployment,
                audience_data=normalized_audience,
                audience_index=idx,
                max_concurrent=max_concurrent,
            )
            enriched_audiences.append(result)

        # Calculate totals
        total_generated = sum(
            aud["metadata"]["generation_stats"]["successfully_generated"]
            for aud in enriched_audiences
        )
        total_failed = sum(
            aud["metadata"]["generation_stats"]["failed"] for aud in enriched_audiences
        )

        processing_time = time.time() - start_time

        await client.close()

        # Convert IDs to strings if they exist (input may have int or string)
        project_id = data.get("projectId")
        user_id = data.get("userId")

        return GenerationResponse(
            project_name=data.get("projectName"),
            project_description=data.get("projectDescription"),
            project_id=str(project_id) if project_id is not None else None,
            user_id=str(user_id) if user_id is not None else None,
            request_id=data.get("requestId"),
            generation_model=f"azure:{deployment}",
            provider="azure",
            total_audiences=len(enriched_audiences),
            total_members_processed=total_generated + total_failed,
            total_successfully_generated=total_generated,
            total_failed=total_failed,
            processing_time_seconds=round(processing_time, 2),
            audiences=enriched_audiences,
        )

    except Exception as e:
        await client.close()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
