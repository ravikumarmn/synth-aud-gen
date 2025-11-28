"""FastAPI service for Audience Characteristics Generation."""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from azure.storage.blob import (
    BlobSasPermissions,
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
)
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from generate_audience_characteristics import (
    _create_azure_client,
    generate_audience_characteristics,
)

# Load environment variables
load_dotenv()

# Logging setup
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audience-generator")

# Constants
OUTPUT_CONTAINER = "generated-synthetic-audience"

app = FastAPI(
    title="Audience Characteristics Generator API",
    description="Generate detailed audience member characteristics using Azure OpenAI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    generation_time_seconds: float
    created_at: str


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


class BlobProcessRequest(BaseModel):
    """Request body for blob-based generation."""

    input_blob_url: str  # Full blob URL to input JSON
    output_blob_prefix: str = "audience_output"  # Prefix for output JSON
    max_concurrent: int = Field(default=10, ge=1, le=50)


class BlobProcessResponse(BaseModel):
    """Response for blob-based generation."""

    status: str
    input_blob: str
    output_blob: str  # SAS URL for output
    total_audiences: int
    total_members_processed: int
    total_successfully_generated: int
    total_failed: int
    processing_time_seconds: float


# Helper: parse container + blob name from full blob URL
def parse_blob_url(blob_url: str) -> tuple[str, str]:
    """Parse container and blob name from full blob URL."""
    parsed = urlparse(blob_url)
    path = parsed.path.lstrip("/")

    if "/" not in path:
        raise ValueError("Invalid blob URL format. Expected /container/blobname")

    container, blob_name = path.split("/", 1)
    return unquote(container), unquote(blob_name)


# Helper: create timestamped output blob name
def generate_output_blob_name(prefix: str) -> str:
    """Generate unique output blob name with timestamp and UUID."""
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    short_uuid = uuid4().hex[:8]
    safe_prefix = prefix.replace(" ", "_")
    return f"{safe_prefix}_{stamp}_{short_uuid}.json"


# Helper: parse AccountName and AccountKey from connection string
def parse_account_from_connection_string(
    conn_str: str,
) -> tuple[str | None, str | None]:
    """Extract account name and key from connection string."""
    parts = {}
    for segment in conn_str.split(";"):
        if "=" in segment:
            k, v = segment.split("=", 1)
            parts[k.strip()] = v.strip()
    account_name = parts.get("AccountName")
    account_key = parts.get("AccountKey")
    return account_name, account_key


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
async def generate_characteristics(request: GenerationRequest) -> GenerationResponse:
    """Generate audience characteristics for all provided audiences."""
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
        audience_data_list = [
            {
                "persona": audience.persona.model_dump(),
                "screenerQuestions": [
                    q.model_dump() for q in audience.screenerQuestions
                ],
                "sampleSize": audience.sampleSize,
            }
            for audience in request.audiences
        ]

        # Run all audiences in parallel
        tasks = [
            generate_audience_characteristics(
                client=client,
                deployment=deployment,
                audience_data=audience_data,
                audience_index=idx,
                max_concurrent=request.maxConcurrent,
            )
            for idx, audience_data in enumerate(audience_data_list)
        ]
        enriched_audiences = await asyncio.gather(*tasks)

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
    """Generate audience members for a single audience."""
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
        audience_start_time = time.time()
        result = await generate_audience_characteristics(
            client=client,
            deployment=deployment,
            audience_data=audience_data,
            audience_index=0,
            max_concurrent=max_concurrent,
        )
        audience_time = time.time() - audience_start_time
        result["metadata"]["generation_time_seconds"] = round(audience_time, 2)
        result["metadata"]["created_at"] = datetime.now(timezone.utc).isoformat()

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
        normalized_audiences = [
            {
                "persona": aud.get("persona", {}),
                "screenerQuestions": aud.get("screenerQuestions", []),
                "sampleSize": aud.get("sampleSize", 1),
            }
            for aud in audiences
        ]

        tasks = [
            generate_audience_characteristics(
                client=client,
                deployment=deployment,
                audience_data=aud,
                audience_index=idx,
                max_concurrent=max_concurrent,
            )
            for idx, aud in enumerate(normalized_audiences)
        ]
        enriched_audiences = await asyncio.gather(*tasks)

        total_generated = sum(
            aud["metadata"]["generation_stats"]["successfully_generated"]
            for aud in enriched_audiences
        )
        total_failed = sum(
            aud["metadata"]["generation_stats"]["failed"] for aud in enriched_audiences
        )

        processing_time = time.time() - start_time

        await client.close()

        project_id = data.get("projectId")
        user_id = data.get("userId")
        response = GenerationResponse(
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

        original_name = file.filename or "output"
        base_name = original_name.rsplit(".", 1)[0]
        output_filename = f"data/{base_name}_output.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(response.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"Output saved to: {output_filename}")

        return response

    except Exception as e:
        await client.close()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )


@app.post(
    "/generate/blob",
    response_model=BlobProcessResponse,
    tags=["Generation"],
    summary="Generate audience from blob URL",
    description="Process a JSON file from Azure Blob Storage URL and upload results to blob storage.",
)
async def generate_from_blob(req: BlobProcessRequest) -> BlobProcessResponse:
    """
    Generate audience characteristics from a blob URL.

    Downloads input JSON from the provided blob URL, processes it,
    uploads the output to 'generated-synthetic-audience' container,
    and returns a SAS URL for the output.
    """
    logger.info(f"Received request for blob: {req.input_blob_url}")
    start_time = time.time()

    # Get connection string from env
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AZURE_STORAGE_CONNECTION_STRING not set",
        )

    # Parse account info (needed for SAS)
    account_name, account_key = parse_account_from_connection_string(conn_str)
    if not account_name or not account_key:
        logger.error(
            "AccountName or AccountKey missing in AZURE_STORAGE_CONNECTION_STRING. "
            "SAS URL generation requires a connection string with AccountKey."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AccountName or AccountKey missing in AZURE_STORAGE_CONNECTION_STRING (cannot generate SAS).",
        )

    # Create BlobServiceClient
    try:
        blob_service = BlobServiceClient.from_connection_string(conn_str)
    except Exception as e:
        logger.exception("Failed to create BlobServiceClient")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to create BlobServiceClient: {e}",
        )

    # Parse input blob URL
    try:
        container, blob_name = parse_blob_url(req.input_blob_url)
        logger.info(f"Parsed container={container}, blob={blob_name}")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input blob URL: {e}",
        )

    tmp_dir = None
    client = None

    try:
        # Create a temp dir (unique per request)
        tmp_dir = tempfile.mkdtemp(prefix="audience_gen_")

        # ---------- Download input blob ----------
        input_uuid = uuid4().hex[:8]
        original_filename = Path(blob_name).name
        local_input_path = os.path.join(tmp_dir, f"{input_uuid}_{original_filename}")

        try:
            blob_client = blob_service.get_blob_client(
                container=container,
                blob=blob_name,
            )
            logger.info(f"Downloading blob to {local_input_path}")
            data = blob_client.download_blob().readall()
            with open(local_input_path, "wb") as f:
                f.write(data)
            logger.info("Download complete")
        except Exception as e:
            logger.exception("Blob download failed")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download input blob: {e}",
            )

        # ---------- Parse input JSON ----------
        try:
            with open(local_input_path, "r", encoding="utf-8") as f:
                input_data = json.load(f)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON in input blob: {e}",
            )

        # Validate required structure
        audiences = input_data.get("audiences", [])
        if not audiences:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input JSON must contain 'audiences' array with at least one audience",
            )

        # ---------- Initialize Azure OpenAI client ----------
        try:
            client, deployment = _create_azure_client()
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Azure OpenAI not configured: {str(e)}",
            )

        # ---------- Run generation ----------
        normalized_audiences = [
            {
                "persona": aud.get("persona", {}),
                "screenerQuestions": aud.get("screenerQuestions", []),
                "sampleSize": aud.get("sampleSize", 1),
            }
            for aud in audiences
        ]

        tasks = [
            generate_audience_characteristics(
                client=client,
                deployment=deployment,
                audience_data=aud,
                audience_index=idx,
                max_concurrent=req.max_concurrent,
            )
            for idx, aud in enumerate(normalized_audiences)
        ]
        enriched_audiences = await asyncio.gather(*tasks)

        total_generated = sum(
            aud["metadata"]["generation_stats"]["successfully_generated"]
            for aud in enriched_audiences
        )
        total_failed = sum(
            aud["metadata"]["generation_stats"]["failed"] for aud in enriched_audiences
        )

        processing_time = time.time() - start_time

        await client.close()
        client = None

        # ---------- Build output response ----------
        project_id = input_data.get("projectId")
        user_id = input_data.get("userId")
        output_data = {
            "project_name": input_data.get("projectName"),
            "project_description": input_data.get("projectDescription"),
            "project_id": str(project_id) if project_id is not None else None,
            "user_id": str(user_id) if user_id is not None else None,
            "request_id": input_data.get("requestId"),
            "generation_model": f"azure:{deployment}",
            "provider": "azure",
            "total_audiences": len(enriched_audiences),
            "total_members_processed": total_generated + total_failed,
            "total_successfully_generated": total_generated,
            "total_failed": total_failed,
            "processing_time_seconds": round(processing_time, 2),
            "audiences": enriched_audiences,
        }

        # ---------- Write output to temp file ----------
        output_blob_name = generate_output_blob_name(req.output_blob_prefix)
        local_output_path = os.path.join(tmp_dir, output_blob_name)

        with open(local_output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info("Generation completed, output written to temp file")

        # ---------- Ensure output container exists ----------
        try:
            container_client = blob_service.get_container_client(OUTPUT_CONTAINER)
            try:
                container_client.get_container_properties()
                logger.info(f"Container '{OUTPUT_CONTAINER}' already exists")
            except Exception:
                container_client.create_container()
                logger.info(f"Created container '{OUTPUT_CONTAINER}'")
        except Exception as e:
            logger.exception("Failed to ensure container exists")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to ensure output container exists: {e}",
            )

        # ---------- Upload output JSON ----------
        try:
            output_blob_client = blob_service.get_blob_client(
                container=OUTPUT_CONTAINER,
                blob=output_blob_name,
            )
            logger.info(
                f"Uploading output JSON to container={OUTPUT_CONTAINER}, "
                f"blob={output_blob_name}"
            )
            with open(local_output_path, "rb") as f:
                output_blob_client.upload_blob(
                    f,
                    overwrite=True,
                    content_settings=ContentSettings(content_type="application/json"),
                )
            logger.info("Upload complete")
        except Exception as e:
            logger.exception("Output upload failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload output JSON: {e}",
            )

        # ---------- Generate SAS URL for the output blob ----------
        try:
            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=OUTPUT_CONTAINER,
                blob_name=output_blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(days=365),  # 1 year validity
            )

            if not sas_token:
                raise RuntimeError("generate_blob_sas returned empty token")

            output_url = f"{output_blob_client.url}?{sas_token}"
            logger.info("Generated SAS URL for output blob")
        except Exception as e:
            logger.exception("Failed to generate SAS URL")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate SAS URL: {e}",
            )

        # ---------- SUCCESS RESPONSE ----------
        return BlobProcessResponse(
            status="success",
            input_blob=req.input_blob_url,
            output_blob=output_url,
            total_audiences=len(enriched_audiences),
            total_members_processed=total_generated + total_failed,
            total_successfully_generated=total_generated,
            total_failed=total_failed,
            processing_time_seconds=round(processing_time, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during processing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {e}",
        )

    finally:
        if client:
            try:
                await client.close()
            except Exception:
                pass
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("Temporary folder cleaned up")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
