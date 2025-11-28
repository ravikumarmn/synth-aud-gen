"""Azure Blob Storage integration for uploading files and updating persona URLs."""

import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import requests
from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureBlobStorageConfig:
    """Configuration for Azure Blob Storage."""

    def __init__(
        self,
        account_name: str | None = None,
        account_key: str | None = None,
        container_name: str | None = None,
        connection_string: str | None = None,
    ):
        self.account_name = account_name or os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.account_key = account_key or os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        self.container_name = container_name or os.getenv(
            "AZURE_STORAGE_CONTAINER_NAME", "audience-characteristics"
        )
        self.connection_string = connection_string or os.getenv(
            "AZURE_STORAGE_CONNECTION_STRING"
        )

        if not self.account_name or not self.account_key:
            if not self.connection_string:
                raise ValueError(
                    "Azure Storage credentials not configured. "
                    "Set AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY, "
                    "or AZURE_STORAGE_CONNECTION_STRING environment variables."
                )


class AzureBlobStorageClient:
    """Client for Azure Blob Storage operations."""

    def __init__(self, config: AzureBlobStorageConfig | None = None):
        self.config = config or AzureBlobStorageConfig()
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the BlobServiceClient."""
        if self.config.connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.config.connection_string
            )
        else:
            account_url = f"https://{self.config.account_name}.blob.core.windows.net"
            self.blob_service_client = BlobServiceClient(
                account_url=account_url,
                credential=self.config.account_key,
            )
        logger.info(
            "Azure Blob Storage client initialized for account: %s",
            self.config.account_name,
        )

    def ensure_container_exists(self, container_name: str | None = None) -> str:
        """Create container if it doesn't exist."""
        container = container_name or self.config.container_name
        container_client = self.blob_service_client.get_container_client(container)

        try:
            container_client.get_container_properties()
            logger.info("Container '%s' already exists", container)
        except Exception:
            container_client.create_container()
            logger.info("Created container '%s'", container)

        return container

    def upload_file(
        self,
        file_path: str,
        blob_name: str | None = None,
        container_name: str | None = None,
        overwrite: bool = True,
    ) -> str:
        """
        Upload a file to Azure Blob Storage.

        Args:
            file_path: Path to the local file to upload.
            blob_name: Name for the blob. If None, generates a unique name.
            container_name: Container to upload to. Uses default if None.
            overwrite: Whether to overwrite existing blob.

        Returns:
            The blob URL (without SAS token).
        """
        container = container_name or self.config.container_name
        self.ensure_container_exists(container)

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if blob_name is None:
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            unique_id = uuid.uuid4().hex[:8]
            blob_name = (
                f"{file_path_obj.stem}_{timestamp}_{unique_id}{file_path_obj.suffix}"
            )

        blob_client = self.blob_service_client.get_blob_client(
            container=container, blob=blob_name
        )

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=overwrite)

        logger.info("Uploaded file '%s' to blob '%s'", file_path, blob_name)
        return blob_client.url

    def upload_data(
        self,
        data: bytes | str,
        blob_name: str,
        container_name: str | None = None,
        overwrite: bool = True,
    ) -> str:
        """
        Upload data directly to Azure Blob Storage.

        Args:
            data: Bytes or string data to upload.
            blob_name: Name for the blob.
            container_name: Container to upload to. Uses default if None.
            overwrite: Whether to overwrite existing blob.

        Returns:
            The blob URL (without SAS token).
        """
        container = container_name or self.config.container_name
        self.ensure_container_exists(container)

        if isinstance(data, str):
            data = data.encode("utf-8")

        blob_client = self.blob_service_client.get_blob_client(
            container=container, blob=blob_name
        )
        blob_client.upload_blob(data, overwrite=overwrite)

        logger.info("Uploaded data to blob '%s'", blob_name)
        return blob_client.url

    def generate_sas_url(
        self,
        blob_name: str,
        container_name: str | None = None,
        expiry_days: int = 365,
        permission: str = "r",
    ) -> str:
        """
        Generate a SAS URL for a blob.

        Args:
            blob_name: Name of the blob.
            container_name: Container name. Uses default if None.
            expiry_days: Number of days until SAS token expires.
            permission: Permission string (r=read, w=write, d=delete).

        Returns:
            Full SAS URL for the blob.
        """
        container = container_name or self.config.container_name

        permissions = BlobSasPermissions(
            read="r" in permission,
            write="w" in permission,
            delete="d" in permission,
        )

        sas_token = generate_blob_sas(
            account_name=self.config.account_name,
            container_name=container,
            blob_name=blob_name,
            account_key=self.config.account_key,
            permission=permissions,
            expiry=datetime.utcnow() + timedelta(days=expiry_days),
        )

        if not sas_token:
            raise RuntimeError("generate_blob_sas returned empty token")

        blob_client = self.blob_service_client.get_blob_client(
            container=container, blob=blob_name
        )
        sas_url = f"{blob_client.url}?{sas_token}"

        logger.info(
            "Generated SAS URL for blob '%s' (expires in %d days)",
            blob_name,
            expiry_days,
        )
        return sas_url

    def upload_and_get_sas_url(
        self,
        file_path: str,
        blob_name: str | None = None,
        container_name: str | None = None,
        expiry_days: int = 365,
    ) -> dict:
        """
        Upload a file and generate a SAS URL for it.

        Args:
            file_path: Path to the local file to upload.
            blob_name: Name for the blob. If None, generates a unique name.
            container_name: Container to upload to. Uses default if None.
            expiry_days: Number of days until SAS token expires.

        Returns:
            Dict with blob_url, sas_url, blob_name, and container_name.
        """
        container = container_name or self.config.container_name

        file_path_obj = Path(file_path)
        if blob_name is None:
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            unique_id = uuid.uuid4().hex[:8]
            blob_name = (
                f"{file_path_obj.stem}_{timestamp}_{unique_id}{file_path_obj.suffix}"
            )

        blob_url = self.upload_file(
            file_path=file_path,
            blob_name=blob_name,
            container_name=container,
        )

        sas_url = self.generate_sas_url(
            blob_name=blob_name,
            container_name=container,
            expiry_days=expiry_days,
        )

        return {
            "blob_url": blob_url,
            "sas_url": sas_url,
            "blob_name": blob_name,
            "container_name": container,
        }


def update_persona_file_url(
    project_id: int,
    sas_url: str,
    api_base_url: str | None = None,
) -> dict:
    """
    Call the middleware API to update the persona file URL.

    Args:
        project_id: The project ID to update.
        sas_url: The SAS URL for the uploaded blob.
        api_base_url: Base URL for the middleware API.

    Returns:
        API response as dict.
    """
    base_url = api_base_url or os.getenv(
        "MIDDLEWARE_API_URL",
        "https://sample-agument-middleware-dev.azurewebsites.net",
    )
    endpoint = f"{base_url}/sample-enrichment/api/projects/update-persona-file-url"

    payload = {
        "projectId": project_id,
        "sasUrl": sas_url,
    }

    headers = {
        "Content-Type": "application/json",
    }

    logger.info("Calling API to update persona file URL for project %d", project_id)

    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        logger.info("Successfully updated persona file URL for project %d", project_id)
        return {"status": "success", "response": result}
    except requests.exceptions.RequestException as e:
        logger.exception("Failed to update persona file URL")
        return {"status": "error", "message": str(e)}


def upload_and_update_persona(
    file_path: str,
    project_id: int,
    blob_name: str | None = None,
    container_name: str | None = None,
    expiry_days: int = 365,
    api_base_url: str | None = None,
) -> dict:
    """
    Upload a file to Azure Blob Storage and update the persona file URL via API.

    Args:
        file_path: Path to the local file to upload.
        project_id: The project ID to update.
        blob_name: Name for the blob. If None, generates a unique name.
        container_name: Container to upload to. Uses default if None.
        expiry_days: Number of days until SAS token expires.
        api_base_url: Base URL for the middleware API.

    Returns:
        Dict with upload details and API response.
    """
    try:
        client = AzureBlobStorageClient()

        upload_result = client.upload_and_get_sas_url(
            file_path=file_path,
            blob_name=blob_name,
            container_name=container_name,
            expiry_days=expiry_days,
        )

        api_result = update_persona_file_url(
            project_id=project_id,
            sas_url=upload_result["sas_url"],
            api_base_url=api_base_url,
        )

        return {
            "status": "success",
            "upload": upload_result,
            "api_response": api_result,
        }

    except Exception as e:
        logger.exception("Failed to upload and update persona")
        return {
            "status": "error",
            "message": str(e),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload file to Azure Blob Storage and update persona URL"
    )
    parser.add_argument(
        "file_path",
        help="Path to the file to upload",
    )
    parser.add_argument(
        "project_id",
        type=int,
        help="Project ID to update",
    )
    parser.add_argument(
        "--blob-name",
        help="Custom blob name (optional)",
    )
    parser.add_argument(
        "--container",
        help="Container name (optional, uses AZURE_STORAGE_CONTAINER_NAME or default)",
    )
    parser.add_argument(
        "--expiry-days",
        type=int,
        default=365,
        help="SAS URL expiry in days (default: 365)",
    )
    parser.add_argument(
        "--api-url",
        help="Middleware API base URL (optional)",
    )

    args = parser.parse_args()

    result = upload_and_update_persona(
        file_path=args.file_path,
        project_id=args.project_id,
        blob_name=args.blob_name,
        container_name=args.container,
        expiry_days=args.expiry_days,
        api_base_url=args.api_url,
    )

    import json

    print(json.dumps(result, indent=2))
