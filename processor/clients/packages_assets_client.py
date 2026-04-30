import json
import logging
from dataclasses import dataclass
from typing import Optional

import backoff
import requests

from .base_client import BaseClient, DEFAULT_TIMEOUT, _is_client_error

log = logging.getLogger()


@dataclass
class UploadCredentials:
    """Short-lived STS credentials returned alongside a created asset.

    Scoped via session policy to PUT under bucket/key_prefix only.
    """

    access_key_id: str
    secret_access_key: str
    session_token: str
    expiration: str
    bucket: str
    region: str
    key_prefix: str

    @classmethod
    def from_dict(cls, data: dict) -> "UploadCredentials":
        return cls(
            access_key_id=data["access_key_id"],
            secret_access_key=data["secret_access_key"],
            session_token=data["session_token"],
            expiration=data["expiration"],
            bucket=data["bucket"],
            region=data["region"],
            key_prefix=data["key_prefix"],
        )


@dataclass
class ViewerAsset:
    id: str
    dataset_id: str
    name: str
    asset_type: str
    status: str
    package_ids: list[str]

    @classmethod
    def from_dict(cls, data: dict) -> "ViewerAsset":
        return cls(
            id=data["id"],
            dataset_id=data["dataset_id"],
            name=data["name"],
            asset_type=data["asset_type"],
            status=data["status"],
            package_ids=data.get("package_ids", []),
        )


@dataclass
class CreatedAsset:
    """Result of POST /assets: an asset row plus STS upload credentials."""

    asset: ViewerAsset
    upload_credentials: UploadCredentials


class PackagesAssetsClient(BaseClient):
    """Client for packages-service viewer-asset endpoints.

    Base URL: {api_host2}/packages
    """

    def __init__(self, api_host2, session_manager):
        super().__init__(session_manager)
        self.base_url = f"{api_host2}/packages"

    def _auth_headers(self) -> dict:
        return {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.session_manager.session_token}",
        }

    @backoff.on_exception(
        backoff.expo,
        requests.RequestException,
        max_tries=3,
        giveup=_is_client_error,
    )
    @BaseClient.retry_with_refresh
    def create_asset(
        self,
        dataset_id: str,
        package_ids: list[str],
        name: str,
        asset_type: str,
        properties: Optional[dict] = None,
    ) -> CreatedAsset:
        """Create a viewer_asset, link the given packages, and return STS upload creds."""
        url = f"{self.base_url}/assets"
        params = {"dataset_id": dataset_id}
        body: dict = {
            "name": name,
            "asset_type": asset_type,
            "package_ids": package_ids,
        }
        if properties is not None:
            body["properties"] = properties

        try:
            response = requests.post(
                url,
                params=params,
                headers=self._auth_headers(),
                json=body,
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            return CreatedAsset(
                asset=ViewerAsset.from_dict(data["asset"]),
                upload_credentials=UploadCredentials.from_dict(data["upload_credentials"]),
            )
        except requests.HTTPError as e:
            log.error("failed to create viewer asset: %s", e)
            raise
        except (KeyError, json.JSONDecodeError) as e:
            log.error("malformed create-asset response: %s", e)
            raise

    @backoff.on_exception(
        backoff.expo,
        requests.RequestException,
        max_tries=3,
        giveup=_is_client_error,
    )
    @BaseClient.retry_with_refresh
    def list_assets_for_package(self, dataset_id: str, package_id: str) -> list[ViewerAsset]:
        """List viewer_assets attached to a package. Used for idempotent re-runs."""
        url = f"{self.base_url}/assets"
        params = {"dataset_id": dataset_id, "package_id": package_id}

        try:
            response = requests.get(
                url,
                params=params,
                headers=self._auth_headers(),
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            return [ViewerAsset.from_dict(a) for a in data.get("assets", [])]
        except requests.HTTPError as e:
            log.error("failed to list viewer assets for package %s: %s", package_id, e)
            raise

    @backoff.on_exception(
        backoff.expo,
        requests.RequestException,
        max_tries=3,
        giveup=_is_client_error,
    )
    @BaseClient.retry_with_refresh
    def update_asset(
        self,
        asset_id: str,
        dataset_id: str,
        status: Optional[str] = None,
        properties: Optional[dict] = None,
        package_ids: Optional[list[str]] = None,
    ) -> ViewerAsset:
        """Patch an asset. Common case: flip status to 'active' once ingest succeeds."""
        url = f"{self.base_url}/assets/{asset_id}"
        params = {"dataset_id": dataset_id}
        body: dict = {}
        if status is not None:
            body["status"] = status
        if properties is not None:
            body["properties"] = properties
        if package_ids is not None:
            body["package_ids"] = package_ids

        try:
            response = requests.patch(
                url,
                params=params,
                headers=self._auth_headers(),
                json=body,
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            return ViewerAsset.from_dict(response.json())
        except requests.HTTPError as e:
            log.error("failed to update viewer asset %s: %s", asset_id, e)
            raise

    @backoff.on_exception(
        backoff.expo,
        requests.RequestException,
        max_tries=3,
        giveup=_is_client_error,
    )
    @BaseClient.retry_with_refresh
    def delete_asset(self, asset_id: str, dataset_id: str) -> None:
        """Delete an asset. Triggers async S3 cleanup via the cleanup-queue lambda."""
        url = f"{self.base_url}/assets/{asset_id}"
        params = {"dataset_id": dataset_id}

        try:
            response = requests.delete(
                url,
                params=params,
                headers=self._auth_headers(),
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
        except requests.HTTPError as e:
            log.error("failed to delete viewer asset %s: %s", asset_id, e)
            raise
