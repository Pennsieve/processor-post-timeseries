"""S3 chunk uploader using STS credentials returned by packages-service.

The credentials are short-lived (~1 hour). For very long uploads,
this will need a refresh mechanism (see packages-service POST
/assets/{id}/upload-credentials, not yet implemented).
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import boto3
import botocore.exceptions

from clients.packages_assets_client import UploadCredentials

log = logging.getLogger()


@dataclass
class ChunkUploadResult:
    """Per-file upload outcome — used downstream to register ranges."""

    relative_key: str  # the chunk's path relative to the asset's prefix
    full_key: str  # bucket-relative S3 key (key_prefix + relative_key)


class AssetUploader:
    """Uploads files to a viewer-asset's S3 prefix using STS credentials."""

    def __init__(self, credentials: UploadCredentials, max_workers: int = 4):
        self._creds = credentials
        self._max_workers = max_workers

    def _build_client(self):
        """Create a fresh boto3 S3 client bound to the STS session."""
        return boto3.client(
            "s3",
            aws_access_key_id=self._creds.access_key_id,
            aws_secret_access_key=self._creds.secret_access_key,
            aws_session_token=self._creds.session_token,
            region_name=self._creds.region,
        )

    def upload_files(
        self,
        files: list[tuple[str, str]],
    ) -> list[ChunkUploadResult]:
        """Upload local files in parallel.

        Args:
            files: list of (local_path, relative_key) tuples. The relative
                key is the path UNDER the asset's prefix; this method
                prepends key_prefix when forming the full S3 key.

        Returns:
            list of ChunkUploadResult in input order.

        Raises:
            botocore.exceptions.ClientError on any failure (including
            ExpiredToken). No partial-success aggregation: if any file
            fails, the exception propagates and the orchestrator should
            treat the asset as failed.
        """
        s3 = self._build_client()
        results: list[ChunkUploadResult] = [None] * len(files)  # type: ignore[list-item]

        def _put(index: int, local_path: str, relative_key: str) -> None:
            full_key = self._creds.key_prefix + relative_key
            try:
                s3.upload_file(local_path, self._creds.bucket, full_key)
            except botocore.exceptions.ClientError as e:
                log.error(
                    "failed to upload %s to s3://%s/%s: %s",
                    local_path,
                    self._creds.bucket,
                    full_key,
                    e,
                )
                raise
            results[index] = ChunkUploadResult(relative_key=relative_key, full_key=full_key)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [
                executor.submit(_put, i, local, relative)
                for i, (local, relative) in enumerate(files)
            ]
            # Wait for all and surface the first exception.
            for future in futures:
                future.result()

        log.info(
            "uploaded %d files to s3://%s/%s",
            len(files),
            self._creds.bucket,
            self._creds.key_prefix,
        )
        return results


def relative_key_for_chunk(chunk_filename: str) -> str:
    """Map a chunk file in the local output dir to its S3 relative key.

    Today's writer produces files named:
        channel-{idx:05d}_{start_us}_{end_us}.bin.gz
    or after channel-id substitution in importer:
        {channel_node_id}_{start_us}_{end_us}.bin.gz

    The relative key is just the basename — no nested directories.
    timeseries-service validates this is a "safe relative path".
    """
    return os.path.basename(chunk_filename)
