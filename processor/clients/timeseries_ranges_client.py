import json
import logging
from dataclasses import asdict, dataclass
from typing import Optional

import backoff
import requests

from .base_client import BaseClient, _is_client_error

log = logging.getLogger()


@dataclass
class RangeChunk:
    """One time-series chunk to register.

    s3_key is RELATIVE to the asset's prefix (no leading slash, no '..',
    no 's3://...'). timeseries-service prepends the prefix server-side.
    """

    channel_node_id: str
    start: int
    end: int
    s3_key: str

    def as_dict(self) -> dict:
        # JSON keys must be snake_case to match the timeseries-service DTO.
        return {
            "channel_node_id": self.channel_node_id,
            "start": self.start,
            "end": self.end,
            "s3_key": self.s3_key,
        }


@dataclass
class CreateRangesResult:
    requested: int
    created: int
    skipped: int

    @classmethod
    def from_dict(cls, data: dict) -> "CreateRangesResult":
        return cls(
            requested=data["requested"],
            created=data["created"],
            skipped=data["skipped"],
        )


class TimeSeriesRangesClient(BaseClient):
    """Client for timeseries-service range-registration endpoint.

    Base URL: {api_host2}/timeseries
    """

    # timeseries-service caps request size; chunk client-side if needed.
    # Mirrors dto.MaxChunksPerCreateRangeRequest in timeseries-service.
    MAX_CHUNKS_PER_REQUEST = 1000

    def __init__(self, api_host2, session_manager):
        super().__init__(session_manager)
        self.base_url = f"{api_host2}/timeseries"

    def _auth_headers(self) -> dict:
        return {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.session_manager.session_token}",
        }

    @backoff.on_exception(
        backoff.expo,
        requests.HTTPError,
        max_tries=3,
        giveup=_is_client_error,
    )
    @BaseClient.retry_with_refresh
    def create_ranges(
        self,
        package_node_id: str,
        viewer_asset_id: str,
        chunks: list[RangeChunk],
    ) -> CreateRangesResult:
        """POST a single batch of range chunks. Caller is responsible for
        splitting > MAX_CHUNKS_PER_REQUEST into multiple calls via
        create_ranges_batched.
        """
        if len(chunks) == 0:
            return CreateRangesResult(requested=0, created=0, skipped=0)
        if len(chunks) > self.MAX_CHUNKS_PER_REQUEST:
            raise ValueError(
                f"too many chunks ({len(chunks)}); max {self.MAX_CHUNKS_PER_REQUEST}"
            )

        url = f"{self.base_url}/package/{package_node_id}/ranges"
        body = {
            "viewer_asset_id": viewer_asset_id,
            "chunks": [c.as_dict() for c in chunks],
        }

        try:
            response = requests.post(url, headers=self._auth_headers(), json=body)
            response.raise_for_status()
            return CreateRangesResult.from_dict(response.json())
        except requests.HTTPError as e:
            # 400 with InvalidChunks payload is the most useful failure to surface
            log.error(
                "failed to register ranges for package %s asset %s: %s",
                package_node_id,
                viewer_asset_id,
                _format_error_response(e.response),
            )
            raise

    def create_ranges_batched(
        self,
        package_node_id: str,
        viewer_asset_id: str,
        chunks: list[RangeChunk],
    ) -> CreateRangesResult:
        """Split chunks into MAX_CHUNKS_PER_REQUEST batches and POST each.

        Returns aggregated counts. Stops on first failure (no rollback —
        each successful batch's ranges remain inserted).
        """
        total = CreateRangesResult(requested=0, created=0, skipped=0)
        for i in range(0, len(chunks), self.MAX_CHUNKS_PER_REQUEST):
            batch = chunks[i : i + self.MAX_CHUNKS_PER_REQUEST]
            result = self.create_ranges(package_node_id, viewer_asset_id, batch)
            total.requested += result.requested
            total.created += result.created
            total.skipped += result.skipped
        return total


def _format_error_response(response: Optional[requests.Response]) -> str:
    if response is None:
        return "<no response>"
    try:
        body = response.json()
        return json.dumps(body)
    except (ValueError, json.JSONDecodeError):
        return response.text
