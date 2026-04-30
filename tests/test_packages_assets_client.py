import json
from unittest.mock import patch

import pytest
import responses
from clients.packages_assets_client import (
    CreatedAsset,
    PackagesAssetsClient,
    UploadCredentials,
    ViewerAsset,
)


def _create_response_body():
    """A canonical POST /assets response body."""
    return {
        "asset": {
            "id": "00000000-0000-0000-0000-000000000001",
            "dataset_id": "N:dataset:abc",
            "name": "mef-asset",
            "asset_type": "timeseries",
            "asset_url": "",
            "properties": {},
            "status": "created",
            "package_ids": ["N:package:p1"],
            "created_at": "2026-04-20T12:00:00Z",
        },
        "upload_credentials": {
            "access_key_id": "AKIATEST",
            "secret_access_key": "SECRET",
            "session_token": "TOKEN",
            "expiration": "2026-04-20T13:00:00Z",
            "bucket": "pennsieve-viewer-assets",
            "region": "us-east-1",
            "key_prefix": "viewer-assets/O19/D2049/00000000-0000-0000-0000-000000000001/",
        },
    }


class TestPackagesAssetsClientInit:
    def test_initialization_appends_packages_path(self, mock_session_manager):
        client = PackagesAssetsClient("https://api2.test.com", mock_session_manager)
        assert client.base_url == "https://api2.test.com/packages"


class TestPackagesAssetsClientCreate:
    @responses.activate
    def test_create_asset_success(self, mock_session_manager):
        responses.add(
            responses.POST,
            "https://api2.test.com/packages/assets",
            json=_create_response_body(),
            status=201,
        )

        client = PackagesAssetsClient("https://api2.test.com", mock_session_manager)
        result = client.create_asset(
            dataset_id="N:dataset:abc",
            package_ids=["N:package:p1"],
            name="mef-asset",
            asset_type="timeseries",
        )

        assert isinstance(result, CreatedAsset)
        assert result.asset.id == "00000000-0000-0000-0000-000000000001"
        assert result.asset.name == "mef-asset"
        assert result.upload_credentials.bucket == "pennsieve-viewer-assets"
        assert result.upload_credentials.key_prefix.startswith("viewer-assets/")

    @responses.activate
    def test_create_asset_sends_dataset_id_as_query_param(self, mock_session_manager):
        responses.add(
            responses.POST,
            "https://api2.test.com/packages/assets",
            json=_create_response_body(),
            status=201,
        )

        client = PackagesAssetsClient("https://api2.test.com", mock_session_manager)
        client.create_asset(
            dataset_id="N:dataset:abc",
            package_ids=["N:package:p1"],
            name="mef-asset",
            asset_type="timeseries",
        )

        request = responses.calls[0].request
        assert "dataset_id=N%3Adataset%3Aabc" in request.url
        body = json.loads(request.body)
        assert body == {
            "name": "mef-asset",
            "asset_type": "timeseries",
            "package_ids": ["N:package:p1"],
        }

    @responses.activate
    def test_create_asset_includes_properties_when_provided(self, mock_session_manager):
        responses.add(
            responses.POST,
            "https://api2.test.com/packages/assets",
            json=_create_response_body(),
            status=201,
        )

        client = PackagesAssetsClient("https://api2.test.com", mock_session_manager)
        client.create_asset(
            dataset_id="N:dataset:abc",
            package_ids=["N:package:p1"],
            name="mef-asset",
            asset_type="timeseries",
            properties={"source": "mef"},
        )

        body = json.loads(responses.calls[0].request.body)
        assert body["properties"] == {"source": "mef"}

    @responses.activate
    def test_create_asset_raises_on_4xx(self, mock_session_manager):
        responses.add(
            responses.POST,
            "https://api2.test.com/packages/assets",
            json={"error": "bad request"},
            status=400,
        )

        client = PackagesAssetsClient("https://api2.test.com", mock_session_manager)
        with pytest.raises(Exception):
            client.create_asset(
                dataset_id="N:dataset:abc",
                package_ids=["N:package:p1"],
                name="mef-asset",
                asset_type="timeseries",
            )


class TestPackagesAssetsClientList:
    @responses.activate
    def test_list_assets_success(self, mock_session_manager):
        responses.add(
            responses.GET,
            "https://api2.test.com/packages/assets",
            json={
                "assets": [
                    {
                        "id": "uuid-1",
                        "dataset_id": "N:dataset:abc",
                        "name": "mef-asset",
                        "asset_type": "timeseries",
                        "asset_url": "",
                        "properties": {},
                        "status": "active",
                        "package_ids": ["N:package:p1"],
                        "created_at": "2026-04-20T12:00:00Z",
                    }
                ]
            },
            status=200,
        )

        client = PackagesAssetsClient("https://api2.test.com", mock_session_manager)
        result = client.list_assets_for_package(
            dataset_id="N:dataset:abc", package_id="N:package:p1"
        )

        assert len(result) == 1
        assert isinstance(result[0], ViewerAsset)
        assert result[0].id == "uuid-1"
        assert result[0].status == "active"

    @responses.activate
    def test_list_assets_returns_empty_when_no_assets(self, mock_session_manager):
        responses.add(
            responses.GET,
            "https://api2.test.com/packages/assets",
            json={"assets": []},
            status=200,
        )

        client = PackagesAssetsClient("https://api2.test.com", mock_session_manager)
        assert client.list_assets_for_package("N:dataset:abc", "N:package:p1") == []


class TestPackagesAssetsClientUpdate:
    @responses.activate
    def test_update_asset_status_only(self, mock_session_manager):
        responses.add(
            responses.PATCH,
            "https://api2.test.com/packages/assets/uuid-1",
            json={
                "id": "uuid-1",
                "dataset_id": "N:dataset:abc",
                "name": "mef-asset",
                "asset_type": "timeseries",
                "status": "active",
                "package_ids": ["N:package:p1"],
                "created_at": "2026-04-20T12:00:00Z",
            },
            status=200,
        )

        client = PackagesAssetsClient("https://api2.test.com", mock_session_manager)
        result = client.update_asset(
            asset_id="uuid-1", dataset_id="N:dataset:abc", status="active"
        )

        assert result.status == "active"
        body = json.loads(responses.calls[0].request.body)
        assert body == {"status": "active"}


class TestPackagesAssetsClientDelete:
    @responses.activate
    def test_delete_asset_returns_none(self, mock_session_manager):
        responses.add(
            responses.DELETE,
            "https://api2.test.com/packages/assets/uuid-1",
            status=204,
        )

        client = PackagesAssetsClient("https://api2.test.com", mock_session_manager)
        assert client.delete_asset(asset_id="uuid-1", dataset_id="N:dataset:abc") is None
        assert "dataset_id=N%3Adataset%3Aabc" in responses.calls[0].request.url


class TestPackagesAssetsClientRetries:
    @patch("time.sleep")  # skip the backoff delays
    @responses.activate
    def test_retries_on_5xx_then_succeeds(self, _sleep, mock_session_manager):
        # Two transient failures, then success.
        responses.add(
            responses.GET,
            "https://api2.test.com/packages/assets",
            json={"error": "transient"},
            status=503,
        )
        responses.add(
            responses.GET,
            "https://api2.test.com/packages/assets",
            json={"error": "transient"},
            status=503,
        )
        responses.add(
            responses.GET,
            "https://api2.test.com/packages/assets",
            json={"assets": []},
            status=200,
        )

        client = PackagesAssetsClient("https://api2.test.com", mock_session_manager)
        result = client.list_assets_for_package("N:dataset:abc", "N:package:p1")
        assert result == []
        assert len(responses.calls) == 3

    @responses.activate
    def test_does_not_retry_on_4xx(self, mock_session_manager):
        responses.add(
            responses.GET,
            "https://api2.test.com/packages/assets",
            json={"error": "not found"},
            status=404,
        )

        client = PackagesAssetsClient("https://api2.test.com", mock_session_manager)
        with pytest.raises(Exception):
            client.list_assets_for_package("N:dataset:abc", "N:package:p1")

        # only 1 call — backoff gave up on 4xx (note: 401/403 path would be
        # one extra attempt via retry_with_refresh, but plain 404 should not retry)
        assert len(responses.calls) == 1
