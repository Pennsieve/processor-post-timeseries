"""Integration-style tests for import_timeseries_via_assets.

Exercises the full orchestration end-to-end with all four external
services mocked: workflows, packages-service (regular + assets),
pennsieve-api timeseries channels, timeseries-service ranges, and S3.
"""
import gzip
import json
import os
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit

import pytest
import responses
from clients.base_client import SessionManager
from importer import import_timeseries_via_assets


def _path_only(url: str) -> str:
    """Strip query string for path-based assertions."""
    return urlsplit(url).path


# ---- canonical responses --------------------------------------------------


WORKFLOW_INSTANCE_ID = "wf-123"
DATASET_NODE_ID = "N:dataset:abc"
PACKAGE_NODE_ID = "N:package:p1"
PARENT_NODE_ID = "N:collection:parent"
CHANNEL_NODE_ID = "N:channel:ch1"
ASSET_ID = "00000000-0000-0000-0000-000000000001"

API_HOST = "https://api.test"
API_HOST2 = "https://api2.test"


def _workflow_response():
    return {
        "uuid": WORKFLOW_INSTANCE_ID,
        "datasetId": DATASET_NODE_ID,
        "dataSources": {"src": {"packageIds": [PACKAGE_NODE_ID]}},
    }


def _channel_create_response():
    return {
        "content": {
            "id": CHANNEL_NODE_ID,
            "name": "ch-0",
            "start": 0,
            "end": 1000,
            "unit": "uV",
            "rate": 1000.0,
            "channelType": "CONTINUOUS",
            "group": "default",
            "lastAnnotation": 0,
        },
        "properties": [],
    }


def _asset_create_response(status="created"):
    return {
        "asset": {
            "id": ASSET_ID,
            "dataset_id": DATASET_NODE_ID,
            "name": "mef-asset",
            "asset_type": "timeseries",
            "asset_url": "",
            "properties": {},
            "status": status,
            "package_ids": [PACKAGE_NODE_ID],
            "created_at": "2026-04-30T12:00:00Z",
        },
        "upload_credentials": {
            "access_key_id": "AKIA",
            "secret_access_key": "SECRET",
            "session_token": "TOKEN",
            "expiration": "2026-04-30T13:00:00Z",
            "bucket": "pennsieve-viewer-assets",
            "region": "us-east-1",
            "key_prefix": f"viewer-assets/O19/D2049/{ASSET_ID}/",
        },
    }


# ---- fixtures -------------------------------------------------------------


@pytest.fixture
def session_manager():
    auth = MagicMock()
    auth.get_session_token.return_value = "test-token"
    return SessionManager(auth)


@pytest.fixture
def staged_files(tmp_path):
    """Create a single channel's staged files (.bin.gz + .metadata.json)."""
    output = tmp_path / "output"
    output.mkdir()

    # Channel metadata file (matches what writer.py emits)
    meta = {
        "name": "ch-0",
        "start": 0,
        "end": 1000,
        "unit": "uV",
        "rate": 1000.0,
        "type": "CONTINUOUS",
        "group": "default",
        "lastAnnotation": 0,
        "properties": [],
    }
    (output / "channel-00000.metadata.json").write_text(json.dumps(meta))

    # Chunk binary file (named with channel-{idx} prefix; importer renames it)
    chunk = output / "channel-00000_0_1000.bin.gz"
    with gzip.open(chunk, "wb") as f:
        f.write(b"\x00" * 8)
    return str(output)


@pytest.fixture
def all_responses():
    """Wires up the canonical happy-path response set across all services."""
    rsps = responses.RequestsMock()
    rsps.start()
    yield rsps
    rsps.stop()
    rsps.reset()


def _wire_happy_path(rsps, *, list_assets_returns=None, asset_status_after_patch="active"):
    """Register all happy-path responses against the given RequestsMock."""
    # WorkflowClient
    rsps.add(
        responses.GET,
        f"{API_HOST2}/compute/workflows/runs/{WORKFLOW_INSTANCE_ID}",
        json=_workflow_response(),
        status=200,
    )

    # PackagesClient.set_timeseries_properties → PUT /packages/{id}?updateStorage=true
    rsps.add(
        responses.PUT,
        f"{API_HOST}/packages/{PACKAGE_NODE_ID}",
        json={},
        status=200,
    )

    # PackagesAssetsClient.list_assets_for_package → GET /packages/assets
    rsps.add(
        responses.GET,
        f"{API_HOST2}/packages/assets",
        json={"assets": list_assets_returns or []},
        status=200,
    )

    # PackagesAssetsClient.create_asset → POST /packages/assets
    rsps.add(
        responses.POST,
        f"{API_HOST2}/packages/assets",
        json=_asset_create_response(),
        status=201,
    )

    # TimeSeriesClient.get_package_channels → GET /timeseries/{pkg}/channels
    rsps.add(
        responses.GET,
        f"{API_HOST}/timeseries/{PACKAGE_NODE_ID}/channels",
        json=[],
        status=200,
    )

    # TimeSeriesClient.create_channel → POST /timeseries/{pkg}/channels
    rsps.add(
        responses.POST,
        f"{API_HOST}/timeseries/{PACKAGE_NODE_ID}/channels",
        json=_channel_create_response(),
        status=201,
    )

    # TimeSeriesRangesClient.create_ranges → POST /timeseries/package/{pkg}/ranges
    rsps.add(
        responses.POST,
        f"{API_HOST2}/timeseries/package/{PACKAGE_NODE_ID}/ranges",
        json={"requested": 1, "created": 1, "skipped": 0},
        status=201,
    )

    # PackagesAssetsClient.update_asset → PATCH /packages/assets/{id}
    rsps.add(
        responses.PATCH,
        f"{API_HOST2}/packages/assets/{ASSET_ID}",
        json={
            "id": ASSET_ID,
            "dataset_id": DATASET_NODE_ID,
            "name": "mef-asset",
            "asset_type": "timeseries",
            "status": asset_status_after_patch,
            "package_ids": [PACKAGE_NODE_ID],
            "created_at": "2026-04-30T12:00:00Z",
        },
        status=200,
    )


# ---- tests ----------------------------------------------------------------


class TestHappyPath:
    @patch("asset_uploader.boto3.client")
    def test_full_flow(self, mock_boto, session_manager, staged_files, all_responses):
        s3 = MagicMock()
        mock_boto.return_value = s3
        _wire_happy_path(all_responses)

        result = import_timeseries_via_assets(
            api_host=API_HOST,
            api2_host=API_HOST2,
            session_manager=session_manager,
            workflow_instance_id=WORKFLOW_INSTANCE_ID,
            file_directory=staged_files,
            asset_name="mef-asset",
            asset_type="timeseries",
        )

        assert result == ASSET_ID
        # S3 upload happened with the asset's prefix
        s3.upload_file.assert_called_once()
        call = s3.upload_file.call_args
        # boto3 upload_file(local_path, bucket, key)
        local_path, bucket, key = call.args
        assert bucket == "pennsieve-viewer-assets"
        assert key.startswith(f"viewer-assets/O19/D2049/{ASSET_ID}/")
        # Local file was renamed to use the channel node id
        assert os.path.basename(local_path).startswith(CHANNEL_NODE_ID)


class TestIdempotentSkip:
    @patch("asset_uploader.boto3.client")
    def test_active_asset_short_circuits(
        self, mock_boto, session_manager, staged_files
    ):
        """When list_assets returns an existing 'active' asset, the function
        returns its id without uploading or registering ranges."""
        rsps = responses.RequestsMock()
        rsps.start()
        try:
            # Workflow + properties update still run
            rsps.add(
                responses.GET,
                f"{API_HOST2}/compute/workflows/runs/{WORKFLOW_INSTANCE_ID}",
                json=_workflow_response(),
                status=200,
            )
            rsps.add(
                responses.PUT,
                f"{API_HOST}/packages/{PACKAGE_NODE_ID}",
                json={},
                status=200,
            )

            # List assets returns one already-active asset matching name+type
            rsps.add(
                responses.GET,
                f"{API_HOST2}/packages/assets",
                json={
                    "assets": [
                        {
                            "id": ASSET_ID,
                            "dataset_id": DATASET_NODE_ID,
                            "name": "mef-asset",
                            "asset_type": "timeseries",
                            "asset_url": "",
                            "properties": {},
                            "status": "active",
                            "package_ids": [PACKAGE_NODE_ID],
                            "created_at": "2026-04-30T12:00:00Z",
                        }
                    ]
                },
                status=200,
            )

            result = import_timeseries_via_assets(
                api_host=API_HOST,
                api2_host=API_HOST2,
                session_manager=session_manager,
                workflow_instance_id=WORKFLOW_INSTANCE_ID,
                file_directory=staged_files,
                asset_name="mef-asset",
                asset_type="timeseries",
            )

            assert result == ASSET_ID
            # No POST /assets, no upload, no ranges call
            mock_boto.assert_not_called()
        finally:
            rsps.stop()
            rsps.reset()


class TestStaleAssetReplaced:
    @patch("asset_uploader.boto3.client")
    def test_non_active_asset_is_deleted_and_recreated(
        self, mock_boto, session_manager, staged_files
    ):
        """Existing asset in any non-active status → deleted and recreated."""
        s3 = MagicMock()
        mock_boto.return_value = s3
        rsps = responses.RequestsMock()
        rsps.start()
        try:
            stale_asset_id = "stale-uuid"

            _wire_happy_path(
                rsps,
                list_assets_returns=[
                    {
                        "id": stale_asset_id,
                        "dataset_id": DATASET_NODE_ID,
                        "name": "mef-asset",
                        "asset_type": "timeseries",
                        "asset_url": "",
                        "properties": {},
                        "status": "created",  # ← prior failed run
                        "package_ids": [PACKAGE_NODE_ID],
                        "created_at": "2026-04-30T11:00:00Z",
                    }
                ],
            )
            # DELETE on the stale asset
            rsps.add(
                responses.DELETE,
                f"{API_HOST2}/packages/assets/{stale_asset_id}",
                status=204,
            )

            result = import_timeseries_via_assets(
                api_host=API_HOST,
                api2_host=API_HOST2,
                session_manager=session_manager,
                workflow_instance_id=WORKFLOW_INSTANCE_ID,
                file_directory=staged_files,
                asset_name="mef-asset",
                asset_type="timeseries",
            )

            assert result == ASSET_ID  # the freshly created one
            # Confirm DELETE for stale id and POST for new one both happened
            assert any(
                c.request.method == "DELETE" and stale_asset_id in c.request.url
                for c in rsps.calls
            )
            assert any(
                c.request.method == "POST"
                and _path_only(c.request.url) == "/packages/assets"
                for c in rsps.calls
            )
        finally:
            rsps.stop()
            rsps.reset()


class TestFailureCleanup:
    @patch("asset_uploader.boto3.client")
    def test_upload_failure_triggers_asset_delete(
        self, mock_boto, session_manager, staged_files
    ):
        """If upload fails after asset creation, the asset is DELETEd to
        enqueue S3 cleanup, and the original error propagates."""
        import botocore.exceptions

        s3 = MagicMock()
        s3.upload_file.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "ExpiredToken", "Message": "expired"}},
            "PutObject",
        )
        mock_boto.return_value = s3

        # Upload fails before ranges/patch responses get hit; allow the
        # happy-path setup to register responses that don't fire.
        rsps = responses.RequestsMock(assert_all_requests_are_fired=False)
        rsps.start()
        try:
            _wire_happy_path(rsps)
            # DELETE for cleanup after upload fails
            rsps.add(
                responses.DELETE,
                f"{API_HOST2}/packages/assets/{ASSET_ID}",
                status=204,
            )

            with pytest.raises(botocore.exceptions.ClientError):
                import_timeseries_via_assets(
                    api_host=API_HOST,
                    api2_host=API_HOST2,
                    session_manager=session_manager,
                    workflow_instance_id=WORKFLOW_INSTANCE_ID,
                    file_directory=staged_files,
                    asset_name="mef-asset",
                    asset_type="timeseries",
                )

            # Verify the cleanup DELETE actually fired
            delete_calls = [
                c for c in rsps.calls
                if c.request.method == "DELETE" and ASSET_ID in c.request.url
            ]
            assert len(delete_calls) == 1
        finally:
            rsps.stop()
            rsps.reset()


class TestMultiPackageIdempotency:
    """Re-run on a multi-package workflow finds the existing asset by
    iterating the workflow packages — not by the parent collection
    that determine_target_package walks up to.

    This is the bug the reviewer caught: lookup-by-parent vs.
    create-with-children produces orphan duplicate assets on re-run.
    """

    @patch("asset_uploader.boto3.client")
    def test_active_asset_found_via_second_child_package(
        self, mock_boto, session_manager, staged_files
    ):
        # Workflow has 3 children; the existing asset is linked to all of
        # them, but list_assets_for_package only returns it for ones
        # actually in viewer_asset_packages.
        child_a = "N:package:mef-a"
        child_b = "N:package:mef-b"
        child_c = "N:package:mef-c"
        parent = "N:collection:recording-parent"

        rsps = responses.RequestsMock()
        rsps.start()
        try:
            # Workflow returns the children, not the parent
            rsps.add(
                responses.GET,
                f"{API_HOST2}/compute/workflows/runs/{WORKFLOW_INSTANCE_ID}",
                json={
                    "uuid": WORKFLOW_INSTANCE_ID,
                    "datasetId": DATASET_NODE_ID,
                    "dataSources": {
                        "src": {"packageIds": [child_a, child_b, child_c]}
                    },
                },
                status=200,
            )
            # determine_target_package walks first child to its parent
            rsps.add(
                responses.GET,
                f"{API_HOST}/packages/{child_a}",
                json={
                    "parent": {"content": {"nodeId": parent}},
                    "content": {"nodeId": child_a},
                },
                status=200,
            )
            # Properties are set on the parent (legacy aggregation)
            rsps.add(
                responses.PUT,
                f"{API_HOST}/packages/{parent}",
                json={},
                status=200,
            )
            # Lookup by first child returns empty (this is the bug-trap:
            # if we'd looked up by parent we'd also get empty here, then
            # create a duplicate asset).
            rsps.add(
                responses.GET,
                f"{API_HOST2}/packages/assets",
                match=[responses.matchers.query_param_matcher(
                    {"dataset_id": DATASET_NODE_ID, "package_id": child_a}
                )],
                json={"assets": []},
                status=200,
            )
            # Lookup by second child returns the active asset
            rsps.add(
                responses.GET,
                f"{API_HOST2}/packages/assets",
                match=[responses.matchers.query_param_matcher(
                    {"dataset_id": DATASET_NODE_ID, "package_id": child_b}
                )],
                json={
                    "assets": [
                        {
                            "id": ASSET_ID,
                            "dataset_id": DATASET_NODE_ID,
                            "name": "mef-asset",
                            "asset_type": "timeseries",
                            "asset_url": "",
                            "properties": {},
                            "status": "active",
                            "package_ids": [child_a, child_b, child_c],
                            "created_at": "2026-04-30T12:00:00Z",
                        }
                    ]
                },
                status=200,
            )

            result = import_timeseries_via_assets(
                api_host=API_HOST,
                api2_host=API_HOST2,
                session_manager=session_manager,
                workflow_instance_id=WORKFLOW_INSTANCE_ID,
                file_directory=staged_files,
                asset_name="mef-asset",
                asset_type="timeseries",
            )

            assert result == ASSET_ID
            # No upload, no POST /assets — purely idempotent skip
            mock_boto.assert_not_called()
            posts = [c for c in rsps.calls if c.request.method == "POST"]
            assert posts == []
            # Iteration stopped after second child — third was never queried
            assets_lookups = [
                c for c in rsps.calls
                if c.request.method == "GET"
                and "/packages/assets" in c.request.url
            ]
            assert len(assets_lookups) == 2
        finally:
            rsps.stop()
            rsps.reset()


class TestEmptyDirectory:
    def test_no_files_returns_none_without_calling_services(
        self, session_manager, tmp_path
    ):
        empty = tmp_path / "empty"
        empty.mkdir()

        # No responses registered — if any HTTP call fires, responses will raise
        with responses.RequestsMock() as rsps:
            result = import_timeseries_via_assets(
                api_host=API_HOST,
                api2_host=API_HOST2,
                session_manager=session_manager,
                workflow_instance_id=WORKFLOW_INSTANCE_ID,
                file_directory=str(empty),
                asset_name="mef-asset",
                asset_type="timeseries",
            )
            assert result is None
            assert len(rsps.calls) == 0
