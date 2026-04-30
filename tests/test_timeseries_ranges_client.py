import json

import pytest
import responses
from clients.timeseries_ranges_client import (
    CreateRangesResult,
    RangeChunk,
    TimeSeriesRangesClient,
)


def _ok_response(requested=2, created=2, skipped=0):
    return {"requested": requested, "created": created, "skipped": skipped}


class TestTimeSeriesRangesClientInit:
    def test_initialization_appends_timeseries_path(self, mock_session_manager):
        client = TimeSeriesRangesClient("https://api2.test.com", mock_session_manager)
        assert client.base_url == "https://api2.test.com/timeseries"


class TestTimeSeriesRangesClientCreate:
    @responses.activate
    def test_create_ranges_success(self, mock_session_manager):
        responses.add(
            responses.POST,
            "https://api2.test.com/timeseries/package/N:package:p1/ranges",
            json=_ok_response(requested=2, created=2, skipped=0),
            status=201,
        )

        client = TimeSeriesRangesClient("https://api2.test.com", mock_session_manager)
        chunks = [
            RangeChunk("N:channel:c1", 0, 1000, "N:channel:c1_0_1000.bin.gz"),
            RangeChunk("N:channel:c2", 0, 1000, "N:channel:c2_0_1000.bin.gz"),
        ]

        result = client.create_ranges("N:package:p1", "asset-uuid", chunks)
        assert isinstance(result, CreateRangesResult)
        assert result.requested == 2
        assert result.created == 2
        assert result.skipped == 0

    @responses.activate
    def test_create_ranges_sends_snake_case_body(self, mock_session_manager):
        responses.add(
            responses.POST,
            "https://api2.test.com/timeseries/package/N:package:p1/ranges",
            json=_ok_response(requested=1, created=1),
            status=201,
        )

        client = TimeSeriesRangesClient("https://api2.test.com", mock_session_manager)
        chunks = [RangeChunk("N:channel:c1", 0, 1000, "rel/key.bin.gz")]
        client.create_ranges("N:package:p1", "asset-uuid", chunks)

        body = json.loads(responses.calls[0].request.body)
        assert body["viewer_asset_id"] == "asset-uuid"
        assert len(body["chunks"]) == 1
        assert body["chunks"][0] == {
            "channel_node_id": "N:channel:c1",
            "start": 0,
            "end": 1000,
            "s3_key": "rel/key.bin.gz",
        }

    def test_create_ranges_short_circuits_on_empty(self, mock_session_manager):
        client = TimeSeriesRangesClient("https://api2.test.com", mock_session_manager)
        result = client.create_ranges("N:package:p1", "asset-uuid", [])
        assert result == CreateRangesResult(requested=0, created=0, skipped=0)

    def test_create_ranges_rejects_oversized_batch(self, mock_session_manager):
        client = TimeSeriesRangesClient("https://api2.test.com", mock_session_manager)
        oversize = [
            RangeChunk("N:channel:c", i, i + 1, f"k_{i}.bin.gz") for i in range(client.MAX_CHUNKS_PER_REQUEST + 1)
        ]
        with pytest.raises(ValueError, match="too many chunks"):
            client.create_ranges("N:package:p1", "asset-uuid", oversize)

    @responses.activate
    def test_create_ranges_raises_on_4xx_with_invalid_chunks(self, mock_session_manager):
        responses.add(
            responses.POST,
            "https://api2.test.com/timeseries/package/N:package:p1/ranges",
            json={
                "message": "one or more chunks are invalid",
                "invalid_chunks": [{"index": 0, "reason": "bad start/end"}],
            },
            status=400,
        )

        client = TimeSeriesRangesClient("https://api2.test.com", mock_session_manager)
        chunks = [RangeChunk("N:channel:c1", 100, 0, "rel/key.bin.gz")]

        with pytest.raises(Exception):
            client.create_ranges("N:package:p1", "asset-uuid", chunks)


class TestTimeSeriesRangesClientBatched:
    @responses.activate
    def test_batched_splits_into_max_chunks(self, mock_session_manager):
        # Send more chunks than the server allows in one call. The client
        # should split them up and make multiple trips to the server.
        max_n = TimeSeriesRangesClient.MAX_CHUNKS_PER_REQUEST
        for size in (max_n, max_n, 500):
            responses.add(
                responses.POST,
                "https://api2.test.com/timeseries/package/N:package:p1/ranges",
                json=_ok_response(requested=size, created=size, skipped=0),
                status=201,
            )

        client = TimeSeriesRangesClient("https://api2.test.com", mock_session_manager)
        chunks = [RangeChunk("N:channel:c", i, i + 1, f"k_{i}.bin.gz") for i in range(2 * max_n + 500)]

        result = client.create_ranges_batched("N:package:p1", "asset-uuid", chunks)
        assert result.requested == 2 * max_n + 500
        assert result.created == 2 * max_n + 500
        assert len(responses.calls) == 3

    @responses.activate
    def test_batched_aborts_on_first_failure(self, mock_session_manager):
        # First batch fails — second batch should not be sent.
        responses.add(
            responses.POST,
            "https://api2.test.com/timeseries/package/N:package:p1/ranges",
            json={"error": "bad chunk"},
            status=400,
        )

        client = TimeSeriesRangesClient("https://api2.test.com", mock_session_manager)
        chunks = [
            RangeChunk("N:channel:c", i, i + 1, f"k_{i}.bin.gz") for i in range(client.MAX_CHUNKS_PER_REQUEST + 1)
        ]

        with pytest.raises(Exception):
            client.create_ranges_batched("N:package:p1", "asset-uuid", chunks)

        assert len(responses.calls) == 1
