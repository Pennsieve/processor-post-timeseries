import pytest
import json
import responses
from unittest.mock import Mock

from clients.timeseries_client import TimeSeriesClient
from timeseries_channel import TimeSeriesChannel


class TestTimeSeriesClientInit:
    """Tests for TimeSeriesClient initialization."""

    def test_initialization(self, mock_session_manager):
        """Test basic initialization."""
        client = TimeSeriesClient("https://api.test.com", mock_session_manager)

        assert client.api_host == "https://api.test.com"
        assert client.session_manager == mock_session_manager


class TestTimeSeriesClientCreateChannel:
    """Tests for TimeSeriesClient.create_channel method."""

    @responses.activate
    def test_create_channel_success(self, mock_session_manager):
        """Test successful channel creation."""
        responses.add(
            responses.POST,
            "https://api.test.com/timeseries/pkg-123/channels",
            json={
                "content": {
                    "id": "N:channel:new-id-456",
                    "name": "Test Channel",
                    "start": 1000000,
                    "end": 2000000,
                    "unit": "uV",
                    "rate": 30000.0,
                    "channelType": "CONTINUOUS",
                    "group": "default",
                    "lastAnnotation": 0,
                },
                "properties": [{"key": "value"}],
            },
            status=200,
        )

        client = TimeSeriesClient("https://api.test.com", mock_session_manager)
        channel = TimeSeriesChannel(index=5, name="Test Channel", rate=30000.0, start=1000000, end=2000000)

        result = client.create_channel("pkg-123", channel)

        assert result.id == "N:channel:new-id-456"
        assert result.name == "Test Channel"
        assert result.index == 5  # Index should be preserved from input

    @responses.activate
    def test_create_channel_sends_correct_body(self, mock_session_manager):
        """Test that create_channel sends correct request body."""
        responses.add(
            responses.POST,
            "https://api.test.com/timeseries/pkg-123/channels",
            json={
                "content": {
                    "id": "N:channel:id",
                    "name": "Ch1",
                    "start": 0,
                    "end": 1000,
                    "unit": "mV",
                    "rate": 1000.0,
                    "channelType": "UNIT",
                    "group": "test_group",
                },
                "properties": [],
            },
            status=200,
        )

        client = TimeSeriesClient("https://api.test.com", mock_session_manager)
        channel = TimeSeriesChannel(
            index=0, name="Ch1", rate=1000.0, start=0, end=1000, type="UNIT", unit="mV", group="test_group"
        )

        client.create_channel("pkg-123", channel)

        body = json.loads(responses.calls[0].request.body)
        assert body["name"] == "Ch1"
        assert body["rate"] == 1000.0
        assert body["channelType"] == "UNIT"  # 'type' should be renamed to 'channelType'
        assert "type" not in body  # Original 'type' key should be removed
        assert body["unit"] == "mV"
        assert body["group"] == "test_group"

    @responses.activate
    def test_create_channel_includes_auth_header(self, mock_session_manager):
        """Test that authorization header is included."""
        responses.add(
            responses.POST,
            "https://api.test.com/timeseries/pkg-123/channels",
            json={
                "content": {
                    "id": "N:channel:id",
                    "name": "Ch1",
                    "start": 0,
                    "end": 1000,
                    "unit": "uV",
                    "rate": 1000.0,
                    "channelType": "CONTINUOUS",
                    "group": "default",
                },
                "properties": [],
            },
            status=200,
        )

        client = TimeSeriesClient("https://api.test.com", mock_session_manager)
        channel = TimeSeriesChannel(index=0, name="Ch1", rate=1000.0, start=0, end=1000)

        client.create_channel("pkg-123", channel)

        assert responses.calls[0].request.headers["Authorization"] == "Bearer mock-token-12345"

    @responses.activate
    def test_create_channel_raises_on_http_error(self, mock_session_manager):
        """Test that HTTP errors are raised."""
        responses.add(
            responses.POST,
            "https://api.test.com/timeseries/pkg-123/channels",
            json={"error": "Bad request"},
            status=400,
        )

        client = TimeSeriesClient("https://api.test.com", mock_session_manager)
        channel = TimeSeriesChannel(index=0, name="Ch1", rate=1000.0, start=0, end=1000)

        with pytest.raises(Exception):
            client.create_channel("pkg-123", channel)


class TestTimeSeriesClientGetPackageChannels:
    """Tests for TimeSeriesClient.get_package_channels method."""

    @responses.activate
    def test_get_package_channels_success(self, mock_session_manager):
        """Test successful channel retrieval."""
        responses.add(
            responses.GET,
            "https://api.test.com/timeseries/pkg-123/channels",
            json=[
                {
                    "content": {
                        "id": "N:channel:ch1",
                        "name": "Channel 1",
                        "start": 0,
                        "end": 1000,
                        "unit": "uV",
                        "rate": 30000.0,
                        "channelType": "CONTINUOUS",
                        "group": "group_a",
                    },
                    "properties": [{"key": "value1"}],
                },
                {
                    "content": {
                        "id": "N:channel:ch2",
                        "name": "Channel 2",
                        "start": 0,
                        "end": 1000,
                        "unit": "mV",
                        "rate": 1000.0,
                        "channelType": "UNIT",
                        "group": "group_b",
                    },
                    "properties": [{"key": "value2"}],
                },
            ],
            status=200,
        )

        client = TimeSeriesClient("https://api.test.com", mock_session_manager)
        channels = client.get_package_channels("pkg-123")

        assert len(channels) == 2
        assert channels[0].name == "Channel 1"
        assert channels[0].id == "N:channel:ch1"
        assert channels[0].rate == 30000.0
        assert channels[1].name == "Channel 2"
        assert channels[1].type == "UNIT"

    @responses.activate
    def test_get_package_channels_empty_list(self, mock_session_manager):
        """Test retrieval of empty channel list."""
        responses.add(responses.GET, "https://api.test.com/timeseries/pkg-123/channels", json=[], status=200)

        client = TimeSeriesClient("https://api.test.com", mock_session_manager)
        channels = client.get_package_channels("pkg-123")

        assert channels == []

    @responses.activate
    def test_get_package_channels_includes_auth_header(self, mock_session_manager):
        """Test that authorization header is included."""
        responses.add(responses.GET, "https://api.test.com/timeseries/pkg-123/channels", json=[], status=200)

        client = TimeSeriesClient("https://api.test.com", mock_session_manager)
        client.get_package_channels("pkg-123")

        assert responses.calls[0].request.headers["Authorization"] == "Bearer mock-token-12345"

    @responses.activate
    def test_get_package_channels_raises_on_http_error(self, mock_session_manager):
        """Test that HTTP errors are raised."""
        responses.add(
            responses.GET, "https://api.test.com/timeseries/pkg-123/channels", json={"error": "Not found"}, status=404
        )

        client = TimeSeriesClient("https://api.test.com", mock_session_manager)

        with pytest.raises(Exception):
            client.get_package_channels("pkg-123")


class TestTimeSeriesClientRetryBehavior:
    """Tests for retry behavior with session refresh."""

    @responses.activate
    def test_create_channel_retries_on_401(self, mock_session_manager):
        """Test that create_channel retries after 401."""
        # First call returns 401
        responses.add(
            responses.POST,
            "https://api.test.com/timeseries/pkg-123/channels",
            json={"error": "Unauthorized"},
            status=401,
        )
        # Second call succeeds
        responses.add(
            responses.POST,
            "https://api.test.com/timeseries/pkg-123/channels",
            json={
                "content": {
                    "id": "N:channel:id",
                    "name": "Ch1",
                    "start": 0,
                    "end": 1000,
                    "unit": "uV",
                    "rate": 1000.0,
                    "channelType": "CONTINUOUS",
                    "group": "default",
                },
                "properties": [],
            },
            status=200,
        )

        client = TimeSeriesClient("https://api.test.com", mock_session_manager)
        channel = TimeSeriesChannel(index=0, name="Ch1", rate=1000.0, start=0, end=1000)

        result = client.create_channel("pkg-123", channel)

        assert result.id == "N:channel:id"
        mock_session_manager.refresh_session.assert_called_once()

    @responses.activate
    def test_get_package_channels_retries_on_403(self, mock_session_manager):
        """Test that get_package_channels retries after 403."""
        # First call returns 403
        responses.add(
            responses.GET, "https://api.test.com/timeseries/pkg-123/channels", json={"error": "Forbidden"}, status=403
        )
        # Second call succeeds
        responses.add(responses.GET, "https://api.test.com/timeseries/pkg-123/channels", json=[], status=200)

        client = TimeSeriesClient("https://api.test.com", mock_session_manager)
        channels = client.get_package_channels("pkg-123")

        assert channels == []
        mock_session_manager.refresh_session.assert_called_once()
