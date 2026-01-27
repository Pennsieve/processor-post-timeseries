import pytest
import responses
from clients.packages_client import PackagesClient


class TestPackagesClientInit:
    """Tests for PackagesClient initialization."""

    def test_initialization(self, mock_session_manager):
        """Test basic initialization."""
        client = PackagesClient("https://api.test.com", mock_session_manager)

        assert client.api_host == "https://api.test.com"
        assert client.session_manager == mock_session_manager


class TestPackagesClientGetParentPackageId:
    """Tests for PackagesClient.get_parent_package_id method."""

    @responses.activate
    def test_get_parent_package_id_success(self, mock_session_manager):
        """Test successful parent package ID retrieval."""
        responses.add(
            responses.GET,
            "https://api.test.com/packages/N:package:child-123?includeAncestors=true&startAtEpoch=false&limit=100&offset=0",
            json={
                "parent": {
                    "content": {
                        "nodeId": "N:collection:parent-456",
                        "name": "Parent Folder",
                    }
                },
                "content": {
                    "nodeId": "N:package:child-123",
                    "name": "Child Package",
                },
            },
            status=200,
        )

        client = PackagesClient("https://api.test.com", mock_session_manager)
        parent_id = client.get_parent_package_id("N:package:child-123")

        assert parent_id == "N:collection:parent-456"

    @responses.activate
    def test_get_parent_package_id_includes_auth_header(self, mock_session_manager):
        """Test that authorization header is included."""
        responses.add(
            responses.GET,
            "https://api.test.com/packages/N:package:test-123?includeAncestors=true&startAtEpoch=false&limit=100&offset=0",
            json={
                "parent": {"content": {"nodeId": "N:collection:parent-456"}},
                "content": {"nodeId": "N:package:test-123"},
            },
            status=200,
        )

        client = PackagesClient("https://api.test.com", mock_session_manager)
        client.get_parent_package_id("N:package:test-123")

        assert responses.calls[0].request.headers["Authorization"] == "Bearer mock-token-12345"

    @responses.activate
    def test_get_parent_package_id_raises_on_http_error(self, mock_session_manager):
        """Test that HTTP errors are raised."""
        responses.add(
            responses.GET,
            "https://api.test.com/packages/N:package:not-found?includeAncestors=true&startAtEpoch=false&limit=100&offset=0",
            json={"error": "Not found"},
            status=404,
        )

        client = PackagesClient("https://api.test.com", mock_session_manager)

        with pytest.raises(Exception):
            client.get_parent_package_id("N:package:not-found")

    @responses.activate
    def test_get_parent_package_id_retries_on_401(self, mock_session_manager):
        """Test that get_parent_package_id retries after 401."""
        # First call returns 401
        responses.add(
            responses.GET,
            "https://api.test.com/packages/N:package:test-123?includeAncestors=true&startAtEpoch=false&limit=100&offset=0",
            json={"error": "Unauthorized"},
            status=401,
        )
        # Second call succeeds
        responses.add(
            responses.GET,
            "https://api.test.com/packages/N:package:test-123?includeAncestors=true&startAtEpoch=false&limit=100&offset=0",
            json={
                "parent": {"content": {"nodeId": "N:collection:parent-456"}},
                "content": {"nodeId": "N:package:test-123"},
            },
            status=200,
        )

        client = PackagesClient("https://api.test.com", mock_session_manager)
        parent_id = client.get_parent_package_id("N:package:test-123")

        assert parent_id == "N:collection:parent-456"
        mock_session_manager.refresh_session.assert_called_once()


class TestPackagesClientUpdateProperties:
    """Tests for PackagesClient.update_properties method."""

    @responses.activate
    def test_update_properties_success(self, mock_session_manager):
        """Test successful property update."""
        responses.add(
            responses.PUT,
            "https://api.test.com/packages/N:package:test-123?updateStorage=true",
            json={"success": True},
            status=200,
        )

        client = PackagesClient("https://api.test.com", mock_session_manager)
        properties = [{"key": "test_key", "value": "test_value", "dataType": "string"}]

        result = client.update_properties("N:package:test-123", properties)

        assert result is None

    @responses.activate
    def test_update_properties_sends_correct_payload(self, mock_session_manager):
        """Test that update_properties sends correct request body."""
        responses.add(
            responses.PUT,
            "https://api.test.com/packages/N:package:test-123?updateStorage=true",
            json={"success": True},
            status=200,
        )

        client = PackagesClient("https://api.test.com", mock_session_manager)
        properties = [
            {"key": "key1", "value": "value1", "dataType": "string"},
            {"key": "key2", "value": "value2", "dataType": "integer"},
        ]

        client.update_properties("N:package:test-123", properties)

        import json

        body = json.loads(responses.calls[0].request.body)
        assert body["properties"] == properties

    @responses.activate
    def test_update_properties_includes_auth_header(self, mock_session_manager):
        """Test that authorization header is included."""
        responses.add(
            responses.PUT,
            "https://api.test.com/packages/N:package:test-123?updateStorage=true",
            json={"success": True},
            status=200,
        )

        client = PackagesClient("https://api.test.com", mock_session_manager)
        client.update_properties("N:package:test-123", [])

        assert responses.calls[0].request.headers["Authorization"] == "Bearer mock-token-12345"

    @responses.activate
    def test_update_properties_raises_on_http_error(self, mock_session_manager):
        """Test that HTTP errors are raised."""
        responses.add(
            responses.PUT,
            "https://api.test.com/packages/N:package:test-123?updateStorage=true",
            json={"error": "Bad request"},
            status=400,
        )

        client = PackagesClient("https://api.test.com", mock_session_manager)

        with pytest.raises(Exception):
            client.update_properties("N:package:test-123", [])


class TestPackagesClientSetTimeseriesProperties:
    """Tests for PackagesClient.set_timeseries_properties method."""

    @responses.activate
    def test_set_timeseries_properties_success(self, mock_session_manager):
        """Test successful timeseries properties update."""
        responses.add(
            responses.PUT,
            "https://api.test.com/packages/N:package:test-123?updateStorage=true",
            json={"success": True},
            status=200,
        )

        client = PackagesClient("https://api.test.com", mock_session_manager)
        result = client.set_timeseries_properties("N:package:test-123")

        assert result is None

    @responses.activate
    def test_set_timeseries_properties_sends_correct_payload(self, mock_session_manager):
        """Test that set_timeseries_properties sends the correct properties."""
        responses.add(
            responses.PUT,
            "https://api.test.com/packages/N:package:test-123?updateStorage=true",
            json={"success": True},
            status=200,
        )

        client = PackagesClient("https://api.test.com", mock_session_manager)
        client.set_timeseries_properties("N:package:test-123")

        import json

        body = json.loads(responses.calls[0].request.body)
        properties = body["properties"]

        # Should have exactly 2 properties
        assert len(properties) == 2

        # Check subtype property
        subtype_prop = next(p for p in properties if p["key"] == "subtype")
        assert subtype_prop["value"] == "pennsieve_timeseries"
        assert subtype_prop["dataType"] == "string"
        assert subtype_prop["category"] == "Viewer"
        assert subtype_prop["hidden"] is True

        # Check icon property
        icon_prop = next(p for p in properties if p["key"] == "icon")
        assert icon_prop["value"] == "timeseries"
        assert icon_prop["dataType"] == "string"
        assert icon_prop["category"] == "Pennsieve"
        assert icon_prop["hidden"] is True
