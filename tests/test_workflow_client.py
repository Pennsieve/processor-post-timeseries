import pytest
import json
import responses
from unittest.mock import Mock

from clients.workflow_client import WorkflowClient, WorkflowInstance


class TestWorkflowInstance:
    """Tests for WorkflowInstance class."""

    def test_initialization(self):
        """Test WorkflowInstance initialization."""
        instance = WorkflowInstance(id="workflow-123", dataset_id="dataset-456", package_ids=["pkg-1", "pkg-2"])

        assert instance.id == "workflow-123"
        assert instance.dataset_id == "dataset-456"
        assert instance.package_ids == ["pkg-1", "pkg-2"]

    def test_initialization_with_empty_package_ids(self):
        """Test WorkflowInstance with empty package list."""
        instance = WorkflowInstance(id="workflow-123", dataset_id="dataset-456", package_ids=[])

        assert instance.package_ids == []

    def test_initialization_with_single_package(self):
        """Test WorkflowInstance with single package."""
        instance = WorkflowInstance(id="workflow-123", dataset_id="dataset-456", package_ids=["single-pkg"])

        assert len(instance.package_ids) == 1
        assert instance.package_ids[0] == "single-pkg"


class TestWorkflowClientInit:
    """Tests for WorkflowClient initialization."""

    def test_initialization(self, mock_session_manager):
        """Test basic initialization."""
        client = WorkflowClient("https://api.test.com", mock_session_manager)

        assert client.api_host == "https://api.test.com"
        assert client.session_manager == mock_session_manager


class TestWorkflowClientGetWorkflowInstance:
    """Tests for WorkflowClient.get_workflow_instance method."""

    @responses.activate
    def test_get_workflow_instance_success(self, mock_session_manager):
        """Test successful workflow instance retrieval."""
        responses.add(
            responses.GET,
            "https://api.test.com/workflows/instances/wf-instance-123",
            json={"uuid": "wf-instance-123", "datasetId": "dataset-456", "packageIds": ["pkg-1", "pkg-2", "pkg-3"]},
            status=200,
        )

        client = WorkflowClient("https://api.test.com", mock_session_manager)
        result = client.get_workflow_instance("wf-instance-123")

        assert isinstance(result, WorkflowInstance)
        assert result.id == "wf-instance-123"
        assert result.dataset_id == "dataset-456"
        assert result.package_ids == ["pkg-1", "pkg-2", "pkg-3"]

    @responses.activate
    def test_get_workflow_instance_includes_auth_header(self, mock_session_manager):
        """Test that authorization header is included."""
        responses.add(
            responses.GET,
            "https://api.test.com/workflows/instances/wf-123",
            json={"uuid": "wf-123", "datasetId": "ds-1", "packageIds": []},
            status=200,
        )

        client = WorkflowClient("https://api.test.com", mock_session_manager)
        client.get_workflow_instance("wf-123")

        assert responses.calls[0].request.headers["Authorization"] == "Bearer mock-token-12345"
        assert responses.calls[0].request.headers["Accept"] == "application/json"

    @responses.activate
    def test_get_workflow_instance_raises_on_http_error(self, mock_session_manager):
        """Test that HTTP errors are raised."""
        responses.add(
            responses.GET, "https://api.test.com/workflows/instances/wf-123", json={"error": "Not found"}, status=404
        )

        client = WorkflowClient("https://api.test.com", mock_session_manager)

        with pytest.raises(Exception):
            client.get_workflow_instance("wf-123")

    @responses.activate
    def test_get_workflow_instance_raises_on_invalid_json(self, mock_session_manager):
        """Test that invalid JSON raises error."""
        responses.add(responses.GET, "https://api.test.com/workflows/instances/wf-123", body="not json", status=200)

        client = WorkflowClient("https://api.test.com", mock_session_manager)

        with pytest.raises(json.JSONDecodeError):
            client.get_workflow_instance("wf-123")

    @responses.activate
    def test_get_workflow_instance_with_single_package(self, mock_session_manager):
        """Test workflow instance with single package ID."""
        responses.add(
            responses.GET,
            "https://api.test.com/workflows/instances/wf-123",
            json={"uuid": "wf-123", "datasetId": "ds-1", "packageIds": ["single-pkg"]},
            status=200,
        )

        client = WorkflowClient("https://api.test.com", mock_session_manager)
        result = client.get_workflow_instance("wf-123")

        assert len(result.package_ids) == 1
        assert result.package_ids[0] == "single-pkg"


class TestWorkflowClientRetryBehavior:
    """Tests for retry behavior with session refresh."""

    @responses.activate
    def test_get_workflow_instance_retries_on_401(self, mock_session_manager):
        """Test that get_workflow_instance retries after 401."""
        # First call returns 401
        responses.add(
            responses.GET, "https://api.test.com/workflows/instances/wf-123", json={"error": "Unauthorized"}, status=401
        )
        # Second call succeeds
        responses.add(
            responses.GET,
            "https://api.test.com/workflows/instances/wf-123",
            json={"uuid": "wf-123", "datasetId": "ds-1", "packageIds": []},
            status=200,
        )

        client = WorkflowClient("https://api.test.com", mock_session_manager)
        result = client.get_workflow_instance("wf-123")

        assert result.id == "wf-123"
        mock_session_manager.refresh_session.assert_called_once()

    @responses.activate
    def test_get_workflow_instance_retries_on_403(self, mock_session_manager):
        """Test that get_workflow_instance retries after 403."""
        # First call returns 403
        responses.add(
            responses.GET, "https://api.test.com/workflows/instances/wf-123", json={"error": "Forbidden"}, status=403
        )
        # Second call succeeds
        responses.add(
            responses.GET,
            "https://api.test.com/workflows/instances/wf-123",
            json={"uuid": "wf-123", "datasetId": "ds-1", "packageIds": []},
            status=200,
        )

        client = WorkflowClient("https://api.test.com", mock_session_manager)
        result = client.get_workflow_instance("wf-123")

        assert result.id == "wf-123"
        mock_session_manager.refresh_session.assert_called_once()
