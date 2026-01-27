from unittest.mock import Mock

import responses
from clients.packages_client import PackagesClient
from importer import determine_target_package, find_first_package_id


class TestFindFirstPackageId:
    """Tests for find_first_package_id function."""

    def test_finds_first_package_with_prefix(self):
        """Test finding first package ID with N:package: prefix."""
        package_ids = [
            "N:collection:abc-123",
            "N:package:def-456",
            "N:package:ghi-789",
        ]

        result = find_first_package_id(package_ids)

        assert result == "N:package:def-456"

    def test_returns_none_when_no_package_prefix(self):
        """Test returns None when no N:package: prefix found."""
        package_ids = [
            "N:collection:abc-123",
            "N:dataset:def-456",
            "some-other-id",
        ]

        result = find_first_package_id(package_ids)

        assert result is None

    def test_returns_none_for_empty_list(self):
        """Test returns None for empty list."""
        result = find_first_package_id([])

        assert result is None

    def test_returns_first_even_if_multiple_packages(self):
        """Test returns first package even when multiple exist."""
        package_ids = [
            "N:package:first-111",
            "N:package:second-222",
            "N:package:third-333",
        ]

        result = find_first_package_id(package_ids)

        assert result == "N:package:first-111"

    def test_single_package_id(self):
        """Test with single package ID."""
        package_ids = ["N:package:only-one"]

        result = find_first_package_id(package_ids)

        assert result == "N:package:only-one"

    def test_case_sensitive_prefix(self):
        """Test that prefix matching is case-sensitive."""
        package_ids = [
            "n:package:lowercase",  # lowercase n
            "N:PACKAGE:uppercase",  # uppercase PACKAGE
            "N:package:correct",
        ]

        result = find_first_package_id(package_ids)

        assert result == "N:package:correct"


class TestDetermineTargetPackage:
    """Tests for determine_target_package function."""

    def test_returns_none_for_empty_list(self, mock_session_manager):
        """Test returns None when package_ids is empty."""
        packages_client = PackagesClient("https://api.test.com", mock_session_manager)

        result = determine_target_package(packages_client, [])

        assert result is None

    def test_returns_single_package_directly(self, mock_session_manager):
        """Test returns the single package ID directly when only one exists."""
        packages_client = PackagesClient("https://api.test.com", mock_session_manager)
        package_ids = ["N:package:single-123"]

        result = determine_target_package(packages_client, package_ids)

        assert result == "N:package:single-123"

    def test_returns_single_package_regardless_of_type(self, mock_session_manager):
        """Test returns single package even if it's not N:package: prefixed."""
        packages_client = PackagesClient("https://api.test.com", mock_session_manager)
        package_ids = ["N:collection:single-collection"]

        result = determine_target_package(packages_client, package_ids)

        assert result == "N:collection:single-collection"

    @responses.activate
    def test_multiple_packages_returns_parent_of_first(self, mock_session_manager):
        """Test with multiple packages returns parent of first N:package: package."""
        responses.add(
            responses.GET,
            "https://api.test.com/packages/N:package:first-123?includeAncestors=true&startAtEpoch=false&limit=100&offset=0",
            json={
                "parent": {"content": {"nodeId": "N:collection:parent-folder"}},
                "content": {"nodeId": "N:package:first-123"},
            },
            status=200,
        )

        packages_client = PackagesClient("https://api.test.com", mock_session_manager)
        package_ids = [
            "N:package:first-123",
            "N:package:second-456",
            "N:package:third-789",
        ]

        result = determine_target_package(packages_client, package_ids)

        assert result == "N:collection:parent-folder"

    @responses.activate
    def test_multiple_packages_finds_package_among_mixed_types(self, mock_session_manager):
        """Test finds N:package: among mixed ID types."""
        responses.add(
            responses.GET,
            "https://api.test.com/packages/N:package:the-package?includeAncestors=true&startAtEpoch=false&limit=100&offset=0",
            json={
                "parent": {"content": {"nodeId": "N:collection:parent-folder"}},
                "content": {"nodeId": "N:package:the-package"},
            },
            status=200,
        )

        packages_client = PackagesClient("https://api.test.com", mock_session_manager)
        package_ids = [
            "N:collection:not-a-package",
            "N:package:the-package",
            "N:dataset:also-not-package",
        ]

        result = determine_target_package(packages_client, package_ids)

        assert result == "N:collection:parent-folder"

    def test_multiple_packages_no_package_prefix_returns_none(self, mock_session_manager):
        """Test returns None when multiple IDs but none have N:package: prefix."""
        packages_client = PackagesClient("https://api.test.com", mock_session_manager)
        package_ids = [
            "N:collection:abc",
            "N:dataset:def",
            "some-other-id",
        ]

        result = determine_target_package(packages_client, package_ids)

        assert result is None

    @responses.activate
    def test_multiple_packages_api_error_returns_none(self, mock_session_manager):
        """Test returns None when API call fails for parent package."""
        responses.add(
            responses.GET,
            "https://api.test.com/packages/N:package:first-123?includeAncestors=true&startAtEpoch=false&limit=100&offset=0",
            json={"error": "Not found"},
            status=404,
        )

        packages_client = PackagesClient("https://api.test.com", mock_session_manager)
        package_ids = [
            "N:package:first-123",
            "N:package:second-456",
        ]

        result = determine_target_package(packages_client, package_ids)

        assert result is None

    @responses.activate
    def test_multiple_packages_uses_first_package_for_parent_lookup(self, mock_session_manager):
        """Test that only the first N:package: is used for parent lookup."""
        # Only set up response for first package
        responses.add(
            responses.GET,
            "https://api.test.com/packages/N:package:first?includeAncestors=true&startAtEpoch=false&limit=100&offset=0",
            json={
                "parent": {"content": {"nodeId": "N:collection:parent"}},
                "content": {"nodeId": "N:package:first"},
            },
            status=200,
        )

        packages_client = PackagesClient("https://api.test.com", mock_session_manager)
        package_ids = [
            "N:package:first",
            "N:package:second",
            "N:package:third",
        ]

        result = determine_target_package(packages_client, package_ids)

        # Verify only one API call was made
        assert len(responses.calls) == 1
        assert "N:package:first" in responses.calls[0].request.url
        assert result == "N:collection:parent"


class TestDetermineTargetPackageIntegration:
    """Integration-style tests for determine_target_package with mocked client."""

    def test_with_mock_packages_client(self):
        """Test using a fully mocked PackagesClient."""
        mock_client = Mock(spec=PackagesClient)
        mock_client.get_parent_package_id.return_value = "N:collection:mocked-parent"

        package_ids = ["N:package:pkg1", "N:package:pkg2"]

        result = determine_target_package(mock_client, package_ids)

        assert result == "N:collection:mocked-parent"
        mock_client.get_parent_package_id.assert_called_once_with("N:package:pkg1")

    def test_mock_client_raises_exception(self):
        """Test handling when mocked client raises exception."""
        mock_client = Mock(spec=PackagesClient)
        mock_client.get_parent_package_id.side_effect = Exception("API Error")

        package_ids = ["N:package:pkg1", "N:package:pkg2"]

        result = determine_target_package(mock_client, package_ids)

        assert result is None
