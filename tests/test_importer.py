from unittest.mock import Mock

import responses
from clients.packages_client import PackagesClient
from importer import determine_target_package


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
        # Name lookups happen for every package; none is a header, so the
        # parent lookup follows and must only touch the first package.
        for package_id, name in [
            ("N:package:first", "a.edf"),
            ("N:package:second", "b.edf"),
            ("N:package:third", "c.edf"),
        ]:
            responses.add(
                responses.GET,
                f"https://api.test.com/packages/{package_id}",
                json={"content": {"nodeId": package_id, "name": name}},
                status=200,
            )
        # Only set up a parent response for the first package
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

        # Verify only one parent lookup was made, and for the first package
        parent_calls = [c for c in responses.calls if "includeAncestors" in c.request.url]
        assert len(parent_calls) == 1
        assert "N:package:first" in parent_calls[0].request.url
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


class TestHeaderPackageTarget:
    """Tests for preferring the .lay header package among several inputs."""

    API = "https://api.test.com"

    def _add_name(self, package_id, name):
        responses.add(
            responses.GET,
            f"{self.API}/packages/{package_id}",
            json={"content": {"nodeId": package_id, "name": name}},
            status=200,
        )

    @responses.activate
    def test_prefers_lay_package_over_parent_folder(self, mock_session_manager):
        """A Persyst pair targets the .lay package, not the folder holding it."""
        self._add_name("N:package:dat-1", "wave_sin.dat")
        self._add_name("N:package:lay-2", "wave_sin.lay")
        packages_client = PackagesClient(self.API, mock_session_manager)

        result = determine_target_package(packages_client, ["N:package:dat-1", "N:package:lay-2"])

        assert result == "N:package:lay-2"
        # the parent lookup must not have been needed
        assert all("includeAncestors" not in c.request.url for c in responses.calls)

    @responses.activate
    def test_suffix_match_ignores_case(self, mock_session_manager):
        self._add_name("N:package:dat-1", "REC.DAT")
        self._add_name("N:package:lay-2", "REC.LAY")
        packages_client = PackagesClient(self.API, mock_session_manager)

        assert determine_target_package(packages_client, ["N:package:dat-1", "N:package:lay-2"]) == "N:package:lay-2"

    @responses.activate
    def test_falls_back_to_parent_when_no_header(self, mock_session_manager):
        """Inputs without a .lay keep the original parent-folder behaviour."""
        self._add_name("N:package:a", "one.edf")
        self._add_name("N:package:b", "two.edf")
        responses.add(
            responses.GET,
            f"{self.API}/packages/N:package:a?includeAncestors=true&startAtEpoch=false&limit=100&offset=0",
            json={"parent": {"content": {"nodeId": "N:collection:parent"}}, "content": {"nodeId": "N:package:a"}},
            status=200,
        )
        packages_client = PackagesClient(self.API, mock_session_manager)

        assert determine_target_package(packages_client, ["N:package:a", "N:package:b"]) == "N:collection:parent"

    @responses.activate
    def test_falls_back_to_parent_when_two_headers(self, mock_session_manager):
        """Two .lay files are ambiguous, so fall back rather than guess."""
        self._add_name("N:package:a", "one.lay")
        self._add_name("N:package:b", "two.lay")
        responses.add(
            responses.GET,
            f"{self.API}/packages/N:package:a?includeAncestors=true&startAtEpoch=false&limit=100&offset=0",
            json={"parent": {"content": {"nodeId": "N:collection:parent"}}, "content": {"nodeId": "N:package:a"}},
            status=200,
        )
        packages_client = PackagesClient(self.API, mock_session_manager)

        assert determine_target_package(packages_client, ["N:package:a", "N:package:b"]) == "N:collection:parent"

    @responses.activate
    def test_falls_back_to_parent_when_name_lookup_fails(self, mock_session_manager):
        responses.add(responses.GET, f"{self.API}/packages/N:package:a", status=500)
        responses.add(
            responses.GET,
            f"{self.API}/packages/N:package:a?includeAncestors=true&startAtEpoch=false&limit=100&offset=0",
            json={"parent": {"content": {"nodeId": "N:collection:parent"}}, "content": {"nodeId": "N:package:a"}},
            status=200,
        )
        packages_client = PackagesClient(self.API, mock_session_manager)

        assert determine_target_package(packages_client, ["N:package:a", "N:package:b"]) == "N:collection:parent"

    def test_single_package_skips_name_lookup(self, mock_session_manager):
        """A single input is used as-is, with no API call at all."""
        packages_client = PackagesClient(self.API, mock_session_manager)
        packages_client.get_package_name = Mock(side_effect=AssertionError("should not be called"))

        assert determine_target_package(packages_client, ["N:package:only"]) == "N:package:only"
