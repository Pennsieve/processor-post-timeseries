from unittest.mock import Mock

import pytest
import requests
from clients.base_client import BaseClient, SessionManager


class TestSessionManager:
    """Tests for SessionManager class."""

    def test_session_token_delegates_to_auth_provider(self):
        """Test that session_token reads from the auth provider."""
        mock_provider = Mock()
        mock_provider.get_session_token.return_value = "my-token"
        manager = SessionManager(mock_provider)

        assert manager.session_token == "my-token"
        mock_provider.get_session_token.assert_called_once()

    def test_refresh_session_delegates_to_auth_provider(self):
        """Test that refresh_session calls auth provider's refresh."""
        mock_provider = Mock()
        manager = SessionManager(mock_provider)

        manager.refresh_session()

        mock_provider.refresh.assert_called_once()


class TestBaseClient:
    """Tests for BaseClient class."""

    def test_initialization(self, mock_session_manager):
        """Test basic initialization."""
        client = BaseClient(mock_session_manager)
        assert client.session_manager == mock_session_manager

    def test_retry_with_refresh_success_on_first_try(self, mock_session_manager):
        """Test that successful call doesn't trigger refresh."""

        class TestClient(BaseClient):
            @BaseClient.retry_with_refresh
            def test_method(self):
                return "success"

        client = TestClient(mock_session_manager)
        result = client.test_method()

        assert result == "success"
        mock_session_manager.refresh_session.assert_not_called()

    def test_retry_with_refresh_on_401(self, mock_session_manager):
        """Test that 401 error triggers session refresh and retry."""
        call_count = [0]

        class TestClient(BaseClient):
            @BaseClient.retry_with_refresh
            def test_method(self):
                call_count[0] += 1
                if call_count[0] == 1:
                    # First call fails with 401
                    response = Mock()
                    response.status_code = 401
                    error = requests.exceptions.HTTPError(response=response)
                    raise error
                return "success_after_retry"

        client = TestClient(mock_session_manager)
        result = client.test_method()

        assert result == "success_after_retry"
        mock_session_manager.refresh_session.assert_called_once()
        assert call_count[0] == 2

    def test_retry_with_refresh_on_403(self, mock_session_manager):
        """Test that 403 error triggers session refresh and retry."""
        call_count = [0]

        class TestClient(BaseClient):
            @BaseClient.retry_with_refresh
            def test_method(self):
                call_count[0] += 1
                if call_count[0] == 1:
                    response = Mock()
                    response.status_code = 403
                    error = requests.exceptions.HTTPError(response=response)
                    raise error
                return "success"

        client = TestClient(mock_session_manager)
        result = client.test_method()

        assert result == "success"
        mock_session_manager.refresh_session.assert_called_once()

    def test_retry_with_refresh_propagates_other_http_errors(self, mock_session_manager):
        """Test that non-401/403 HTTP errors are propagated without retry."""

        class TestClient(BaseClient):
            @BaseClient.retry_with_refresh
            def test_method(self):
                response = Mock()
                response.status_code = 500
                error = requests.exceptions.HTTPError(response=response)
                raise error

        client = TestClient(mock_session_manager)

        with pytest.raises(requests.exceptions.HTTPError):
            client.test_method()

        mock_session_manager.refresh_session.assert_not_called()

    def test_retry_with_refresh_propagates_non_http_errors(self, mock_session_manager):
        """Test that non-HTTP exceptions are propagated without retry."""

        class TestClient(BaseClient):
            @BaseClient.retry_with_refresh
            def test_method(self):
                raise ValueError("Something went wrong")

        client = TestClient(mock_session_manager)

        with pytest.raises(ValueError, match="Something went wrong"):
            client.test_method()

        mock_session_manager.refresh_session.assert_not_called()

    def test_retry_with_refresh_passes_args(self, mock_session_manager):
        """Test that arguments are passed correctly to decorated method."""

        class TestClient(BaseClient):
            @BaseClient.retry_with_refresh
            def test_method(self, arg1, arg2, kwarg1=None):
                return f"{arg1}-{arg2}-{kwarg1}"

        client = TestClient(mock_session_manager)
        result = client.test_method("a", "b", kwarg1="c")

        assert result == "a-b-c"

    def test_retry_with_refresh_fails_on_persistent_401(self, mock_session_manager):
        """Test that persistent 401 after refresh is propagated."""

        class TestClient(BaseClient):
            @BaseClient.retry_with_refresh
            def test_method(self):
                response = Mock()
                response.status_code = 401
                error = requests.exceptions.HTTPError(response=response)
                raise error

        client = TestClient(mock_session_manager)

        with pytest.raises(requests.exceptions.HTTPError):
            client.test_method()

        # Should have refreshed once and then re-raised on second failure
        mock_session_manager.refresh_session.assert_called_once()


class TestBaseClientIntegration:
    """Integration tests for BaseClient with SessionManager."""

    def test_client_uses_session_token(self):
        """Test that client methods can access session token via auth provider."""
        mock_provider = Mock()
        mock_provider.get_session_token.return_value = "my-access-token"
        session_manager = SessionManager(mock_provider)

        class TestClient(BaseClient):
            @BaseClient.retry_with_refresh
            def get_auth_header(self):
                return f"Bearer {self.session_manager.session_token}"

        client = TestClient(session_manager)
        header = client.get_auth_header()

        assert header == "Bearer my-access-token"

    def test_retry_refreshes_token_and_succeeds(self):
        """Test that a 401 triggers refresh via auth provider and the retry uses the new token."""
        mock_provider = Mock()
        # get_session_token is only called on the successful retry (first attempt raises before reading token)
        mock_provider.get_session_token.return_value = "refreshed-token"

        session_manager = SessionManager(mock_provider)

        call_count = [0]

        class TestClient(BaseClient):
            @BaseClient.retry_with_refresh
            def get_token(self):
                call_count[0] += 1
                if call_count[0] == 1:
                    response = Mock()
                    response.status_code = 401
                    raise requests.exceptions.HTTPError(response=response)
                return self.session_manager.session_token

        client = TestClient(session_manager)
        token = client.get_token()

        assert token == "refreshed-token"
        mock_provider.refresh.assert_called_once()
        assert call_count[0] == 2
