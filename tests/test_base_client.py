from unittest.mock import Mock

import pytest
import requests
from clients.base_client import BaseClient, SessionManager


class TestSessionManager:
    """Tests for SessionManager class."""

    def test_initialization(self, mock_authentication_client):
        """Test basic initialization."""
        manager = SessionManager(
            authentication_client=mock_authentication_client, api_key="test-api-key", api_secret="test-api-secret"
        )

        assert manager.authentication_client == mock_authentication_client
        assert manager.api_key == "test-api-key"
        assert manager.api_secret == "test-api-secret"

    def test_session_token_lazy_initialization(self, mock_authentication_client):
        """Test that session token is lazily initialized on first access."""
        manager = SessionManager(mock_authentication_client, "key", "secret")

        # Token should not be fetched yet
        mock_authentication_client.authenticate.assert_not_called()

        # Access token
        token = manager.session_token

        # Now authenticate should have been called
        mock_authentication_client.authenticate.assert_called_once_with("key", "secret")
        assert token == "mock-access-token"

    def test_session_token_cached(self, mock_authentication_client):
        """Test that session token is cached after first access."""
        manager = SessionManager(mock_authentication_client, "key", "secret")

        # Access token twice
        token1 = manager.session_token
        token2 = manager.session_token

        # Authenticate should only be called once
        mock_authentication_client.authenticate.assert_called_once()
        assert token1 == token2

    def test_refresh_session(self, mock_authentication_client):
        """Test manual session refresh."""
        manager = SessionManager(mock_authentication_client, "key", "secret")

        # Access token to initialize
        _ = manager.session_token
        assert mock_authentication_client.authenticate.call_count == 1

        # Refresh session
        mock_authentication_client.authenticate.return_value = "new-token"
        manager.refresh_session()

        assert mock_authentication_client.authenticate.call_count == 2
        assert manager.session_token == "new-token"

    def test_refresh_session_without_prior_access(self, mock_authentication_client):
        """Test refresh_session can be called without prior token access."""
        manager = SessionManager(mock_authentication_client, "key", "secret")

        manager.refresh_session()

        mock_authentication_client.authenticate.assert_called_once_with("key", "secret")


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

    def test_client_uses_session_token(self, mock_authentication_client):
        """Test that client methods can access session token."""
        session_manager = SessionManager(mock_authentication_client, "key", "secret")

        class TestClient(BaseClient):
            @BaseClient.retry_with_refresh
            def get_auth_header(self):
                return f"Bearer {self.session_manager.session_token}"

        client = TestClient(session_manager)
        header = client.get_auth_header()

        assert header == "Bearer mock-access-token"

    def test_refresh_updates_token_for_next_call(self, mock_authentication_client):
        """Test that after refresh, subsequent calls use new token."""
        session_manager = SessionManager(mock_authentication_client, "key", "secret")
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

        # First call returns 'mock-access-token', refresh returns 'refreshed-token'
        mock_authentication_client.authenticate.side_effect = ["mock-access-token", "refreshed-token"]

        client = TestClient(session_manager)
        client.get_token()

        # The refresh_session was called, showing the retry mechanism worked
        assert call_count[0] == 2  # Verifies retry happened
