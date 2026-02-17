from unittest.mock import Mock, patch

import pytest
import responses
from clients.authentication_client import AuthenticationClient


class TestAuthenticationClientInit:
    """Tests for AuthenticationClient initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        client = AuthenticationClient("https://api.test.com")
        assert client.api_host == "https://api.test.com"
        assert client._cognito_config is None


class TestAuthenticationClientRefresh:
    """Tests for AuthenticationClient.refresh method."""

    @responses.activate
    def test_refresh_success(self):
        """Test successful token refresh."""
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"tokenPool": {"appClientId": "test-client-id"}, "region": "us-east-1"},
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.return_value = {
            "AuthenticationResult": {"AccessToken": "refreshed-access-token"}
        }

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client):
            client = AuthenticationClient("https://api.test.com")
            token = client.refresh("my-refresh-token")

        assert token == "refreshed-access-token"

    @responses.activate
    def test_refresh_calls_cognito_with_correct_params(self):
        """Test that Cognito is called with REFRESH_TOKEN_AUTH flow."""
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"tokenPool": {"appClientId": "my-app-client-id"}, "region": "us-west-2"},
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.return_value = {"AuthenticationResult": {"AccessToken": "token"}}

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client):
            client = AuthenticationClient("https://api.test.com")
            client.refresh("the-refresh-token")

        mock_cognito_client.initiate_auth.assert_called_once_with(
            AuthFlow="REFRESH_TOKEN_AUTH",
            AuthParameters={"REFRESH_TOKEN": "the-refresh-token"},
            ClientId="my-app-client-id",
        )

    @responses.activate
    def test_refresh_raises_on_cognito_error(self):
        """Test that Cognito errors during refresh are raised."""
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"tokenPool": {"appClientId": "client-id"}, "region": "us-east-1"},
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.side_effect = Exception("Token expired or revoked")

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client):
            client = AuthenticationClient("https://api.test.com")

            with pytest.raises(Exception, match="Token expired or revoked"):
                client.refresh("bad-refresh-token")

    @responses.activate
    def test_refresh_raises_on_config_http_error(self):
        """Test that HTTP errors from config endpoint are raised during refresh."""
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"error": "Server error"},
            status=500,
        )

        client = AuthenticationClient("https://api.test.com")

        with pytest.raises(Exception):
            client.refresh("refresh-token")


class TestCognitoConfigCaching:
    """Tests for Cognito config caching behavior."""

    @responses.activate
    def test_cognito_config_cached_across_calls(self):
        """Test that the cognito config endpoint is only called once."""
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"tokenPool": {"appClientId": "client-id"}, "region": "us-east-1"},
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.return_value = {"AuthenticationResult": {"AccessToken": "token"}}

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client):
            client = AuthenticationClient("https://api.test.com")
            client.refresh("refresh-token")
            client.refresh("refresh-token")

        # Config endpoint should only be called once despite two refresh calls
        assert len(responses.calls) == 1
