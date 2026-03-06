import base64
import json
from unittest.mock import Mock, patch

import pytest
import responses
from clients.authentication_client import (
    CognitoClient,
    KeySecretAuthProvider,
    TokenAuthProvider,
)


def _make_jwt(payload):
    """Build a fake JWT with the given payload dict (no signature verification)."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "RS256"}).encode()).rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"{header}.{body}.fake-signature"


class TestCognitoClient:
    """Tests for shared CognitoClient logic."""

    def test_initialization(self):
        client = CognitoClient("https://api.test.com")
        assert client.api_host == "https://api.test.com"
        assert client._cognito_config is None

    @responses.activate
    def test_authenticate_success(self):
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"userPool": {"appClientId": "test-client-id"}, "region": "us-east-1"},
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.return_value = {
            "AuthenticationResult": {
                "AccessToken": "test-access-token-12345",
                "RefreshToken": "test-refresh-token-67890",
            }
        }

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client):
            client = CognitoClient("https://api.test.com")
            access_token, refresh_token = client.authenticate("api-key", "api-secret")

        assert access_token == "test-access-token-12345"
        assert refresh_token == "test-refresh-token-67890"

    @responses.activate
    def test_authenticate_calls_cognito_with_correct_params(self):
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"userPool": {"appClientId": "my-app-client-id"}, "region": "us-west-2"},
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.return_value = {
            "AuthenticationResult": {"AccessToken": "token", "RefreshToken": "refresh"}
        }

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client) as mock_boto:
            client = CognitoClient("https://api.test.com")
            client.authenticate("my-api-key", "my-api-secret")

        mock_boto.assert_called_once_with(
            "cognito-idp", region_name="us-west-2", aws_access_key_id="", aws_secret_access_key=""
        )

        mock_cognito_client.initiate_auth.assert_called_once_with(
            AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters={"USERNAME": "my-api-key", "PASSWORD": "my-api-secret"},
            ClientId="my-app-client-id",
        )

    @responses.activate
    def test_authenticate_raises_on_config_http_error(self):
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"error": "Server error"},
            status=500,
        )

        client = CognitoClient("https://api.test.com")

        with pytest.raises(Exception):
            client.authenticate("key", "secret")

    @responses.activate
    def test_refresh_token_success(self):
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"userPool": {"appClientId": "test-client-id"}, "region": "us-east-1"},
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.return_value = {
            "AuthenticationResult": {"AccessToken": "refreshed-access-token"}
        }

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client):
            client = CognitoClient("https://api.test.com")
            token = client.refresh_token("my-refresh-token")

        assert token == "refreshed-access-token"

    @responses.activate
    def test_refresh_token_calls_cognito_with_correct_params(self):
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"userPool": {"appClientId": "my-app-client-id"}, "region": "us-west-2"},
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.return_value = {"AuthenticationResult": {"AccessToken": "token"}}

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client):
            client = CognitoClient("https://api.test.com")
            client.refresh_token("the-refresh-token")

        mock_cognito_client.initiate_auth.assert_called_once_with(
            AuthFlow="REFRESH_TOKEN_AUTH",
            AuthParameters={"REFRESH_TOKEN": "the-refresh-token"},
            ClientId="my-app-client-id",
        )

    @responses.activate
    def test_refresh_token_includes_device_key_from_session_token(self):
        """Test that device_key is extracted from session token and included in refresh params."""
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"userPool": {"appClientId": "client-id"}, "region": "us-east-1"},
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.return_value = {"AuthenticationResult": {"AccessToken": "token"}}

        session_token = _make_jwt({"device_key": "us-east-1_device-abc-123"})

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client):
            client = CognitoClient("https://api.test.com")
            client.refresh_token("the-refresh-token", session_token=session_token)

        mock_cognito_client.initiate_auth.assert_called_once_with(
            AuthFlow="REFRESH_TOKEN_AUTH",
            AuthParameters={"REFRESH_TOKEN": "the-refresh-token", "DEVICE_KEY": "us-east-1_device-abc-123"},
            ClientId="client-id",
        )

    @responses.activate
    def test_refresh_token_without_device_key_in_session_token(self):
        """Test that refresh works without device_key when token doesn't contain one."""
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"userPool": {"appClientId": "client-id"}, "region": "us-east-1"},
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.return_value = {"AuthenticationResult": {"AccessToken": "token"}}

        session_token = _make_jwt({"sub": "user-123"})

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client):
            client = CognitoClient("https://api.test.com")
            client.refresh_token("the-refresh-token", session_token=session_token)

        mock_cognito_client.initiate_auth.assert_called_once_with(
            AuthFlow="REFRESH_TOKEN_AUTH",
            AuthParameters={"REFRESH_TOKEN": "the-refresh-token"},
            ClientId="client-id",
        )

    @responses.activate
    def test_refresh_token_without_session_token(self):
        """Test that refresh works without session_token (no device_key extraction attempted)."""
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"userPool": {"appClientId": "client-id"}, "region": "us-east-1"},
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.return_value = {"AuthenticationResult": {"AccessToken": "token"}}

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client):
            client = CognitoClient("https://api.test.com")
            client.refresh_token("the-refresh-token")

        mock_cognito_client.initiate_auth.assert_called_once_with(
            AuthFlow="REFRESH_TOKEN_AUTH",
            AuthParameters={"REFRESH_TOKEN": "the-refresh-token"},
            ClientId="client-id",
        )

    @responses.activate
    def test_cognito_config_cached_across_calls(self):
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"userPool": {"appClientId": "client-id"}, "region": "us-east-1"},
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.return_value = {"AuthenticationResult": {"AccessToken": "token"}}

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client):
            client = CognitoClient("https://api.test.com")
            client.refresh_token("refresh-token")
            client.refresh_token("refresh-token")

        # Config endpoint should only be called once despite two refresh calls
        assert len(responses.calls) == 1


class TestTokenAuthProvider:
    """Tests for TokenAuthProvider (production path: pre-supplied tokens)."""

    def test_get_session_token(self):
        provider = TokenAuthProvider.__new__(TokenAuthProvider)
        provider._session_token = "my-session-token"
        provider._refresh_token = "my-refresh-token"
        provider._cognito = Mock()

        assert provider.get_session_token() == "my-session-token"

    def test_refresh_updates_session_token(self):
        mock_cognito = Mock()
        mock_cognito.refresh_token.return_value = "new-access-token"

        provider = TokenAuthProvider.__new__(TokenAuthProvider)
        provider._session_token = "old-token"
        provider._refresh_token = "my-refresh-token"
        provider._cognito = mock_cognito

        result = provider.refresh()

        assert result == "new-access-token"
        assert provider.get_session_token() == "new-access-token"
        mock_cognito.refresh_token.assert_called_once_with("my-refresh-token", "old-token")

    def test_refresh_raises_without_refresh_token(self):
        provider = TokenAuthProvider.__new__(TokenAuthProvider)
        provider._session_token = "session-token"
        provider._refresh_token = None
        provider._cognito = Mock()

        with pytest.raises(RuntimeError, match="no refresh token"):
            provider.refresh()


class TestKeySecretAuthProvider:
    """Tests for KeySecretAuthProvider (local dev path: key/secret → tokens)."""

    @responses.activate
    def test_authenticates_eagerly_on_init(self):
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"userPool": {"appClientId": "client-id"}, "region": "us-east-1"},
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.return_value = {
            "AuthenticationResult": {
                "AccessToken": "initial-access-token",
                "RefreshToken": "initial-refresh-token",
            }
        }

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client):
            provider = KeySecretAuthProvider("https://api.test.com", "my-key", "my-secret")

        assert provider.get_session_token() == "initial-access-token"

    def test_refresh_uses_refresh_token(self):
        mock_cognito = Mock()
        mock_cognito.refresh_token.return_value = "refreshed-token"

        provider = KeySecretAuthProvider.__new__(KeySecretAuthProvider)
        provider._api_key = "key"
        provider._api_secret = "secret"
        provider._session_token = "old-token"
        provider._refresh_token = "my-refresh-token"
        provider._cognito = mock_cognito

        result = provider.refresh()

        assert result == "refreshed-token"
        assert provider.get_session_token() == "refreshed-token"
        mock_cognito.refresh_token.assert_called_once_with("my-refresh-token", "old-token")

    def test_refresh_re_authenticates_when_no_refresh_token(self):
        mock_cognito = Mock()
        mock_cognito.authenticate.return_value = ("new-access", "new-refresh")

        provider = KeySecretAuthProvider.__new__(KeySecretAuthProvider)
        provider._api_key = "key"
        provider._api_secret = "secret"
        provider._session_token = "old-token"
        provider._refresh_token = None
        provider._cognito = mock_cognito

        result = provider.refresh()

        assert result == "new-access"
        assert provider._refresh_token == "new-refresh"
        mock_cognito.authenticate.assert_called_once_with("key", "secret")
