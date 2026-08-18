import base64
import json
from unittest.mock import Mock, patch

import pytest
import responses
from clients.authentication_client import (
    CognitoClient,
    KeySecretAuthProvider,
    resolve_auth_provider,
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
    def test_authenticate_uses_token_pool_not_user_pool(self):
        """API key/secret are token-pool users, so the token pool's client must be used.

        Regression test. Authenticating an API key against the user pool's
        client fails as NotAuthorizedException "Incorrect username or
        password" — that username only exists in the token pool. The bug
        was invisible for months because the production path used
        SESSION_TOKEN (a genuine user-pool token) and only local dev
        exercised key/secret. Every other test here mocks a config with no
        tokenPool at all, so none of them can catch it.
        """
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={
                "region": "us-east-1",
                "userPool": {"appClientId": "user-pool-client"},
                "tokenPool": {"appClientId": "token-pool-client"},
            },
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.return_value = {
            "AuthenticationResult": {"AccessToken": "token", "RefreshToken": "refresh"}
        }

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client):
            client = CognitoClient("https://api.test.com")
            client.authenticate("my-api-key", "my-api-secret")

        _, kwargs = mock_cognito_client.initiate_auth.call_args
        assert kwargs["ClientId"] == "token-pool-client"

    @responses.activate
    def test_refresh_token_uses_token_pool_not_user_pool(self):
        """Refresh must target the same pool that minted the token."""
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={
                "region": "us-east-1",
                "userPool": {"appClientId": "user-pool-client"},
                "tokenPool": {"appClientId": "token-pool-client"},
            },
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.return_value = {
            "AuthenticationResult": {"AccessToken": "new-token"}
        }

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client):
            client = CognitoClient("https://api.test.com")
            client.refresh_token("some-refresh-token")

        _, kwargs = mock_cognito_client.initiate_auth.call_args
        assert kwargs["ClientId"] == "token-pool-client"

    @responses.activate
    def test_falls_back_to_user_pool_when_token_pool_absent(self):
        """Deployments that publish no tokenPool must still resolve a client."""
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"userPool": {"appClientId": "user-pool-client"}, "region": "us-east-1"},
            status=200,
        )

        client = CognitoClient("https://api.test.com")
        assert client._get_cognito_config()["app_client_id"] == "user-pool-client"

    @responses.activate
    def test_falls_back_to_user_pool_when_token_pool_client_empty(self):
        """An empty appClientId is 'not configured', not a usable client id.

        The identityPool in the real prod response carries exactly this
        shape ("appClientId": ""), so treating presence-of-key as
        presence-of-value would hand Cognito an empty ClientId.
        """
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={
                "region": "us-east-1",
                "userPool": {"appClientId": "user-pool-client"},
                "tokenPool": {"appClientId": ""},
            },
            status=200,
        )

        client = CognitoClient("https://api.test.com")
        assert client._get_cognito_config()["app_client_id"] == "user-pool-client"

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


class TestKeySecretAuthProvider:
    """Tests for KeySecretAuthProvider (the only supported auth path)."""

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

    def test_refresh_falls_back_to_key_secret_when_refresh_fails(self):
        """A revoked/expired refresh token must not fail the ingest — we still hold key/secret."""
        mock_cognito = Mock()
        mock_cognito.refresh_token.side_effect = Exception("NotAuthorizedException: Refresh Token has expired")
        mock_cognito.authenticate.return_value = ("reauth-access", "reauth-refresh")

        provider = KeySecretAuthProvider.__new__(KeySecretAuthProvider)
        provider._api_key = "key"
        provider._api_secret = "secret"
        provider._session_token = "old-token"
        provider._refresh_token = "expired-refresh-token"
        provider._cognito = mock_cognito

        result = provider.refresh()

        assert result == "reauth-access"
        assert provider.get_session_token() == "reauth-access"
        assert provider._refresh_token == "reauth-refresh"
        mock_cognito.refresh_token.assert_called_once_with("expired-refresh-token", "old-token")
        mock_cognito.authenticate.assert_called_once_with("key", "secret")

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


class TestResolveAuthProvider:
    """API key/secret is the only accepted credential — no session-token fallback."""

    def test_builds_key_secret_provider(self):
        with patch("clients.authentication_client.KeySecretAuthProvider") as mock_key_secret:
            provider = resolve_auth_provider("https://api.test.com", "my-key", "my-secret")

        mock_key_secret.assert_called_once_with("https://api.test.com", "my-key", "my-secret")
        assert provider is mock_key_secret.return_value

    @pytest.mark.parametrize(
        "api_key,api_secret",
        [(None, None), ("my-key", None), (None, "my-secret")],
        ids=["neither", "key-without-secret", "secret-without-key"],
    )
    def test_raises_without_complete_key_secret(self, api_key, api_secret):
        """A partial credential is unusable, and there is nothing left to fall back to."""
        with pytest.raises(RuntimeError, match="no authentication credentials provided"):
            resolve_auth_provider("https://api.test.com", api_key, api_secret)
