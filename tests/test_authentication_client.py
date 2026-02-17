import json
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


class TestAuthenticationClientAuthenticate:
    """Tests for AuthenticationClient.authenticate method."""

    @responses.activate
    def test_authenticate_success(self):
        """Test successful authentication flow."""
        # Mock cognito config response
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"tokenPool": {"appClientId": "test-client-id"}, "region": "us-east-1"},
            status=200,
        )

        # Mock boto3 cognito client
        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.return_value = {
            "AuthenticationResult": {"AccessToken": "test-access-token-12345"}
        }

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client):
            client = AuthenticationClient("https://api.test.com")
            token = client.authenticate("api-key", "api-secret")

        assert token == "test-access-token-12345"

    @responses.activate
    def test_authenticate_calls_cognito_with_correct_params(self):
        """Test that Cognito is called with correct parameters."""
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"tokenPool": {"appClientId": "my-app-client-id"}, "region": "us-west-2"},
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.return_value = {"AuthenticationResult": {"AccessToken": "token"}}

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client) as mock_boto:
            client = AuthenticationClient("https://api.test.com")
            client.authenticate("my-api-key", "my-api-secret")

        # Check boto3 client was created with correct parameters
        mock_boto.assert_called_once_with(
            "cognito-idp", region_name="us-west-2", aws_access_key_id="", aws_secret_access_key=""
        )

        # Check initiate_auth was called with correct parameters
        mock_cognito_client.initiate_auth.assert_called_once_with(
            AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters={"USERNAME": "my-api-key", "PASSWORD": "my-api-secret"},
            ClientId="my-app-client-id",
        )

    @responses.activate
    def test_authenticate_raises_on_config_http_error(self):
        """Test that HTTP errors from config endpoint are raised."""
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"error": "Server error"},
            status=500,
        )

        client = AuthenticationClient("https://api.test.com")

        with pytest.raises(Exception):
            client.authenticate("key", "secret")

    @responses.activate
    def test_authenticate_raises_on_invalid_json(self):
        """Test that invalid JSON response raises error."""
        responses.add(
            responses.GET, "https://api.test.com/authentication/cognito-config", body="not valid json", status=200
        )

        client = AuthenticationClient("https://api.test.com")

        with pytest.raises(json.JSONDecodeError):
            client.authenticate("key", "secret")

    @responses.activate
    def test_authenticate_raises_on_cognito_error(self):
        """Test that Cognito errors are raised."""
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"tokenPool": {"appClientId": "client-id"}, "region": "us-east-1"},
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.side_effect = Exception("Cognito auth failed")

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client):
            client = AuthenticationClient("https://api.test.com")

            with pytest.raises(Exception, match="Cognito auth failed"):
                client.authenticate("key", "secret")

    @responses.activate
    def test_authenticate_extracts_access_token(self):
        """Test that access token is correctly extracted from response."""
        responses.add(
            responses.GET,
            "https://api.test.com/authentication/cognito-config",
            json={"tokenPool": {"appClientId": "client-id"}, "region": "us-east-1"},
            status=200,
        )

        mock_cognito_client = Mock()
        mock_cognito_client.initiate_auth.return_value = {
            "AuthenticationResult": {
                "AccessToken": "the-access-token",
                "RefreshToken": "refresh-token",
                "IdToken": "id-token",
                "ExpiresIn": 3600,
            }
        }

        with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client):
            client = AuthenticationClient("https://api.test.com")
            token = client.authenticate("key", "secret")

        # Should return only the access token
        assert token == "the-access-token"


class TestAuthenticationClientEdgeCases:
    """Edge case tests for AuthenticationClient."""

    @responses.activate
    def test_authenticate_with_empty_credentials(self):
        """Test authentication with empty credentials."""
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
            # Empty credentials should still be passed to Cognito
            client.authenticate("", "")

        mock_cognito_client.initiate_auth.assert_called_once()
        call_args = mock_cognito_client.initiate_auth.call_args
        assert call_args[1]["AuthParameters"]["USERNAME"] == ""
        assert call_args[1]["AuthParameters"]["PASSWORD"] == ""

    @responses.activate
    def test_authenticate_with_different_regions(self):
        """Test authentication with different AWS regions."""
        for region in ["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-1"]:
            responses.reset()
            responses.add(
                responses.GET,
                "https://api.test.com/authentication/cognito-config",
                json={"tokenPool": {"appClientId": "client-id"}, "region": region},
                status=200,
            )

            mock_cognito_client = Mock()
            mock_cognito_client.initiate_auth.return_value = {"AuthenticationResult": {"AccessToken": "token"}}

            with patch("clients.authentication_client.boto3.client", return_value=mock_cognito_client) as mock_boto:
                client = AuthenticationClient("https://api.test.com")
                client.authenticate("key", "secret")

            # Verify correct region was used
            mock_boto.assert_called_with(
                "cognito-idp", region_name=region, aws_access_key_id="", aws_secret_access_key=""
            )


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
