import pytest
import json
import responses
from unittest.mock import Mock, patch, MagicMock

from clients.authentication_client import AuthenticationClient


class TestAuthenticationClientInit:
    """Tests for AuthenticationClient initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        client = AuthenticationClient("https://api.test.com")
        assert client.api_host == "https://api.test.com"


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
