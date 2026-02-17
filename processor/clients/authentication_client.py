import json
import logging

import boto3
import requests

log = logging.getLogger()


class AuthenticationClient:
    def __init__(self, api_host):
        self.api_host = api_host
        self._cognito_config = None

    def _get_cognito_config(self):
        """Fetch and cache Cognito configuration from the Pennsieve API."""
        if self._cognito_config is not None:
            return self._cognito_config

        url = f"{self.api_host}/authentication/cognito-config"
        response = requests.get(url)
        response.raise_for_status()
        data = json.loads(response.content)

        self._cognito_config = {
            "app_client_id": data["tokenPool"]["appClientId"],
            "region": data["region"],
        }
        return self._cognito_config

    def _get_cognito_client(self):
        """Create a Cognito IDP client using cached config."""
        config = self._get_cognito_config()
        return boto3.client(
            "cognito-idp",
            region_name=config["region"],
            aws_access_key_id="",
            aws_secret_access_key="",
        )

    def authenticate(self, api_key, api_secret):
        url = f"{self.api_host}/authentication/cognito-config"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = json.loads(response.content)

            cognito_app_client_id = data["tokenPool"]["appClientId"]
            cognito_region = data["region"]

            cognito_idp_client = boto3.client(
                "cognito-idp",
                region_name=cognito_region,
                aws_access_key_id="",
                aws_secret_access_key="",
            )

            login_response = cognito_idp_client.initiate_auth(
                AuthFlow="USER_PASSWORD_AUTH",
                AuthParameters={"USERNAME": api_key, "PASSWORD": api_secret},
                ClientId=cognito_app_client_id,
            )

            auth_result = login_response["AuthenticationResult"]
            access_token = auth_result["AccessToken"]
            refresh_token = auth_result["RefreshToken"]
            return access_token, refresh_token
        except requests.HTTPError as e:
            log.error(f"failed to reach authentication server with error: {e}")
            raise e
        except json.JSONDecodeError as e:
            log.error(f"failed to decode authentication response with error: {e}")
            raise e
        except Exception as e:
            log.error(f"failed to authenticate with error: {e}")
            raise e

    def refresh(self, refresh_token):
        """Use a Cognito refresh token to obtain a new access token."""
        try:
            config = self._get_cognito_config()
            cognito_idp_client = self._get_cognito_client()

            response = cognito_idp_client.initiate_auth(
                AuthFlow="REFRESH_TOKEN_AUTH",
                AuthParameters={"REFRESH_TOKEN": refresh_token},
                ClientId=config["app_client_id"],
            )

            access_token = response["AuthenticationResult"]["AccessToken"]
            return access_token
        except requests.HTTPError as e:
            log.error(f"failed to reach authentication server with error: {e}")
            raise e
        except json.JSONDecodeError as e:
            log.error(f"failed to decode authentication response with error: {e}")
            raise e
        except Exception as e:
            log.error(f"failed to refresh session with error: {e}")
            raise e
