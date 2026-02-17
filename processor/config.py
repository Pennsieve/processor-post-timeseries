import logging
import os
import uuid

log = logging.getLogger()


class Config:
    def __init__(self):
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "local")

        self.INPUT_DIR = os.getenv("INPUT_DIR")
        self.OUTPUT_DIR = os.getenv("OUTPUT_DIR")

        self.CHUNK_SIZE_MB = int(os.getenv("CHUNK_SIZE_MB", "1"))

        # continue to use INTEGRATION_ID environment variable until runner
        # has been converted to use  a different variable to represent the workflow instance ID
        self.WORKFLOW_INSTANCE_ID = os.getenv("INTEGRATION_ID", str(uuid.uuid4()))

        self.SESSION_TOKEN = os.getenv("SESSION_TOKEN")
        self.REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")
        self.API_HOST = os.getenv("PENNSIEVE_API_HOST", "https://api.pennsieve.net")
        self.API_HOST2 = os.getenv("PENNSIEVE_API_HOST2", "https://api2.pennsieve.net")

        # fall back to API key/secret auth for local development or when no session token is provided
        if self.SESSION_TOKEN is None or self.ENVIRONMENT == "local":
            api_key = os.getenv("PENNSIEVE_API_KEY")
            api_secret = os.getenv("PENNSIEVE_API_SECRET")
            if api_key and api_secret:
                from clients.authentication_client import AuthenticationClient

                log.info("no session token provided, authenticating with API key/secret")
                auth_client = AuthenticationClient(self.API_HOST)
                self.SESSION_TOKEN = auth_client.authenticate(api_key, api_secret)

        self.IMPORTER_ENABLED = getboolenv("IMPORTER_ENABLED", self.ENVIRONMENT != "local")


def getboolenv(key, default=False):
    return os.getenv(key, str(default)).lower() in ("true", "1")
