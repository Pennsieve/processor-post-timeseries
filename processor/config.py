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
        self.API_KEY = os.getenv("PENNSIEVE_API_KEY")
        self.API_SECRET = os.getenv("PENNSIEVE_API_SECRET")
        self.API_HOST = os.getenv("PENNSIEVE_API_HOST", "https://api.pennsieve.net")
        self.API_HOST2 = os.getenv("PENNSIEVE_API_HOST2", "https://api2.pennsieve.net")

        self.IMPORTER_ENABLED = getboolenv("IMPORTER_ENABLED", self.ENVIRONMENT != "local")

        # Per-converter pipeline name embedded in the viewer_asset row,
        # e.g. "mef-asset", "edf-asset". Each deployment of post-timeseries
        # sets its own value.
        self.ASSET_NAME = os.getenv("ASSET_NAME", "timeseries-asset")

        # When true, fall back to the pre-viewer-asset flow that uploads
        # via the Pennsieve import-manifest API. Default false → new flow
        # that creates a viewer_asset, uploads via packages-service STS
        # creds, and registers ranges via timeseries-service.
        self.LEGACY_IMPORT_FLOW = getboolenv("LEGACY_IMPORT_FLOW", False)


def getboolenv(key, default=False):
    return os.getenv(key, str(default)).lower() in ("true", "1")
