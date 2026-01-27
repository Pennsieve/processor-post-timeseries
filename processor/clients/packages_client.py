import json
import logging

import requests

from .base_client import BaseClient

log = logging.getLogger()


class PackagesClient(BaseClient):
    def __init__(self, api_host, session_manager):
        super().__init__(session_manager)

        self.api_host = api_host

    @BaseClient.retry_with_refresh
    def get_parent_package_id(self, package_id: str) -> str:
        """
        Get the parent package ID for a given package.

        Args:
            package_id: The package ID to query

        Returns:
            str: The parent node ID

        Raises:
            requests.HTTPError: If the API request fails
        """
        url = f"{self.api_host}/packages/{package_id}?includeAncestors=true&startAtEpoch=false&limit=100&offset=0"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.session_manager.session_token}",
        }

        try:
            log.info(f"Fetching parent package ID for package: {package_id}")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            package_info = response.json()
            parent_node_id = package_info["parent"]["content"]["nodeId"]
            return parent_node_id
        except requests.HTTPError as e:
            log.error(f"failed to get parent package ID for {package_id}: {e}")
            raise e
        except json.JSONDecodeError as e:
            log.error(f"failed to decode package response: {e}")
            raise e
        except Exception as e:
            log.error(f"failed to get parent package ID: {e}")
            raise e

    @BaseClient.retry_with_refresh
    def update_properties(self, package_id: str, properties: list[dict]) -> None:
        """
        Updates a package's properties on the Pennsieve API.

        Args:
            package_id: The package (node) ID
            properties: List of property dicts with keys: key, value, dataType, category, fixed, hidden
        """
        url = f"{self.api_host}/packages/{package_id}?updateStorage=true"

        payload = {"properties": properties}

        headers = {
            "accept": "*/*",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.session_manager.session_token}",
        }

        try:
            response = requests.put(url, json=payload, headers=headers)
            response.raise_for_status()
            return None
        except Exception as e:
            log.error(f"failed to update package {package_id} properties: {e}")
            raise e

    def set_timeseries_properties(self, package_id: str) -> None:
        """
        Sets the time series viewer properties on a package.

        Args:
            package_id: The package (node) ID
        """
        properties = [
            {
                "key": "subtype",
                "value": "pennsieve_timeseries",
                "dataType": "string",
                "category": "Viewer",
                "fixed": False,
                "hidden": True,
            },
            {
                "key": "icon",
                "value": "timeseries",
                "dataType": "string",
                "category": "Pennsieve",
                "fixed": False,
                "hidden": True,
            },
        ]
        return self.update_properties(package_id, properties)
