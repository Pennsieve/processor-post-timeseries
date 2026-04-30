import json
import logging

import requests

from processor.timeseries_channel import TimeSeriesChannel

from .base_client import BaseClient

log = logging.getLogger()


class TimeSeriesClient(BaseClient):
    def __init__(self, api_host, session_manager):
        super().__init__(session_manager)

        self.api_host = api_host

    @BaseClient.retry_with_refresh
    def create_channel(self, package_id, channel):
        url = f"{self.api_host}/timeseries/{package_id}/channels"

        headers = {"Content-type": "application/json", "Authorization": f"Bearer {self.session_manager.session_token}"}

        body = channel.as_dict()
        body["channelType"] = body.pop("type")

        try:
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            created_channel = TimeSeriesChannel.from_dict(data["content"], data["properties"])
            created_channel.index = channel.index

            return created_channel
        except requests.HTTPError as e:
            log.error("failed to create time series channel: %s", e)
            raise e
        except json.JSONDecodeError as e:
            log.error("failed to decode time series channel response: %s", e)
            raise e
        except Exception as e:
            log.error("failed to create time series channel: %s", e)
            raise e

    @BaseClient.retry_with_refresh
    def delete_channel(self, package_id, channel_id):
        """Delete a single channel by node id. Best-effort; logs and
        re-raises on failure so callers can decide how to handle it.

        Used during asset-flow ingest cleanup: when an ingest fails
        after channels were created, we delete those channels (along
        with deleting the asset itself) so the next run starts clean
        rather than tripping over orphaned channels still pointing at
        the deleted asset's id.
        """
        url = f"{self.api_host}/timeseries/{package_id}/channels/{channel_id}"
        headers = {
            "Content-type": "application/json",
            "Authorization": f"Bearer {self.session_manager.session_token}",
        }

        try:
            response = requests.delete(url, headers=headers)
            response.raise_for_status()
        except requests.HTTPError as e:
            log.error(
                "failed to delete channel %s on package %s: %s",
                channel_id,
                package_id,
                e,
            )
            raise

    @BaseClient.retry_with_refresh
    def get_package_channels(self, package_id):
        url = f"{self.api_host}/timeseries/{package_id}/channels"

        headers = {"Content-type": "application/json", "Authorization": f"Bearer {self.session_manager.session_token}"}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            channels = []
            for item in data:
                content = item["content"]
                properties = item["properties"]

                channel = TimeSeriesChannel.from_dict(content, properties)
                channels.append(channel)

            return channels
        except requests.HTTPError as e:
            log.error("failed to fetch time series channels for package %s: %s", package_id, e)
            raise e
        except json.JSONDecodeError as e:
            log.error("failed to decode time series package channels response: %s", e)
            raise e
        except Exception as e:
            log.error("failed to time series channels: %s", e)
            raise e
