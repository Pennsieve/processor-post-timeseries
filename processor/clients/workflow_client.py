import requests
import json
import logging

log = logging.getLogger()

class WorkflowInstance:
    def __init__(self, id, application_id, dataset_id, package_ids, params):
        self.id = id
        self.application_id = application_id
        self.dataset_id = dataset_id
        self.package_ids = package_ids
        self.params = params

class WorkflowClient:
    def __init__(self, api_host):
        self.api_host = api_host

    # NOTE: workflows API currently returns a 200 response
    #       with an empty body even when a workflow instance does not exist
    def get_workflow_instance(self, session_token, workflow_instance_id):
        url = f"{self.api_host}/workflows/instances/{workflow_instance_id}"

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {session_token}"
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            workflow_instance = WorkflowInstance(
                id=data["uuid"],
                application_id=data["applicationId"],
                dataset_id=data["datasetId"],
                package_ids=data["packageIds"],
                params=data["params"]
            )

            return workflow_instance_id
        except requests.HTTPError as e:
            log.error(f"failed to fetch workflow instance with error: {e}")
            raise e
        except json.JSONDecodeError as e:
            log.error(f"failed to decode workflow instance response with error: {e}")
            raise e
        except Exception as e:
            log.error(f"failed to get workflow instance with error: {e}")
            raise e