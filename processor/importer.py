import backoff
import logging
import os
import json
import re
import requests
import uuid

from clients import AuthenticationClient
from clients import ImportClient, ImportFile
from clients import TimeSeriesClient
from clients import WorkflowClient, WorkflowInstance

from constants import TIME_SERIES_BINARY_FILE_EXTENSION, TIME_SERIES_METADATA_FILE_EXTENSION

from timeseries_channel import TimeSeriesChannel

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value, Lock

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

log = logging.getLogger()

"""
Uses the Pennsieve API to initialize and upload time series files
for import into Pennsieve data ecosystem.

# note: this will be moved to a separated post-processor once the analysis pipeline is more
# easily able to handle > 3 processors
"""

def import_timeseries(api_host, api2_host, api_key, api_secret, workflow_instance_id, file_directory):
    # gather all the time series files from the output directory
    timeseries_data_files = []
    timeseries_channel_files = []

    for root, _, files in os.walk(file_directory):
        for file in files:
            if file.endswith(TIME_SERIES_METADATA_FILE_EXTENSION):
                timeseries_channel_files.append(os.path.join(root, file))
            elif file.endswith(TIME_SERIES_BINARY_FILE_EXTENSION):
                timeseries_data_files.append(os.path.join(root, file))

    if len(timeseries_channel_files) == 0 or len(timeseries_data_files) == 0:
        log.info("no time series channels or data")
        return None

    # authentication against the Pennsieve API
    authorization_client = AuthenticationClient(api_host)
    session_token = authorization_client.authenticate(api_key, api_secret)

    # fetch workflow instance for parameters (dataset_id, package_id, etc.)
    workflow_client = WorkflowClient(api2_host)
    # workflow_instance  = workflow_client.get_workflow_instance(session_token, workflow_instance_id)
    workflow_instance = WorkflowInstance(
        id = workflow_instance_id,
        application_id = workflow_instance_id,
        dataset_id = "N:dataset:3c10091e-4ef8-45ac-b3ae-4497eb34c7dc",
        package_ids = ["N:package:ee5534ca-8c32-405b-bcb1-cab2c55c59bd"],
        params = []
    )

    # constraint until we implement (upstream) performing imports over directories
    # and specifying how to group time series files together into an imported package
    assert len(workflow_instance.package_ids) == 1, "NWB post processor only supports a single package for import"
    package_id = workflow_instance.package_ids[0]

    log.info(f"dataset_id={workflow_instance.dataset_id} package_id={package_id} starting import of time series files")

    # used to strip the channel index (intra-processor channel identifier) off both data and metadata time series files
    channel_index_pattern = re.compile(r"(channel-\d+)")

    timeseries_client = TimeSeriesClient(api_host)
    existing_channels = timeseries_client.get_package_channels(session_token, package_id)

    channels = {}
    for file_path in timeseries_channel_files:
        channel_index = channel_index_pattern.search(os.path.basename(file_path)).group(1)
        with open(file_path, 'r') as file:
            ch = TimeSeriesChannel.from_dict(json.load(file))
            log.info(f"channel read with start={ch.start} end={ch.end}")
            channel = next((channel for channel in existing_channels if channel == ch), None)
            if channel is not None:
                log.info(f"package_id={package_id} channel_id={channel.id} found existing package channel: {channel.name}")
            else:
                channel = timeseries_client.create_channel(session_token, package_id, ch)
                log.info(f"package_id={package_id} channel_id={channel.id} created new time series channel: {channel.name}")
            channel.index = channel_index
            channels[channel_index] = channel

    # (to match the currently existing pattern)
    # replace the prefix on the time series binary data chunk file name with the channel node ID e.g.
    # channel-00000_1549968912000000_1549968926998750.bin.gz
    #  => N:channel:c957d73f-84ca-41d9-83b0-d23c2000a6e6_1549968912000000_1549968926998750.bin.gz
    import_files = []
    for file_path in timeseries_data_files:
        channel_index = channel_index_pattern.search(os.path.basename(file_path)).group(1)
        channel = channels[channel_index]
        import_file = ImportFile(
            upload_key=uuid.uuid4(),
            file_path=re.sub(channel_index_pattern, channel.id, os.path.basename(file_path)),
            local_path = file_path
        )
        import_files.append(import_file)

    # initialize import
    import_client = ImportClient(api2_host)
    import_id = import_client.create(session_token, workflow_instance.id, workflow_instance.dataset_id, package_id, import_files)

    log.info(f"import_id={import_id} initialized import with {len(import_files)} time series data files for upload")

    # track time series file upload count
    upload_counter = Value('i', 0)
    upload_counter_lock = Lock()

    # upload time series files to Pennsieve S3 import bucket
    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_tries=5
    )
    def upload_timeseries_file(timeseries_file):
        try:
            with upload_counter_lock:
                upload_counter.value += 1
                log.info(f"import_id={import_id} upload_key={timeseries_file.upload_key} uploading {upload_counter.value}/{len(import_files)} {timeseries_file.local_path}")
            upload_url = import_client.get_presign_url(session_token, import_id, workflow_instance.dataset_id, timeseries_file.upload_key)
            with open(timeseries_file.local_path, 'rb') as f:
                response = requests.put(upload_url, data=f)
                response.raise_for_status()  # raise an error if the request failed
            return True
        except Exception as e:
            with upload_counter_lock:
                upload_counter.value -= 1
            log.error(f"import_id={import_id} upload_key={timeseries_file.upload_key} failed to upload {timeseries_file.local_path}: %s", e)
            raise e

    successful_uploads = list()
    with ThreadPoolExecutor(max_workers=4) as executor:
        # wrapping in a list forces the executor to wait for all threads to finish uploading time series files
        successful_uploads = list(executor.map(upload_timeseries_file, import_files))

    log.info(f"import_id={import_id} uploaded {upload_counter.value} time series files")

    assert sum(successful_uploads) == len(import_files), "Failed to upload all time series files"
