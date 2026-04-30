import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Lock, Value
from typing import Optional

import backoff
import requests
from clients import (
    CreatedAsset,
    ImportClient,
    ImportFile,
    PackagesAssetsClient,
    PackagesClient,
    RangeChunk,
    TimeSeriesClient,
    TimeSeriesRangesClient,
    WorkflowClient,
)
from constants import TIME_SERIES_BINARY_FILE_EXTENSION, TIME_SERIES_METADATA_FILE_EXTENSION
from timeseries_channel import TimeSeriesChannel

from processor.asset_uploader import AssetUploader, relative_key_for_chunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

log = logging.getLogger()

# Pattern for files written by writer.py before channel-id substitution.
_CHANNEL_INDEX_PATTERN = re.compile(r"(channel-\d+)")

# Pattern to pull start/end microsecond timestamps out of a chunk filename:
#     channel-00007_1549968912000000_1549968926998750.bin.gz
#     N:channel:abc..._1549968912000000_1549968926998750.bin.gz
_CHUNK_TIMESTAMP_PATTERN = re.compile(r"_(\d+)_(\d+)\.bin\.gz$")


def import_timeseries(
    config,
    session_manager,
):
    """Top-level entry point. Dispatches to legacy or asset-aware flow.

    Replaces the previous single-flow implementation. Behavior is selected
    by `config.LEGACY_IMPORT_FLOW`:
      - false (default): create a viewer_asset, upload chunks via STS,
        register ranges in timeseries-service.
      - true: original Pennsieve import-manifest path (kept for
        rollback).
    """
    if config.LEGACY_IMPORT_FLOW:
        log.info("LEGACY_IMPORT_FLOW=true; using import-manifest flow")
        return import_timeseries_legacy(
            config.API_HOST,
            config.API_HOST2,
            session_manager,
            config.WORKFLOW_INSTANCE_ID,
            config.OUTPUT_DIR,
        )
    log.info("using viewer-asset flow")
    return import_timeseries_via_assets(
        config.API_HOST,
        config.API_HOST2,
        session_manager,
        config.WORKFLOW_INSTANCE_ID,
        config.OUTPUT_DIR,
        config.ASSET_NAME,
    )


# ---------------------------------------------------------------------------
# New flow: viewer_asset + STS upload + timeseries-service POST /ranges
# ---------------------------------------------------------------------------


def import_timeseries_via_assets(
    api_host,
    api2_host,
    session_manager,
    workflow_instance_id,
    file_directory,
    asset_name,
    asset_type: str = "timeseries",
):
    """Asset-aware ingest.

    1. Resolve target package from the workflow.
    2. Find or create a viewer_asset linked to all workflow packages.
    3. Create channels with viewer_asset_id set.
    4. Rename chunk files to use channel node ids (matching the legacy
       naming convention).
    5. Upload chunks to S3 using the asset's STS credentials.
    6. Register ranges via timeseries-service.
    7. Mark the asset 'active'.
    8. On any failure, delete the asset (cleanup-queue lambda purges S3).
    """
    timeseries_data_files, timeseries_channel_files = _collect_timeseries_files(file_directory)
    if not timeseries_data_files or not timeseries_channel_files:
        log.info("no time series channels or data")
        return None

    workflow_client = WorkflowClient(api2_host, session_manager)
    workflow_instance = workflow_client.get_workflow_instance(workflow_instance_id)

    packages_client = PackagesClient(api_host, session_manager)
    target_package_id = determine_target_package(packages_client, workflow_instance.package_ids)
    if not target_package_id:
        log.error(
            "dataset_id=%s could not determine target time series package",
            workflow_instance.dataset_id,
        )
        return None

    packages_client.set_timeseries_properties(target_package_id)
    log.info("updated package %s with time series properties", target_package_id)

    log.info(
        "dataset_id=%s target_package_id=%s asset_name=%s starting asset-flow ingest",
        workflow_instance.dataset_id,
        target_package_id,
        asset_name,
    )

    assets_client = PackagesAssetsClient(api2_host, session_manager)

    asset, upload_credentials = _find_or_create_asset(
        assets_client,
        dataset_id=workflow_instance.dataset_id,
        package_ids=workflow_instance.package_ids,
        asset_name=asset_name,
        asset_type=asset_type,
    )

    # upload_credentials is None when an already-active asset was found;
    # treat as a successful no-op so workflow re-runs are idempotent.
    if upload_credentials is None:
        return asset.id

    # Channels we create during this ingest (not reused from a prior run).
    # Tracked so the cleanup path can delete them if anything downstream
    # fails — otherwise they outlive the asset and break the next re-run.
    created_channel_node_ids: list[str] = []
    timeseries_client = TimeSeriesClient(api_host, session_manager)

    try:
        # Channels: create with viewer_asset_id set so timeseries-service
        # accepts the range registration. Reuse existing channels if name/
        # type/rate match.
        existing_channels = timeseries_client.get_package_channels(target_package_id)

        channels_by_index, created_channel_node_ids = _create_or_resolve_channels(
            timeseries_client,
            target_package_id,
            timeseries_channel_files,
            existing_channels,
            viewer_asset_id=asset.id,
        )
        if not channels_by_index:
            raise RuntimeError(
                "no channels were resolved from staged metadata files; refusing to mark asset active with empty data"
            )

        # Rename data files to use channel node ids in their basenames
        # (matching the legacy naming convention so timeseries.ranges.location
        # aligns with what streaming expects to fetch from S3).
        renamed_data_files = _rename_data_files_to_node_ids(timeseries_data_files, channels_by_index)
        if not renamed_data_files:
            raise RuntimeError(
                "no chunk files were resolved from the output directory; refusing to mark asset active with empty data"
            )

        # Upload to S3 using the STS creds returned by create_asset.
        uploader = AssetUploader(upload_credentials)
        uploads = uploader.upload_files(
            [(local_path, relative_key_for_chunk(local_path)) for local_path in renamed_data_files]
        )
        log.info("uploaded %d chunk files for asset %s", len(uploads), asset.id)

        # Register ranges. Build chunks from filenames + channel map.
        ranges_client = TimeSeriesRangesClient(api2_host, session_manager)
        chunks = _build_range_chunks(uploads, channels_by_index)
        result = ranges_client.create_ranges_batched(target_package_id, asset.id, chunks)
        log.info(
            "registered ranges for asset %s: requested=%d created=%d skipped=%d",
            asset.id,
            result.requested,
            result.created,
            result.skipped,
        )

        # Flip status to active. This MUST succeed — status='active' is
        # what makes re-runs idempotent. Swallowing it would leave the
        # asset in 'created', and the next run would treat it as stale,
        # delete + recreate, and lose the channel-asset link. Let any
        # failure propagate so the cleanup path runs and the next attempt
        # starts fresh.
        assets_client.update_asset(asset.id, dataset_id=workflow_instance.dataset_id, status="active")

    except Exception as e:
        log.error("asset-flow ingest failed for asset %s: %s", asset.id, e)
        # Delete channels we created BEFORE deleting the asset.
        # channels.viewer_asset_id has no FK to viewer_assets — deleting
        # the asset alone would orphan our newly-created channels with a
        # dangling viewer_asset_id, and the next run's
        # _create_or_resolve_channels would raise on the mismatch instead
        # of a clean restart. Reused channels (not in the created list)
        # are left untouched.
        for channel_node_id in created_channel_node_ids:
            try:
                timeseries_client.delete_channel(target_package_id, channel_node_id)
                log.info("deleted channel %s during cleanup", channel_node_id)
            except Exception as channel_cleanup_err:
                # Best-effort; keep going so the asset still gets cleaned
                # up. A leftover channel is recoverable in code; a
                # leftover asset row + S3 prefix is worse.
                log.error(
                    "failed to delete channel %s during cleanup: %s",
                    channel_node_id,
                    channel_cleanup_err,
                )

        # Now delete the asset row → triggers the S3 cleanup queue.
        try:
            assets_client.delete_asset(asset.id, dataset_id=workflow_instance.dataset_id)
            log.info("queued asset %s for cleanup", asset.id)
        except Exception as cleanup_err:
            log.error(
                "failed to delete failed asset %s — will need manual cleanup: %s",
                asset.id,
                cleanup_err,
            )
        raise

    return asset.id


def _find_or_create_asset(
    assets_client: PackagesAssetsClient,
    dataset_id: str,
    package_ids: list[str],
    asset_name: str,
    asset_type: str,
):
    """Look up an existing asset for this workflow; create one if absent.

    Lookup must use the same package set we link at creation time
    (workflow.package_ids). The aggregating "target package" walked to
    by determine_target_package — the parent collection in multi-package
    workflows — is *not* in viewer_asset_packages, so looking up by it
    would always return empty and we'd create duplicate assets on
    re-runs.

    Iterates the workflow packages and returns the first asset whose
    name + asset_type matches.

    Status-aware behavior:
      - Existing asset with status='active' → return (asset, None).
        upload_credentials being None signals "skip ingest, this asset is
        already done." Caller short-circuits.
      - Existing asset in any other state → assumed to be from a failed
        prior run. Delete it (triggers S3 cleanup queue) and create a
        fresh one.
      - No existing asset → create.

    Returns (asset, upload_credentials | None).
    """
    match = _find_asset_by_workflow_packages(assets_client, dataset_id, package_ids, asset_name, asset_type)

    if match is not None and match.status == "active":
        log.info(
            "asset %s already active for workflow packages %s; idempotent re-run, skipping ingest",
            match.id,
            package_ids,
        )
        return match, None

    if match is not None:
        log.info(
            "asset %s in status %r for workflow packages %s; assuming prior run failed, deleting and recreating",
            match.id,
            match.status,
            package_ids,
        )
        assets_client.delete_asset(match.id, dataset_id)

    log.info(
        "creating new asset %s/%s for dataset %s linking %d package(s)",
        asset_name,
        asset_type,
        dataset_id,
        len(package_ids),
    )
    created: CreatedAsset = assets_client.create_asset(
        dataset_id=dataset_id,
        package_ids=package_ids,
        name=asset_name,
        asset_type=asset_type,
    )
    return created.asset, created.upload_credentials


def _find_asset_by_workflow_packages(
    assets_client: PackagesAssetsClient,
    dataset_id: str,
    package_ids: list[str],
    asset_name: str,
    asset_type: str,
):
    """Search each workflow package for an asset matching name+type.

    Stops on first hit. Returns None if no package surfaces a match.
    """
    for package_id in package_ids:
        existing = assets_client.list_assets_for_package(dataset_id, package_id)
        match = next(
            (asset for asset in existing if asset.name == asset_name and asset.asset_type == asset_type),
            None,
        )
        if match is not None:
            return match
    return None


def _create_or_resolve_channels(
    timeseries_client: TimeSeriesClient,
    package_id: str,
    timeseries_channel_files: list[str],
    existing_channels: list[TimeSeriesChannel],
    viewer_asset_id: str,
) -> tuple[dict[str, TimeSeriesChannel], list[str]]:
    """For each channel metadata file, return a TimeSeriesChannel keyed by index.

    Reuses an existing channel when name/type/rate match, otherwise
    creates a new one with viewer_asset_id set. Mutates the returned
    channels' .index field so callers can map channel filenames to
    channel objects by index.

    Returns:
        (channels_by_index, created_channel_node_ids)
        - channels_by_index: dict {channel-index → TimeSeriesChannel}
        - created_channel_node_ids: list of node ids the *current* ingest
          created (i.e. did not reuse). The caller must delete these on
          ingest failure so that orphan channels don't outlive the asset.
    """
    channels: dict[str, TimeSeriesChannel] = {}
    created_channel_node_ids: list[str] = []
    for file_path in timeseries_channel_files:
        match = _CHANNEL_INDEX_PATTERN.search(os.path.basename(file_path))
        if match is None:
            raise RuntimeError(f"channel metadata filename does not match expected channel-NNNNN pattern: {file_path}")
        channel_index = match.group(1)

        with open(file_path, "r") as f:
            local_channel = TimeSeriesChannel.from_dict(json.load(f))
        local_channel.viewer_asset_id = viewer_asset_id

        existing = next((ec for ec in existing_channels if ec == local_channel), None)
        if existing is not None:
            if existing.viewer_asset_id != viewer_asset_id:
                raise RuntimeError(
                    f"channel {existing.id} ({existing.name}) on package "
                    f"{package_id} is linked to viewer_asset_id="
                    f"{existing.viewer_asset_id!r} but the current ingest "
                    f"expects {viewer_asset_id!r}. Resolve manually before "
                    "re-running."
                )
            log.info(
                "package_id=%s channel_id=%s reusing existing channel: %s",
                package_id,
                existing.id,
                existing.name,
            )
            channel = existing
        else:
            channel = timeseries_client.create_channel(package_id, local_channel)
            created_channel_node_ids.append(channel.id)
            log.info(
                "package_id=%s channel_id=%s created new channel: %s",
                package_id,
                channel.id,
                channel.name,
            )

        channel.index = channel_index
        channels[channel_index] = channel

    return channels, created_channel_node_ids


def _rename_data_files_to_node_ids(
    timeseries_data_files: list[str],
    channels_by_index: dict[str, TimeSeriesChannel],
) -> list[str]:
    """Rename chunk binary files in place: channel-{idx} → {channel.node_id}.

    Mirrors the legacy importer's substitution. Returns the list of new
    paths (original list is invalidated).
    """
    renamed: list[str] = []
    for file_path in timeseries_data_files:
        match = _CHANNEL_INDEX_PATTERN.search(os.path.basename(file_path))
        if match is None:
            raise RuntimeError(f"chunk filename does not match expected channel-NNNNN_*_*.bin.gz pattern: {file_path}")
        channel_index = match.group(1)
        channel = channels_by_index.get(channel_index)
        if channel is None:
            raise RuntimeError(
                f"chunk file {file_path} references channel index "
                f"{channel_index!r} for which no channel metadata was resolved; "
                "every chunk must map to a known channel"
            )

        new_basename = re.sub(_CHANNEL_INDEX_PATTERN, channel.id, os.path.basename(file_path))
        new_path = os.path.join(os.path.dirname(file_path), new_basename)
        os.rename(file_path, new_path)
        renamed.append(new_path)
    return renamed


def _build_range_chunks(
    uploads,
    channels_by_index: dict[str, TimeSeriesChannel],
) -> list[RangeChunk]:
    """Convert each upload result into a RangeChunk for POST /ranges.

    The basename embeds {channel.node_id}_{start_us}_{end_us}.bin.gz —
    we pull start/end from the filename and look up the channel by
    matching the node_id prefix.
    """
    # Build a lookup from channel.id (node id) to channel object.
    channels_by_node_id = {ch.id: ch for ch in channels_by_index.values()}
    chunks: list[RangeChunk] = []
    for upload in uploads:
        basename = upload.relative_key
        # Match either node-id-prefixed or index-prefixed names; we expect
        # the former post-rename, but the timestamp pattern works for both.
        ts_match = _CHUNK_TIMESTAMP_PATTERN.search(basename)
        if ts_match is None:
            raise ValueError(f"chunk filename does not contain start/end timestamps: {basename}")
        start = int(ts_match.group(1))
        end = int(ts_match.group(2))

        node_id = basename[: ts_match.start()]
        channel = channels_by_node_id.get(node_id)
        if channel is None:
            raise ValueError(f"chunk basename {basename} has no matching channel (node_id={node_id})")

        chunks.append(
            RangeChunk(
                channel_node_id=channel.id,
                start=start,
                end=end,
                s3_key=basename,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Legacy flow: kept for rollback via LEGACY_IMPORT_FLOW=true.
# ---------------------------------------------------------------------------


def import_timeseries_legacy(
    api_host,
    api2_host,
    session_manager,
    workflow_instance_id,
    file_directory,
):
    """Original pre-viewer-asset flow: import-manifest + S3 staging bucket.

    Preserved verbatim for rollback. New ingests should not hit this path.
    """
    timeseries_data_files, timeseries_channel_files = _collect_timeseries_files(file_directory)
    if not timeseries_channel_files or not timeseries_data_files:
        log.info("no time series channels or data")
        return None

    workflow_client = WorkflowClient(api2_host, session_manager)
    workflow_instance = workflow_client.get_workflow_instance(workflow_instance_id)

    packages_client = PackagesClient(api_host, session_manager)
    package_id = determine_target_package(packages_client, workflow_instance.package_ids)
    if not package_id:
        log.error("dataset_id=%s could not determine target time series package", workflow_instance.dataset_id)
        return None

    packages_client.set_timeseries_properties(package_id)
    log.info(f"updated package {package_id} with time series properties")

    log.info(f"dataset_id={workflow_instance.dataset_id} package_id={package_id} starting import of time series files")

    timeseries_client = TimeSeriesClient(api_host, session_manager)
    existing_channels = timeseries_client.get_package_channels(package_id)

    channels = {}
    for file_path in timeseries_channel_files:
        channel_index = _CHANNEL_INDEX_PATTERN.search(os.path.basename(file_path)).group(1)

        with open(file_path, "r") as file:
            local_channel = TimeSeriesChannel.from_dict(json.load(file))

        channel = next(
            (existing_channel for existing_channel in existing_channels if existing_channel == local_channel), None
        )
        if channel is not None:
            log.info(f"package_id={package_id} channel_id={channel.id} found existing package channel: {channel.name}")
        else:
            channel = timeseries_client.create_channel(package_id, local_channel)
            log.info(f"package_id={package_id} channel_id={channel.id} created new time series channel: {channel.name}")
        channel.index = channel_index
        channels[channel_index] = channel

    import_files = []
    for file_path in timeseries_data_files:
        channel_index = _CHANNEL_INDEX_PATTERN.search(os.path.basename(file_path)).group(1)
        channel = channels[channel_index]
        import_file = ImportFile(
            upload_key=uuid.uuid4(),
            file_path=re.sub(_CHANNEL_INDEX_PATTERN, channel.id, os.path.basename(file_path)),
            local_path=file_path,
        )
        import_files.append(import_file)

    import_client = ImportClient(api2_host, session_manager)
    import_id = import_client.create_batched(
        workflow_instance.id, workflow_instance.dataset_id, package_id, import_files
    )

    log.info(f"import_id={import_id} initialized import with {len(import_files)} time series data files for upload")

    upload_counter = Value("i", 0)
    upload_counter_lock = Lock()

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=5)
    def upload_timeseries_file(timeseries_file):
        try:
            with upload_counter_lock:
                upload_counter.value += 1
                log.info(
                    f"import_id={import_id} upload_key={timeseries_file.upload_key} uploading {upload_counter.value}/{len(import_files)} {timeseries_file.local_path}"
                )
            upload_url = import_client.get_presign_url(
                import_id, workflow_instance.dataset_id, timeseries_file.upload_key
            )
            with open(timeseries_file.local_path, "rb") as f:
                response = requests.put(upload_url, data=f)
                response.raise_for_status()
            return True
        except Exception as e:
            with upload_counter_lock:
                upload_counter.value -= 1
            log.error(
                f"import_id={import_id} upload_key={timeseries_file.upload_key} failed to upload {timeseries_file.local_path}: %s",
                e,
            )
            raise e

    successful_uploads = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        successful_uploads = list(executor.map(upload_timeseries_file, import_files))

    log.info(f"import_id={import_id} uploaded {upload_counter.value} time series files")

    assert sum(successful_uploads) == len(import_files), "Failed to upload all time series files"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _collect_timeseries_files(file_directory):
    timeseries_data_files = []
    timeseries_channel_files = []

    for root, _, files in os.walk(file_directory):
        for file in files:
            if file.endswith(TIME_SERIES_METADATA_FILE_EXTENSION):
                timeseries_channel_files.append(os.path.join(root, file))
            elif file.endswith(TIME_SERIES_BINARY_FILE_EXTENSION):
                timeseries_data_files.append(os.path.join(root, file))

    return timeseries_data_files, timeseries_channel_files


def determine_target_package(packages_client: PackagesClient, package_ids: list[str]) -> Optional[str]:
    """
    Determine which package should receive the time series data and properties.

    If there's only one package ID, use that package directly.
    If there are multiple package IDs, find the first one with 'N:package:' prefix
    and get its parent package ID.

    Args:
        packages_client: PackagesClient instance for API calls
        package_ids: List of package IDs from the workflow instance

    Returns:
        The package ID to update with properties, or None if unable to determine
    """
    if not package_ids:
        log.warning("No package IDs provided")
        return None

    if len(package_ids) == 1:
        log.info("Single package ID found, using it directly: %s", package_ids[0])
        return package_ids[0]

    first_package = None
    for package_id in package_ids:
        if package_id.startswith("N:package:"):
            first_package = package_id
            break

    if first_package is None:
        log.warning("No package ID with 'N:package:' prefix found in: %s", package_ids)
        return None

    log.info("Multiple package IDs found, getting parent of first package: %s", first_package)
    try:
        parent_id = packages_client.get_parent_package_id(first_package)
        log.info("Parent package ID: %s", parent_id)
        return parent_id
    except Exception as e:
        log.error("Failed to get parent package ID: %s", e)
        return None
