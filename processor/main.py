import os
import logging
import pyedflib
from datetime import datetime, timezone

from pynwb import NWBHDF5IO
from pynwb.ecephys import ElectricalSeries

from config import Config
from importer import import_timeseries
from writer import TimeSeriesChunkWriter
from bdf_reader import BDFElectricalSeriesReader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

log = logging.getLogger()

if __name__ == "__main__":
    config = Config()

    bytes_per_mb = pow(2, 20)
    bytes_per_sample = 8 # 64-bit floating point value
    chunk_size = int(config.CHUNK_SIZE_MB * bytes_per_mb / bytes_per_sample)

    input_files = [
        f.path
        for f in os.scandir(config.INPUT_DIR)
        if f.is_file() and os.path.splitext(f.name)[1].lower() == '.bdf'
    ]

    assert len(input_files) == 1, "BDF post processor only supports a single file as input"

    with pyedflib.EdfReader(input_files[0]) as edf:

        start_datetime = edf.getStartdatetime()

        # Stop timezone warning. Explicity set tz to UTC
        session_start_time = start_datetime.replace(tzinfo=timezone.utc)

        reader = BDFElectricalSeriesReader(edf, session_start_time)

        chunked_writer = TimeSeriesChunkWriter(session_start_time, config.OUTPUT_DIR, chunk_size)
        chunked_writer.write_electrical_series(reader)

    # import requires Pennsieve API access; when developing locally this is most often not required
    # note: this will be moved to a separated post-processor once the analysis pipeline is more
    # easily able to handle > 3 processors
    if config.IMPORTER_ENABLED:
        importer = import_timeseries(config.API_HOST, config.API_HOST2, config.API_KEY, config.API_SECRET, config.WORKFLOW_INSTANCE_ID, config.OUTPUT_DIR)
