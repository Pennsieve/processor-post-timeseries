import gzip
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from constants import TIME_SERIES_BINARY_FILE_EXTENSION, TIME_SERIES_METADATA_FILE_EXTENSION
from reader import NWBElectricalSeriesReader
from utils import to_big_endian

log = logging.getLogger()


def _write_channel_chunk(chunk_data, start_time, end_time, channel_index, output_dir):
    """
    Write a single channel's chunk data to a gzipped file.

    Args:
        chunk_data: numpy array of sample data for the channel
        start_time: start timestamp in seconds
        end_time: end timestamp in seconds
        channel_index: channel index for filename
        output_dir: directory to write output file
    """
    formatted_data = to_big_endian(chunk_data.astype(np.float64))

    channel_index_str = "{index:05d}".format(index=channel_index)
    file_name = "channel-{}_{}_{}{}".format(
        channel_index_str, int(start_time * 1e6), int(end_time * 1e6), TIME_SERIES_BINARY_FILE_EXTENSION
    )
    file_path = os.path.join(output_dir, file_name)

    with gzip.open(file_path, mode="wb", compresslevel=0) as f:
        f.write(formatted_data)


class TimeSeriesChunkWriter:
    """
    Attributes:
        output_dir (str): path to output directory for chunked sample data binary files
        chunk_size (int): number of samples (rounded down) to include in a single chunked sample data binary file (pre-compression)
            each sample is represented as a 64-bit (8 byte) floating-point value
    """

    def __init__(self, session_start_time, output_dir, chunk_size):
        self.session_start_time = session_start_time
        self.output_dir = output_dir
        self.chunk_size = chunk_size

    def write_electrical_series(self, electrical_series, max_workers=None):
        """
        Chunks the sample data in two stages:
            1. Splits sample data into contiguous segments using the given or generated timestamp values
            2. Chunks each contiguous segment into the given chunk_size (number of samples to include per file)

        Writes each chunk to the given output directory.
        Channel processing is parallelized using ThreadPoolExecutor. Threads share memory
        (no serialization overhead) and the GIL is released during gzip compression and file I/O.

        Args:
            electrical_series: NWB ElectricalSeries object
            max_workers: Maximum number of threads (defaults to min(32, cpu_count + 4))
        """
        reader = NWBElectricalSeriesReader(electrical_series, self.session_start_time)
        num_channels = len(reader.channels)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for contiguous_start, contiguous_end in reader.contiguous_chunks():
                for chunk_start in range(contiguous_start, contiguous_end, self.chunk_size):
                    chunk_end = min(contiguous_end, chunk_start + self.chunk_size)

                    start_time = reader.get_timestamp(chunk_start)
                    end_time = reader.get_timestamp(chunk_end - 1)

                    channel_chunks = reader.get_chunk(chunk_start, chunk_end)

                    futures = [
                        executor.submit(
                            _write_channel_chunk,
                            channel_chunks[i],
                            start_time,
                            end_time,
                            reader.channels[i].index,
                            self.output_dir,
                        )
                        for i in range(num_channels)
                    ]

                    for future in futures:
                        future.result()

        for channel in reader.channels:
            self.write_channel(channel)

    def write_chunk(self, chunk, start_time, end_time, channel):
        """
        Writes the chunked sample data to a gzipped binary file.

        Args:
            chunk: numpy array of sample data
            start_time: start timestamp in seconds
            end_time: end timestamp in seconds
            channel: TimeSeriesChannel object
        """
        _write_channel_chunk(chunk, start_time, end_time, channel.index, self.output_dir)

    def write_channel(self, channel):
        file_name = f"channel-{channel.index:05d}{TIME_SERIES_METADATA_FILE_EXTENSION}"
        file_path = os.path.join(self.output_dir, file_name)

        with open(file_path, "w") as file:
            json.dump(channel.as_dict(), file)
