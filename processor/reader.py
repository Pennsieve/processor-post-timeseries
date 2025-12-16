import logging

import numpy as np
from pandas import DataFrame, Series
from timeseries_channel import TimeSeriesChannel
from utils import infer_sampling_rate

log = logging.getLogger()


class NWBElectricalSeriesReader:
    """
    Wrapper class around the NWB ElectricalSeries object.

    Provides helper functions and attributes for understanding the object's underlying sample and timeseries data.

    Timestamps are computed on-demand to avoid loading the entire array into memory.

    Attributes:
        electrical_series (ElectricalSeries): Raw acquired data from a NWB file
        num_samples(int): Number of samples per-channel
        num_channels (int): Number of channels
        sampling_rate (int): Sampling rate (in Hz) either given by the raw file or calculated from given timestamp values
        channels (list[TimeSeriesChannel]): list of channels and their respective metadata
    """

    def __init__(self, electrical_series, session_start_time):
        self.electrical_series = electrical_series
        self.session_start_time_secs = session_start_time.timestamp()
        self.num_samples, self.num_channels = self.electrical_series.data.shape

        assert self.num_samples > 0, "Electrical series has no sample data"
        assert (
            len(self.electrical_series.electrodes.table) == self.num_channels
        ), "Electrode channels do not align with data shape"

        self._sampling_rate = None
        self._has_explicit_timestamps = False
        self._compute_sampling_rate()

        self._channels = None

    def _compute_sampling_rate(self):
        """
        Computes and stores the sampling rate.

        Note: NWB specifies timestamps in seconds.

        Note: PyNWB disallows both sampling_rate and timestamps to be set on
        TimeSeries objects but its worth handling this case by validating the
        sampling_rate against the timestamps if this case does somehow appear.
        """
        if self.electrical_series.rate is None and self.electrical_series.timestamps is None:
            raise Exception("electrical series has no defined sampling rate or timestamp values")

        # if both the timestamps and rate properties are set on the electrical
        # series validate that the given rate is within a 2% margin of the
        # sampling rate calculated off of the given timestamps
        if self.electrical_series.rate and self.electrical_series.timestamps is not None:
            self._has_explicit_timestamps = True
            sampling_rate = self.electrical_series.rate

            sample_size = min(10000, self.num_samples)
            sample_timestamps = self.electrical_series.timestamps[:sample_size]
            inferred_sampling_rate = infer_sampling_rate(sample_timestamps)

            error = abs(inferred_sampling_rate - sampling_rate) * (1.0 / sampling_rate)
            if error > 0.02:
                raise Exception(
                    "Inferred rate from timestamps ({inferred_rate:.4f}) does not match given rate ({given_rate:.4f}).".format(
                        inferred_rate=inferred_sampling_rate, given_rate=sampling_rate
                    )
                )
            self._sampling_rate = sampling_rate

        # if only the rate is given, timestamps will be computed on-demand
        elif self.electrical_series.rate:
            self._sampling_rate = self.electrical_series.rate
            self._has_explicit_timestamps = False

        # if only the timestamps are given, calculate the sampling rate using a sample of timestamps
        elif self.electrical_series.timestamps is not None:
            self._has_explicit_timestamps = True
            sample_size = min(10000, self.num_samples)
            sample_timestamps = self.electrical_series.timestamps[:sample_size]
            self._sampling_rate = round(infer_sampling_rate(sample_timestamps))

    def get_timestamp(self, index):
        """
        Get timestamp for a single sample index.
        Computes on-demand when timestamps are not explicitly set.
        """
        timestamp = (
            float(self.electrical_series.timestamps[index])
            if self._has_explicit_timestamps
            else (index / self._sampling_rate)
        )
        return timestamp + self.session_start_time_secs

    def get_timestamps(self, start, end):
        """
        Get timestamps for a range of indices [start, end).
        Computes on-demand when timestamps are not explicitly set.
        Returns a numpy array.
        """
        timestamps = (
            np.array(self.electrical_series.timestamps[start:end])
            if self._has_explicit_timestamps
            else np.linspace(
                start / self._sampling_rate,
                end / self._sampling_rate,
                end - start,
                endpoint=False,
            )
        )
        return timestamps + self.session_start_time_secs

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @property
    def channels(self):
        if not self._channels:
            channels = []

            start_timestamp = self.get_timestamp(0)
            end_timestamp = self.get_timestamp(self.num_samples - 1)

            for index, electrode in enumerate(self.electrical_series.electrodes):
                name = ""
                if isinstance(electrode, DataFrame):
                    if "channel_name" in electrode:
                        name = electrode["channel_name"]
                    elif "label" in electrode:
                        name = electrode["label"]

                if isinstance(name, Series):
                    name = name.iloc[0]

                group_name = electrode.group_name
                if isinstance(group_name, Series):
                    group_name = group_name.iloc[0]

                channels.append(
                    # convert start / end to microseconds to maintain precision
                    TimeSeriesChannel(
                        index=index,
                        name=name,
                        rate=self.sampling_rate,
                        start=start_timestamp * 1e6,
                        end=end_timestamp * 1e6,
                        group=group_name,
                    )
                )

            self._channels = channels

        return self._channels

    def contiguous_chunks(self):
        """
        Returns a generator of the index ranges for contiguous segments in data.

        An index range is of the form [start, end).

        Boundaries are identified as follows:

            sampling_period = 1 / sampling_rate

            (timestamp_difference) > 2 * sampling_period
        """
        # if no explicit timestamps, data is continuous by definition
        if not self._has_explicit_timestamps:
            yield 0, self.num_samples
            return

        # process timestamps in batches to find gaps without loading all into memory
        gap_threshold = (1.0 / self.sampling_rate) * 2
        batch_size = 100000

        boundaries = [0]
        prev_timestamp = None

        for batch_start in range(0, self.num_samples, batch_size):
            batch_end = min(batch_start + batch_size, self.num_samples)
            batch_timestamps = self.electrical_series.timestamps[batch_start:batch_end]

            # check gap between batches
            if prev_timestamp is not None:
                if batch_timestamps[0] - prev_timestamp > gap_threshold:
                    boundaries.append(batch_start)

            # find gaps within batch
            diffs = np.diff(batch_timestamps)
            gap_indices = np.where(diffs > gap_threshold)[0]
            for gap_idx in gap_indices:
                boundaries.append(batch_start + gap_idx + 1)

            prev_timestamp = batch_timestamps[-1]

        boundaries.append(self.num_samples)

        for i in range(len(boundaries) - 1):
            yield boundaries[i], boundaries[i + 1]

    def get_chunk(self, channel_index, start=None, end=None):
        """
        Returns a chunk of sample data from the electrical series
        for the given channel (index)

        If start and end are not specified the entire channel's data is read into memory.

        The sample data is scaled by the conversion and offset factors
        set in the electrical series.
        """
        scale_factor = self.electrical_series.conversion

        if self.electrical_series.channel_conversion:
            scale_factor *= self.electrical_series.channel_conversion[channel_index]

        return self.electrical_series.data[start:end, channel_index] * scale_factor + self.electrical_series.offset
