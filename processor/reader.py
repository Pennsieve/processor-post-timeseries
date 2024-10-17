import logging
import numpy as np

from pandas import DataFrame, Series
from pynwb.ecephys import ElectricalSeries
from timeseries_channel import TimeSeriesChannel
from utils import infer_sampling_rate

log = logging.getLogger()

class NWBElectricalSeriesReader:
    """
    Wrapper class around the NWB ElectricalSeries object.

    Provides helper functions and attributes for understanding the object's underlying sample and timeseries data

    Attributes:
        electrical_series (ElectricalSeries): Raw acquired data from a NWB file
        num_samples(int): Number of samples per-channel
        num_channels (int): Number of channels
        sampling_rate (int): Sampling rate (in Hz) either given by the raw file or calculated from given timestamp values
        timestamps (int): Timestamps (offset seconds from 0) either given by the raw file or calculated from given sampling rate
        channels (list[TimeSeriesChannel]): list of channels and their respective metadata
    """

    def __init__(self, electrical_series, session_start_time):
        self.electrical_series = electrical_series
        self.session_start_time_secs = session_start_time.timestamp()
        self.num_samples, self.num_channels = self.electrical_series.data.shape

        assert self.num_samples > 0, 'Electrical series has no sample data'
        assert len(self.electrical_series.electrodes.table) == self.num_channels, 'Electrode channels do not align with data shape'

        self._sampling_rate = None
        self._timestamps = None
        self._compute_sampling_rate_and_timestamps()

        assert self.num_samples == len(self.timestamps), "Differing number of sample and timestamp value"

        self._channels = None


    def _compute_sampling_rate_and_timestamps(self):
        """
        Sets the sampling_rate and timestamps properties on the reader object.

        Computes either the sampling_rate or the timestamps given the other
        is provided in the NWB file.

        Note: NWB specifies timestamps in seconds

        Note: PyNWB disallows both sampling_rate and timestamps to be set on
        TimeSeries objects but its worth handling this case by validating the
        sampling_rate against the timestamps if this case does somehow appear
        """
        if self.electrical_series.rate is None and self.electrical_series.timestamps is None:
            raise Exception("electrical series has no defined sampling rate or timestamp values")

        # if both the timestamps and rate properties are set on the electrical series
        # validate that the given rate is within a 2% margin of the rate calculated
        # off of the given timestamps
        if self.electrical_series.rate and self.electrical_series.timestamps:
            # validate sampling rate against timestamps
            timestamps = self.electrical_series.timestamps
            sampling_rate = self.electrical_series.rate

            inferred_sampling_rate = infer_sampling_rate(timestamps)
            error = abs(inferred_sampling_rate-sampling_rate) * (1.0 / sampling_rate)
            if error > 0.02:
                # error is greater than 2%
                raise Exception("Inferred rate from timestamps ({inferred_rate:.4f}) does not match given rate ({given_rate:.4f})." \
                        .format(inferred_rate=inferred_sampling_rate, given_rate=sampling_rate))

        # if only the rate is given, calculate the timestamps for the samples
        # using the given number of samples (size of the data)
        if self.electrical_series.rate:
            sampling_rate = self.electrical_series.rate
            timestamps = np.linspace(0, self.num_samples / sampling_rate, self.num_samples, endpoint = False)

        # if only the timestamps are given, calculate the sampling rate using the timestamps
        if self.electrical_series.timestamps:
            timestamps = self.electrical_series.timestamps
            sampling_rate = round(infer_sampling_rate(self._timestamps))

        self._sampling_rate = sampling_rate
        self._timestamps = timestamps + self.session_start_time_secs

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @property
    def channels(self):
        if not self._channels:
            channels = list()
            for index, electrode in enumerate(self.electrical_series.electrodes):
                name = ""
                if isinstance(electrode, DataFrame):
                    if 'channel_name' in electrode:
                        name = electrode['channel_name']
                    elif 'label' in electrode:
                        name = electrode['label']

                if isinstance(name, Series):
                    name = name.iloc[0]

                group_name = electrode.group_name
                if isinstance(group_name, Series):
                    group_name = group_name.iloc[0]

                channels.append(
                        # convert start / end to microseconds to maintain precision
                        TimeSeriesChannel(
                            index = index,
                            name = name,
                            rate = self.sampling_rate,
                            start = self.timestamps[0] * 1e6 , # safe access gaurenteed by initialization assertions
                            end = self.timestamps[-1] * 1e6,
                            group = group_name
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
        gap_threshold = (1.0 / self.sampling_rate) * 2

        boundaries = np.concatenate(
            ([0], (np.diff(self.timestamps) > gap_threshold).nonzero()[0] + 1, [len(self.timestamps)]))

        for i in np.arange(len(boundaries)-1):
            yield boundaries[i], boundaries[i + 1]

    def get_chunk(self, channel_index, start = None, end = None):
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
