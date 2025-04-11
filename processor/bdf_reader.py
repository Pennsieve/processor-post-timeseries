import numpy as np
from timeseries_channel import TimeSeriesChannel
import logging

log = logging.getLogger()

class BDFElectricalSeriesReader:
    """
    BDF Reader : Wraps PyEDFLib

    Attributes:
        num_samples(int): Number of samples per-channel
        num_channels (int): Number of channels
        sampling_rate (int): Sampling rate (in Hz) either given by the raw file or calculated from given timestamp values
        timestamps (int): Timestamps (offset seconds from 0) either given by the raw file or calculated from given sampling rate
        channels (list[TimeSeriesChannel]): list of channels and their respective metadata
    """
    def __init__(self, edf, session_start_time):

        self.edf = edf
        self.session_start_time_secs = session_start_time.timestamp()
        self.num_channels = self.edf.signals_in_file
        self.num_samples = self.edf.getNSamples()[0] #Assume same sample coun across all channels
        self.sampling_rate = self.edf.getSampleFrequency(0) #Assume same frequency across all channels

        self._timestamps = np.linspace(
            0, self.num_samples / self.sampling_rate, self.num_samples, endpoint=False
        ) + self.session_start_time_secs

        self._channels = None

        self.scale_info = []
        for ch in range(self.num_channels):
            dmin = self.edf.getDigitalMinimum(ch)
            dmax = self.edf.getDigitalMaximum(ch)
            pmin = self.edf.getPhysicalMinimum(ch)
            pmax = self.edf.getPhysicalMaximum(ch)
            self.scale_info.append((dmin, dmax, pmin, pmax))

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def channels(self):
        if self._channels is None:
            self._channels = list()
            for ch in range(self.num_channels):
                label = self.edf.getLabel(ch)
                self._channels.append(
                    TimeSeriesChannel(
                        index=ch,
                        name=label,
                        rate=self.sampling_rate,
                        start=self.timestamps[0] * 1e6,
                        end=self.timestamps[-1] * 1e6,
                        group=""
                    )
                )
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

    def get_chunk(self, channel_index, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = self.num_samples

        raw = self.edf.readSignal(channel_index)[start:end]
        return raw
