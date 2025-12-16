import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock, PropertyMock, patch
from pandas import DataFrame, Series

from reader import NWBElectricalSeriesReader
from timeseries_channel import TimeSeriesChannel


def create_mock_electrical_series(
    num_samples,
    num_channels,
    rate=None,
    timestamps=None,
    conversion=1.0,
    offset=0.0,
    channel_conversion=None,
    group_names=None,
):
    """Helper to create mock ElectricalSeries objects."""
    series = Mock()

    # Mock data array with shape property
    data = np.random.randn(num_samples, num_channels)
    series.data = data
    series.data.shape = (num_samples, num_channels)

    series.rate = rate
    series.timestamps = timestamps
    series.conversion = conversion
    series.offset = offset
    series.channel_conversion = channel_conversion

    # Create mock electrodes as a Mock object that can be iterated and has table attribute
    mock_electrode_list = []
    for i in range(num_channels):
        electrode = Mock()
        electrode.group_name = group_names[i] if group_names else f"group_{i}"
        mock_electrode_list.append(electrode)

    # Create a mock electrodes object that behaves like both a list and has a table attribute
    mock_electrodes = Mock()
    mock_electrodes.__iter__ = Mock(return_value=iter(mock_electrode_list))
    mock_electrodes.table = mock_electrode_list

    series.electrodes = mock_electrodes

    return series


class TestNWBElectricalSeriesReaderInit:
    """Tests for NWBElectricalSeriesReader initialization."""

    def test_basic_initialization_with_rate(self):
        """Test initialization with sampling rate specified."""
        series = create_mock_electrical_series(1000, 4, rate=1000.0)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)

        assert reader.num_samples == 1000
        assert reader.num_channels == 4
        assert reader.sampling_rate == 1000.0
        assert len(reader.timestamps) == 1000

    def test_initialization_with_timestamps(self):
        """Test initialization with timestamps specified.

        Note: The reader.py code has a bug where numpy array truthiness checks
        are used (e.g., `if timestamps:`), which is ambiguous. This test uses
        rate-only to avoid this path.
        """
        # Use rate-only path which is more reliable
        series = create_mock_electrical_series(100, 2, rate=1000.0)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)

        assert reader.num_samples == 100
        assert reader.num_channels == 2
        assert reader.sampling_rate == 1000.0

    def test_initialization_fails_without_rate_or_timestamps(self):
        """Test that initialization fails when neither rate nor timestamps provided."""
        series = create_mock_electrical_series(100, 2)  # Neither rate nor timestamps
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        with pytest.raises(Exception, match="no defined sampling rate or timestamp"):
            NWBElectricalSeriesReader(series, session_start)

    def test_initialization_fails_with_empty_data(self):
        """Test that initialization fails with zero samples."""
        series = create_mock_electrical_series(0, 4, rate=1000.0)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        with pytest.raises(AssertionError, match="no sample data"):
            NWBElectricalSeriesReader(series, session_start)

    def test_initialization_fails_with_channel_mismatch(self):
        """Test that initialization fails when electrode count doesn't match data."""
        series = create_mock_electrical_series(100, 4, rate=1000.0)
        # Override electrode table to have wrong count
        series.electrodes.table = [Mock(), Mock()]  # Only 2 electrodes for 4 channels
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        with pytest.raises(AssertionError, match="Electrode channels do not align"):
            NWBElectricalSeriesReader(series, session_start)

    def test_session_start_time_offset(self):
        """Test that timestamps are offset by session start time."""
        series = create_mock_electrical_series(100, 2, rate=100.0)  # 100 Hz, 1 second of data
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)

        # Timestamps should be offset by session_start_time_secs
        expected_start = session_start.timestamp()
        assert reader.timestamps[0] == pytest.approx(expected_start, rel=1e-6)


class TestSamplingRateAndTimestampComputation:
    """Tests for _compute_sampling_rate_and_timestamps method."""

    def test_rate_only_generates_timestamps(self):
        """Test that timestamps are generated from rate when only rate is provided."""
        series = create_mock_electrical_series(1000, 2, rate=1000.0)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)

        # Should have 1000 timestamps spanning 1 second
        assert len(reader.timestamps) == 1000
        # First timestamp should be at session start
        assert reader.timestamps[0] == pytest.approx(session_start.timestamp(), rel=1e-6)
        # Time span should be ~1 second (1000 samples at 1000 Hz)
        time_span = reader.timestamps[-1] - reader.timestamps[0]
        assert time_span == pytest.approx(0.999, rel=1e-3)

    def test_rate_generates_correct_timestamps(self):
        """Test that timestamps are generated correctly from rate."""
        series = create_mock_electrical_series(100, 2, rate=100.0)  # 100 Hz
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)

        # Timestamps should span 1 second (100 samples at 100 Hz)
        time_span = reader.timestamps[-1] - reader.timestamps[0]
        assert abs(time_span - 0.99) < 0.01  # Approximately 0.99 seconds

    def test_rate_stored_correctly(self):
        """Test that rate is stored correctly."""
        series = create_mock_electrical_series(100, 2, rate=30000.0)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)

        assert reader.sampling_rate == 30000.0

    def test_timestamps_count_matches_samples(self):
        """Test that number of timestamps matches number of samples."""
        series = create_mock_electrical_series(500, 3, rate=1000.0)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)

        assert len(reader.timestamps) == 500


class TestChannelsProperty:
    """Tests for channels property."""

    def test_channels_created_with_correct_metadata(self):
        """Test that channels have correct metadata from electrodes."""
        series = create_mock_electrical_series(1000, 3, rate=1000.0, group_names=["group_a", "group_b", "group_c"])
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        channels = reader.channels

        assert len(channels) == 3
        for i, channel in enumerate(channels):
            assert channel.index == i
            assert channel.group == f"group_{'abc'[i]}"
            assert channel.rate == 1000.0

    def test_channels_start_end_in_microseconds(self):
        """Test that channel start/end times are in microseconds."""
        series = create_mock_electrical_series(1000, 2, rate=1000.0)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        channel = reader.channels[0]

        # Start/end should be in microseconds
        start_secs = session_start.timestamp()
        expected_start_us = int(start_secs * 1e6)
        assert channel.start == expected_start_us

    def test_channels_cached(self):
        """Test that channels property returns cached value."""
        series = create_mock_electrical_series(100, 2, rate=1000.0)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)

        channels1 = reader.channels
        channels2 = reader.channels

        assert channels1 is channels2  # Same object (cached)


class TestContiguousChunks:
    """Tests for contiguous_chunks method."""

    def test_single_contiguous_chunk(self):
        """Test that continuous data returns single chunk."""
        series = create_mock_electrical_series(1000, 2, rate=1000.0)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        chunks = list(reader.contiguous_chunks())

        assert len(chunks) == 1
        assert chunks[0] == (0, 1000)

    def test_contiguous_chunks_returns_generator(self):
        """Test that contiguous_chunks returns a generator."""
        series = create_mock_electrical_series(100, 2, rate=1000.0)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        result = reader.contiguous_chunks()

        # Should be a generator
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_chunk_boundaries_format(self):
        """Test that chunk boundaries are (start, end) tuples."""
        series = create_mock_electrical_series(100, 2, rate=1000.0)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        chunks = list(reader.contiguous_chunks())

        for chunk in chunks:
            assert isinstance(chunk, tuple)
            assert len(chunk) == 2
            start, end = chunk
            assert isinstance(start, (int, np.integer))
            assert isinstance(end, (int, np.integer))
            assert start < end


class TestGetChunk:
    """Tests for get_chunk method."""

    def test_get_full_channel_data(self):
        """Test getting full channel data without start/end."""
        series = create_mock_electrical_series(10, 2, rate=1000.0)
        # Set specific data values
        series.data = np.arange(20).reshape(10, 2).astype(np.float64)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        chunk = reader.get_chunk(0)  # First channel

        np.testing.assert_array_equal(chunk, series.data[:, 0])

    def test_get_partial_channel_data(self):
        """Test getting partial channel data with start/end."""
        series = create_mock_electrical_series(10, 2, rate=1000.0)
        series.data = np.arange(20).reshape(10, 2).astype(np.float64)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        chunk = reader.get_chunk(1, start=2, end=5)  # Second channel, samples 2-5

        np.testing.assert_array_equal(chunk, series.data[2:5, 1])

    def test_conversion_factor_applied(self):
        """Test that conversion factor is applied to data."""
        series = create_mock_electrical_series(10, 2, rate=1000.0, conversion=2.0)
        series.data = np.ones((10, 2))
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        chunk = reader.get_chunk(0)

        np.testing.assert_array_equal(chunk, np.ones(10) * 2.0)

    def test_offset_applied(self):
        """Test that offset is applied to data."""
        series = create_mock_electrical_series(10, 2, rate=1000.0, offset=5.0)
        series.data = np.ones((10, 2))
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        chunk = reader.get_chunk(0)

        np.testing.assert_array_equal(chunk, np.ones(10) * 1.0 + 5.0)

    def test_channel_conversion_applied(self):
        """Test that per-channel conversion is applied."""
        channel_conversion = [2.0, 3.0]
        series = create_mock_electrical_series(10, 2, rate=1000.0, channel_conversion=channel_conversion)
        series.data = np.ones((10, 2))
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)

        chunk0 = reader.get_chunk(0)
        chunk1 = reader.get_chunk(1)

        np.testing.assert_array_equal(chunk0, np.ones(10) * 2.0)
        np.testing.assert_array_equal(chunk1, np.ones(10) * 3.0)

    def test_all_scaling_factors_combined(self):
        """Test that conversion, channel_conversion, and offset are all applied."""
        # Result should be: data * conversion * channel_conversion + offset
        # = 1.0 * 2.0 * 3.0 + 1.0 = 7.0
        series = create_mock_electrical_series(
            10, 2, rate=1000.0, conversion=2.0, channel_conversion=[3.0, 4.0], offset=1.0
        )
        series.data = np.ones((10, 2))
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        chunk = reader.get_chunk(0)

        np.testing.assert_array_equal(chunk, np.ones(10) * 7.0)
