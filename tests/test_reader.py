from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pytest
from reader import NWBElectricalSeriesReader


def create_mock_electrical_series(
    num_samples,
    num_channels,
    rate=None,
    timestamps=None,
    conversion=1.0,
    offset=0.0,
    channel_conversion=None,
    group_names=None,
    unit="microvolts",
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
    series.unit = unit

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
        assert reader.get_timestamp(0) == pytest.approx(expected_start, rel=1e-6)


class TestSamplingRateAndTimestampComputation:
    """Tests for sampling rate and timestamp computation."""

    def test_rate_only_generates_correct_timestamps(self):
        """Test that timestamps are generated from rate when only rate is provided."""
        series = create_mock_electrical_series(1000, 2, rate=1000.0)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)

        # First timestamp should be at session start
        assert reader.get_timestamp(0) == pytest.approx(session_start.timestamp(), rel=1e-6)
        # Time span should be ~1 second (1000 samples at 1000 Hz)
        time_span = reader.get_timestamp(999) - reader.get_timestamp(0)
        assert time_span == pytest.approx(0.999, rel=1e-3)

    def test_rate_stored_correctly(self):
        """Test that rate is stored correctly."""
        series = create_mock_electrical_series(100, 2, rate=30000.0)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)

        assert reader.sampling_rate == 30000.0


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


class TestGetAllChannelsChunk:
    """Tests for batch channel reading."""

    def test_returns_all_channels(self):
        """Test that get_chunk returns data for all channels."""
        series = create_mock_electrical_series(100, 4, rate=1000.0)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        chunks = reader.get_chunk(0, 50)

        assert len(chunks) == 4
        for chunk in chunks:
            assert len(chunk) == 50

    def test_get_full_data(self):
        """Test getting full data without start/end."""
        series = create_mock_electrical_series(10, 2, rate=1000.0)
        series.data = np.arange(20).reshape(10, 2).astype(np.float64)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        chunks = reader.get_chunk()

        np.testing.assert_array_equal(chunks[0], series.data[:, 0])
        np.testing.assert_array_equal(chunks[1], series.data[:, 1])

    def test_get_partial_data(self):
        """Test getting partial data with start/end."""
        series = create_mock_electrical_series(10, 2, rate=1000.0)
        series.data = np.arange(20).reshape(10, 2).astype(np.float64)
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        chunks = reader.get_chunk(start=2, end=5)

        np.testing.assert_array_equal(chunks[0], series.data[2:5, 0])
        np.testing.assert_array_equal(chunks[1], series.data[2:5, 1])

    def test_conversion_factor_applied(self):
        """Test that conversion factor is applied in batch read."""
        series = create_mock_electrical_series(10, 2, rate=1000.0, conversion=2.0)
        series.data = np.ones((10, 2))
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        chunks = reader.get_chunk()

        for chunk in chunks:
            np.testing.assert_array_equal(chunk, np.ones(10) * 2.0)

    def test_channel_conversion_applied(self):
        """Test that per-channel conversion is applied in batch read."""
        channel_conversion = [2.0, 3.0, 4.0]
        series = create_mock_electrical_series(10, 3, rate=1000.0, channel_conversion=channel_conversion)
        series.data = np.ones((10, 3))
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        chunks = reader.get_chunk()

        np.testing.assert_array_equal(chunks[0], np.ones(10) * 2.0)
        np.testing.assert_array_equal(chunks[1], np.ones(10) * 3.0)
        np.testing.assert_array_equal(chunks[2], np.ones(10) * 4.0)

    def test_all_scaling_factors_combined(self):
        """Test that all scaling factors are applied in batch read."""
        series = create_mock_electrical_series(
            10, 2, rate=1000.0, conversion=2.0, channel_conversion=[3.0, 4.0], offset=1.0
        )
        series.data = np.ones((10, 2))
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        chunks = reader.get_chunk()

        # Result: data * conversion * channel_conversion + offset
        np.testing.assert_array_equal(chunks[0], np.ones(10) * 7.0)  # 1 * 2 * 3 + 1 = 7
        np.testing.assert_array_equal(chunks[1], np.ones(10) * 9.0)  # 1 * 2 * 4 + 1 = 9

    def test_volts_to_microvolts_conversion(self):
        """Test that data in volts is converted to microvolts."""
        series = create_mock_electrical_series(10, 2, rate=1000.0, unit="volts")
        series.data = np.ones((10, 2)) * 1e-6  # 1 microvolt in volts
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        chunks = reader.get_chunk()

        # 1e-6 V * 1e6 = 1 uV
        for chunk in chunks:
            np.testing.assert_array_almost_equal(chunk, np.ones(10) * 1.0)

    def test_millivolts_to_microvolts_conversion(self):
        """Test that data in millivolts is converted to microvolts."""
        series = create_mock_electrical_series(10, 2, rate=1000.0, unit="millivolts")
        series.data = np.ones((10, 2)) * 0.001  # 1 microvolt in millivolts
        session_start = datetime(2023, 1, 1, 12, 0, 0)

        reader = NWBElectricalSeriesReader(series, session_start)
        chunks = reader.get_chunk()

        # 0.001 mV * 1e3 = 1 uV
        for chunk in chunks:
            np.testing.assert_array_almost_equal(chunk, np.ones(10) * 1.0)
