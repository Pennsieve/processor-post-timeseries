import gzip
import json
import os
from unittest.mock import Mock, patch

import numpy as np
from constants import TIME_SERIES_BINARY_FILE_EXTENSION, TIME_SERIES_METADATA_FILE_EXTENSION
from timeseries_channel import TimeSeriesChannel
from writer import TimeSeriesChunkWriter


class TestTimeSeriesChunkWriterInit:
    """Tests for TimeSeriesChunkWriter initialization."""

    def test_initialization(self, temp_output_dir, session_start_time):
        """Test basic initialization."""
        writer = TimeSeriesChunkWriter(
            session_start_time=session_start_time, output_dir=temp_output_dir, chunk_size=1000
        )

        assert writer.session_start_time == session_start_time
        assert writer.output_dir == temp_output_dir
        assert writer.chunk_size == 1000


class TestWriteChunk:
    """Tests for write_chunk static method."""

    def test_write_chunk_creates_file(self, temp_output_dir):
        """Test that write_chunk creates a binary file."""
        chunk = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        start_time = 1.0
        end_time = 1.005

        TimeSeriesChunkWriter.write_chunk(chunk, start_time, end_time, 0, temp_output_dir)

        # Check file was created (use round() to match writer behavior)
        expected_filename = (
            f"channel-00000_{round(start_time * 1e6)}_{round(end_time * 1e6)}{TIME_SERIES_BINARY_FILE_EXTENSION}"
        )
        file_path = os.path.join(temp_output_dir, expected_filename)
        assert os.path.exists(file_path)

    def test_write_chunk_gzip_compressed(self, temp_output_dir):
        """Test that output file is gzip compressed."""
        chunk = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        TimeSeriesChunkWriter.write_chunk(chunk, 1.0, 1.003, 0, temp_output_dir)

        # Find the file (timestamps may vary slightly)
        files = [f for f in os.listdir(temp_output_dir) if f.endswith(".bin.gz")]
        assert len(files) == 1
        file_path = os.path.join(temp_output_dir, files[0])

        # Should be readable as gzip
        with gzip.open(file_path, "rb") as f:
            data = f.read()
            assert len(data) > 0

    def test_write_chunk_big_endian_format(self, temp_output_dir):
        """Test that data is written in big-endian format."""
        chunk = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        TimeSeriesChunkWriter.write_chunk(chunk, 1.0, 1.003, 0, temp_output_dir)

        # Find the file
        files = [f for f in os.listdir(temp_output_dir) if f.endswith(".bin.gz")]
        assert len(files) == 1
        file_path = os.path.join(temp_output_dir, files[0])

        with gzip.open(file_path, "rb") as f:
            data = f.read()

        # Read as big-endian float64
        result = np.frombuffer(data, dtype=">f8")
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_write_chunk_channel_index_formatting(self, temp_output_dir):
        """Test that channel index is zero-padded to 5 digits."""
        chunk = np.array([1.0], dtype=np.float64)

        # Test various channel indices with unique timestamps to avoid overwriting
        for i, index in enumerate([0, 5, 42, 999, 12345]):
            start_time = 1.0 + i * 0.1
            end_time = start_time + 0.001
            TimeSeriesChunkWriter.write_chunk(chunk, start_time, end_time, index, temp_output_dir)

            # Check that file with correct channel index prefix exists
            files = [f for f in os.listdir(temp_output_dir) if f.startswith(f"channel-{index:05d}_")]
            assert len(files) >= 1, f"No file found for channel index {index}"

    def test_write_chunk_preserves_data_precision(self, temp_output_dir):
        """Test that float64 precision is preserved."""
        # Use values that require float64 precision
        chunk = np.array([1.123456789012345, -9.87654321098765e10, 1e-15], dtype=np.float64)

        TimeSeriesChunkWriter.write_chunk(chunk, 1.0, 1.003, 0, temp_output_dir)

        # Find the file
        files = [f for f in os.listdir(temp_output_dir) if f.endswith(".bin.gz")]
        assert len(files) == 1
        file_path = os.path.join(temp_output_dir, files[0])

        with gzip.open(file_path, "rb") as f:
            data = f.read()

        result = np.frombuffer(data, dtype=">f8")
        np.testing.assert_array_almost_equal(result, chunk, decimal=14)


class TestWriteChannel:
    """Tests for write_channel method."""

    def test_write_channel_creates_metadata_file(self, temp_output_dir, session_start_time):
        """Test that write_channel creates a JSON metadata file."""
        writer = TimeSeriesChunkWriter(session_start_time, temp_output_dir, 1000)

        channel = TimeSeriesChannel(
            index=5, name="Test Channel", rate=30000.0, start=1000000, end=2000000, group="electrode_group"
        )

        writer.write_channel(channel)

        expected_filename = f"channel-00005{TIME_SERIES_METADATA_FILE_EXTENSION}"
        file_path = os.path.join(temp_output_dir, expected_filename)
        assert os.path.exists(file_path)

    def test_write_channel_json_content(self, temp_output_dir, session_start_time):
        """Test that metadata file contains correct JSON."""
        writer = TimeSeriesChunkWriter(session_start_time, temp_output_dir, 1000)

        channel = TimeSeriesChannel(
            index=0,
            name="Test Channel",
            rate=30000.0,
            start=1000000,
            end=2000000,
            type="CONTINUOUS",
            group="test_group",
            last_annotation=100,
            properties=[{"key": "value"}],
        )

        writer.write_channel(channel)

        file_path = os.path.join(temp_output_dir, "channel-00000.metadata.json")

        with open(file_path, "r") as f:
            data = json.load(f)

        assert data["name"] == "Test Channel"
        assert data["rate"] == 30000.0
        assert data["start"] == 1000000
        assert data["end"] == 2000000
        assert data["unit"] == "uV"  # unit is always uV
        assert data["type"] == "CONTINUOUS"
        assert data["group"] == "test_group"
        assert data["lastAnnotation"] == 100
        assert data["properties"] == [{"key": "value"}]


class TestWriteElectricalSeries:
    """Tests for write_electrical_series method."""

    def test_write_electrical_series_single_chunk(self, temp_output_dir, session_start_time):
        """Test writing electrical series that fits in single chunk."""
        writer = TimeSeriesChunkWriter(session_start_time, temp_output_dir, chunk_size=1000)

        # Create mock electrical series with 500 samples (less than chunk_size)
        mock_reader = Mock()
        mock_reader.channels = [TimeSeriesChannel(index=0, name="Ch0", rate=1000.0, start=0, end=1000)]
        mock_reader.contiguous_chunks.return_value = [(0, 500)]
        mock_reader.get_chunk.return_value = [np.random.randn(500).astype(np.float64)]
        mock_reader.get_timestamp.side_effect = lambda idx: float(idx) / 1000.0

        with patch("writer.NWBElectricalSeriesReader", return_value=mock_reader):
            mock_series = Mock()
            writer.write_electrical_series(mock_series)

        # Should have created 1 binary file and 1 metadata file
        files = os.listdir(temp_output_dir)
        bin_files = [f for f in files if f.endswith(".bin.gz")]
        json_files = [f for f in files if f.endswith(".metadata.json")]

        assert len(bin_files) == 1
        assert len(json_files) == 1

    def test_write_electrical_series_multiple_chunks(self, temp_output_dir, session_start_time):
        """Test writing electrical series that requires multiple chunks."""
        writer = TimeSeriesChunkWriter(session_start_time, temp_output_dir, chunk_size=100)

        # Create mock with 250 samples, 2 channels
        mock_reader = Mock()
        mock_reader.channels = [
            TimeSeriesChannel(index=0, name="Ch0", rate=1000.0, start=0, end=1000),
            TimeSeriesChannel(index=1, name="Ch1", rate=1000.0, start=0, end=1000),
        ]
        mock_reader.contiguous_chunks.return_value = [(0, 250)]
        mock_reader.get_chunk.return_value = [
            np.random.randn(100).astype(np.float64),
            np.random.randn(100).astype(np.float64),
        ]
        mock_reader.get_timestamp.side_effect = lambda idx: float(idx) / 1000.0

        with patch("writer.NWBElectricalSeriesReader", return_value=mock_reader):
            mock_series = Mock()
            writer.write_electrical_series(mock_series)

        files = os.listdir(temp_output_dir)
        bin_files = [f for f in files if f.endswith(".bin.gz")]
        json_files = [f for f in files if f.endswith(".metadata.json")]

        # 250 samples / 100 chunk_size = 3 chunks per channel, 2 channels = 6 binary files
        assert len(bin_files) == 6
        # 2 metadata files (one per channel)
        assert len(json_files) == 2

    def test_write_electrical_series_with_gap(self, temp_output_dir, session_start_time):
        """Test writing electrical series with data gap."""
        writer = TimeSeriesChunkWriter(session_start_time, temp_output_dir, chunk_size=100)

        mock_reader = Mock()
        mock_reader.channels = [TimeSeriesChannel(index=0, name="Ch0", rate=1000.0, start=0, end=1000)]
        # Two contiguous segments
        mock_reader.contiguous_chunks.return_value = [(0, 100), (100, 200)]
        mock_reader.get_chunk.return_value = [np.random.randn(100).astype(np.float64)]
        mock_reader.get_timestamp.side_effect = lambda idx: float(idx) / 1000.0

        with patch("writer.NWBElectricalSeriesReader", return_value=mock_reader):
            mock_series = Mock()
            writer.write_electrical_series(mock_series)

        files = os.listdir(temp_output_dir)
        bin_files = [f for f in files if f.endswith(".bin.gz")]

        # 2 contiguous segments, 1 chunk each = 2 binary files
        assert len(bin_files) == 2

    def test_write_electrical_series_chunk_timestamps(self, temp_output_dir, session_start_time):
        """Test that chunk filenames have correct timestamp values."""
        writer = TimeSeriesChunkWriter(session_start_time, temp_output_dir, chunk_size=50)

        mock_reader = Mock()
        mock_reader.channels = [TimeSeriesChannel(index=0, name="Ch0", rate=1000.0, start=0, end=1000)]
        # 100 samples at 1000 Hz = 0.1 seconds
        timestamps = np.linspace(1.0, 1.1, 100, endpoint=False)
        mock_reader.contiguous_chunks.return_value = [(0, 100)]
        mock_reader.get_chunk.return_value = [np.random.randn(50).astype(np.float64)]
        mock_reader.get_timestamp.side_effect = lambda idx: float(timestamps[idx])

        with patch("writer.NWBElectricalSeriesReader", return_value=mock_reader):
            mock_series = Mock()
            writer.write_electrical_series(mock_series)

        files = os.listdir(temp_output_dir)
        bin_files = sorted([f for f in files if f.endswith(".bin.gz")])

        # First chunk: timestamps[0] to timestamps[49]
        # Second chunk: timestamps[50] to timestamps[99]
        assert len(bin_files) == 2

        # Check first chunk filename contains correct timestamps
        assert "1000000_1049000" in bin_files[0]  # 1.0 to 1.049 seconds in microseconds
        assert "1050000_1099000" in bin_files[1]  # 1.05 to 1.099 seconds


class TestWriteChunkEdgeCases:
    """Edge case tests for chunk writing."""

    def test_write_empty_chunk(self, temp_output_dir):
        """Test writing an empty chunk."""
        chunk = np.array([], dtype=np.float64)

        TimeSeriesChunkWriter.write_chunk(chunk, 1.0, 1.0, 0, temp_output_dir)

        file_path = os.path.join(temp_output_dir, "channel-00000_1000000_1000000.bin.gz")

        with gzip.open(file_path, "rb") as f:
            data = f.read()

        assert len(data) == 0

    def test_write_large_chunk(self, temp_output_dir):
        """Test writing a large chunk."""
        # 1 million samples
        chunk = np.random.randn(1000000).astype(np.float64)

        TimeSeriesChunkWriter.write_chunk(chunk, 0.0, 1000.0, 0, temp_output_dir)

        file_path = os.path.join(temp_output_dir, "channel-00000_0_1000000000.bin.gz")
        assert os.path.exists(file_path)

        # Verify data integrity
        with gzip.open(file_path, "rb") as f:
            data = f.read()

        result = np.frombuffer(data, dtype=">f8")
        assert len(result) == 1000000

    def test_write_chunk_special_float_values(self, temp_output_dir):
        """Test writing chunks with special float values."""
        chunk = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0], dtype=np.float64)

        TimeSeriesChunkWriter.write_chunk(chunk, 1.0, 1.005, 0, temp_output_dir)

        # Find the file
        files = [f for f in os.listdir(temp_output_dir) if f.endswith(".bin.gz")]
        assert len(files) == 1
        file_path = os.path.join(temp_output_dir, files[0])

        with gzip.open(file_path, "rb") as f:
            data = f.read()

        result = np.frombuffer(data, dtype=">f8")
        assert np.isinf(result[0]) and result[0] > 0
        assert np.isinf(result[1]) and result[1] < 0
        assert np.isnan(result[2])


class TestParallelProcessing:
    """Tests for parallel channel processing functionality."""

    def test_parallel_processing_many_channels(self, temp_output_dir, session_start_time):
        """Test parallel processing with many channels (typical neuroscience scenario)."""
        num_channels = 64
        writer = TimeSeriesChunkWriter(session_start_time, temp_output_dir, chunk_size=1000)

        mock_reader = Mock()
        mock_reader.channels = [
            TimeSeriesChannel(index=i, name=f"Ch{i}", rate=30000.0, start=0, end=1000) for i in range(num_channels)
        ]
        mock_reader.contiguous_chunks.return_value = [(0, 500)]
        mock_reader.get_chunk.return_value = [np.random.randn(500).astype(np.float64) for _ in range(num_channels)]
        mock_reader.get_timestamp.side_effect = lambda idx: float(idx) / 30000.0

        with patch("writer.NWBElectricalSeriesReader", return_value=mock_reader):
            mock_series = Mock()
            writer.write_electrical_series(mock_series)

        files = os.listdir(temp_output_dir)
        bin_files = [f for f in files if f.endswith(".bin.gz")]
        json_files = [f for f in files if f.endswith(".metadata.json")]

        # Should have 64 binary files (1 chunk x 64 channels)
        assert len(bin_files) == num_channels
        # Should have 64 metadata files
        assert len(json_files) == num_channels

    def test_parallel_processing_with_max_workers(self, temp_output_dir, session_start_time):
        """Test that max_workers parameter is respected."""
        writer = TimeSeriesChunkWriter(session_start_time, temp_output_dir, chunk_size=1000)

        mock_reader = Mock()
        mock_reader.channels = [
            TimeSeriesChannel(index=i, name=f"Ch{i}", rate=1000.0, start=0, end=1000) for i in range(8)
        ]
        mock_reader.contiguous_chunks.return_value = [(0, 500)]
        mock_reader.get_chunk.return_value = [np.random.randn(500).astype(np.float64) for _ in range(8)]
        mock_reader.get_timestamp.side_effect = lambda idx: float(idx) / 1000.0

        with patch("writer.NWBElectricalSeriesReader", return_value=mock_reader):
            mock_series = Mock()
            # Limit to 2 workers
            writer.write_electrical_series(mock_series, max_workers=2)

        files = os.listdir(temp_output_dir)
        bin_files = [f for f in files if f.endswith(".bin.gz")]

        # Should still produce correct output
        assert len(bin_files) == 8

    def test_parallel_processing_data_integrity(self, temp_output_dir, session_start_time):
        """Test that parallel processing maintains data integrity across channels."""
        num_channels = 4
        writer = TimeSeriesChunkWriter(session_start_time, temp_output_dir, chunk_size=100)

        # Create distinct data for each channel
        channel_data = [np.arange(100, dtype=np.float64) + i * 1000 for i in range(num_channels)]

        mock_reader = Mock()
        mock_reader.channels = [
            TimeSeriesChannel(index=i, name=f"Ch{i}", rate=1000.0, start=0, end=1000) for i in range(num_channels)
        ]
        mock_reader.contiguous_chunks.return_value = [(0, 100)]
        mock_reader.get_chunk.return_value = channel_data
        mock_reader.get_timestamp.side_effect = lambda idx: float(idx) / 1000.0

        with patch("writer.NWBElectricalSeriesReader", return_value=mock_reader):
            mock_series = Mock()
            writer.write_electrical_series(mock_series)

        # Verify each channel's data
        for i in range(num_channels):
            file_pattern = f"channel-{i:05d}_"
            matching_files = [
                f for f in os.listdir(temp_output_dir) if f.startswith(file_pattern) and f.endswith(".bin.gz")
            ]
            assert len(matching_files) == 1

            file_path = os.path.join(temp_output_dir, matching_files[0])
            with gzip.open(file_path, "rb") as f:
                data = f.read()

            result = np.frombuffer(data, dtype=">f8")
            expected = np.arange(100, dtype=np.float64) + i * 1000
            np.testing.assert_array_equal(result, expected)
