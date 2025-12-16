import os
import sys
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from datetime import datetime

# Add processor directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "processor"))


@pytest.fixture
def sample_channel_dict():
    """Sample channel dictionary as returned from API."""
    return {
        "name": "Channel 1",
        "start": 1000000,
        "end": 2000000,
        "unit": "uV",
        "rate": 30000.0,
        "type": "CONTINUOUS",
        "group": "default",
        "lastAnnotation": 0,
        "properties": [],
        "id": "N:channel:test-id-123",
    }


@pytest.fixture
def sample_channel_dict_with_channel_type():
    """Sample channel dictionary with channelType key (API response format)."""
    return {
        "name": "Channel 1",
        "start": 1000000,
        "end": 2000000,
        "unit": "uV",
        "rate": 30000.0,
        "channelType": "CONTINUOUS",
        "group": "default",
        "lastAnnotation": 0,
        "properties": [],
        "id": "N:channel:test-id-123",
    }


@pytest.fixture
def mock_session_manager():
    """Mock session manager for API clients."""
    manager = Mock()
    manager.session_token = "mock-token-12345"
    manager.refresh_session = Mock()
    return manager


@pytest.fixture
def mock_authentication_client():
    """Mock authentication client."""
    client = Mock()
    client.authenticate = Mock(return_value="mock-access-token")
    return client


@pytest.fixture
def sample_timestamps():
    """Sample evenly-spaced timestamps at 1000 Hz."""
    return np.linspace(0, 1.0, 1000, endpoint=False)


@pytest.fixture
def sample_timestamps_with_gap():
    """Sample timestamps with a gap in the middle (for contiguous chunk testing)."""
    # First 500 samples at 1000 Hz (0 to 0.5 seconds)
    first_segment = np.linspace(0, 0.5, 500, endpoint=False)
    # Gap of 0.1 seconds, then another 500 samples
    second_segment = np.linspace(0.6, 1.1, 500, endpoint=False)
    return np.concatenate([first_segment, second_segment])


@pytest.fixture
def sample_electrical_series_data():
    """Sample 2D array of electrical series data (samples x channels)."""
    np.random.seed(42)
    return np.random.randn(1000, 4).astype(np.float64)


@pytest.fixture
def mock_electrical_series(sample_electrical_series_data, sample_timestamps):
    """Mock pynwb ElectricalSeries object."""
    series = Mock()
    series.data = sample_electrical_series_data
    series.rate = 1000.0
    series.timestamps = None
    series.conversion = 1.0
    series.offset = 0.0
    series.channel_conversion = None

    # Mock electrodes table
    mock_electrodes = []
    for i in range(4):
        electrode = Mock()
        electrode.group_name = f"group_{i}"
        mock_electrodes.append(electrode)

    series.electrodes = mock_electrodes
    series.electrodes.table = mock_electrodes

    return series


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for output files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def session_start_time():
    """Sample session start time."""
    return datetime(2023, 1, 1, 12, 0, 0)


@pytest.fixture
def sample_import_files():
    """Sample list of ImportFile objects for testing."""
    from clients.import_client import ImportFile
    import uuid

    files = []
    for i in range(100):
        files.append(
            ImportFile(
                upload_key=uuid.uuid4(),
                file_path=f"N:channel:test-id_{i}_1000000_2000000.bin.gz",
                local_path=f"/path/to/channel-{i:05d}_1000000_2000000.bin.gz",
            )
        )
    return files
