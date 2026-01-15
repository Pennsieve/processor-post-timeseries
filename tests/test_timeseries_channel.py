import pytest
from timeseries_channel import TimeSeriesChannel


class TestTimeSeriesChannelInit:
    """Tests for TimeSeriesChannel initialization."""

    def test_basic_initialization(self):
        """Test basic channel creation with required parameters."""
        channel = TimeSeriesChannel(index=0, name="Test Channel", rate=30000.0, start=1000000, end=2000000)

        assert channel.index == 0
        assert channel.name == "Test Channel"
        assert channel.rate == 30000.0
        assert channel.start == 1000000
        assert channel.end == 2000000
        assert channel.type == "CONTINUOUS"
        assert TimeSeriesChannel.UNIT == "uV"
        assert channel.group == "default"
        assert channel.last_annotation == 0
        assert channel.properties == []
        assert channel.id is None

    def test_initialization_with_all_parameters(self):
        """Test channel creation with all parameters specified."""
        channel = TimeSeriesChannel(
            index=5,
            name="  Channel 5  ",
            rate=10000.0,
            start=500000,
            end=1500000,
            type="UNIT",
            group="  electrode_group  ",
            last_annotation=100,
            properties=[{"key": "value"}],
            id="N:channel:123",
        )

        assert channel.index == 5
        assert channel.name == "Channel 5"  # should be stripped
        assert channel.rate == 10000.0
        assert channel.start == 500000
        assert channel.end == 1500000
        assert channel.type == "UNIT"  # should be uppercased
        assert TimeSeriesChannel.UNIT == "uV"  # unit is always uV
        assert channel.group == "electrode_group"  # should be stripped
        assert channel.last_annotation == 100
        assert channel.properties == [{"key": "value"}]
        assert channel.id == "N:channel:123"

    def test_type_case_insensitive(self):
        """Test that type is converted to uppercase."""
        channel = TimeSeriesChannel(index=0, name="Test", rate=1000.0, start=0, end=1000, type="continuous")
        assert channel.type == "CONTINUOUS"

        channel2 = TimeSeriesChannel(index=0, name="Test", rate=1000.0, start=0, end=1000, type="Unit")
        assert channel2.type == "UNIT"

    def test_invalid_type_raises_assertion(self):
        """Test that invalid type raises AssertionError."""
        with pytest.raises(AssertionError, match="Type must be CONTINUOUS or UNIT"):
            TimeSeriesChannel(index=0, name="Test", rate=1000.0, start=0, end=1000, type="INVALID")

    def test_start_end_converted_to_int(self):
        """Test that start and end are converted to integers."""
        channel = TimeSeriesChannel(index=0, name="Test", rate=1000.0, start=1000000.5, end=2000000.9)
        assert channel.start == 1000000
        assert channel.end == 2000000
        assert isinstance(channel.start, int)
        assert isinstance(channel.end, int)


class TestTimeSeriesChannelAsDict:
    """Tests for TimeSeriesChannel.as_dict() method."""

    def test_as_dict_without_id(self):
        """Test as_dict when id is None."""
        channel = TimeSeriesChannel(index=0, name="Test Channel", rate=30000.0, start=1000000, end=2000000)

        result = channel.as_dict()

        assert result == {
            "name": "Test Channel",
            "start": 1000000,
            "end": 2000000,
            "unit": "uV",
            "rate": 30000.0,
            "type": "CONTINUOUS",
            "group": "default",
            "lastAnnotation": 0,
            "properties": [],
        }
        assert "id" not in result

    def test_as_dict_with_id(self):
        """Test as_dict when id is set."""
        channel = TimeSeriesChannel(
            index=0, name="Test Channel", rate=30000.0, start=1000000, end=2000000, id="N:channel:abc-123"
        )

        result = channel.as_dict()

        assert "id" in result
        assert result["id"] == "N:channel:abc-123"

    def test_as_dict_with_custom_properties(self):
        """Test as_dict with custom properties."""
        channel = TimeSeriesChannel(
            index=0, name="Test", rate=1000.0, start=0, end=1000, properties=[{"key1": "value1"}, {"key2": "value2"}]
        )

        result = channel.as_dict()
        assert result["properties"] == [{"key1": "value1"}, {"key2": "value2"}]


class TestTimeSeriesChannelFromDict:
    """Tests for TimeSeriesChannel.from_dict() static method."""

    def test_from_dict_with_type_key(self, sample_channel_dict):
        """Test from_dict with 'type' key."""
        channel = TimeSeriesChannel.from_dict(sample_channel_dict)

        assert channel.name == "Channel 1"
        assert channel.start == 1000000
        assert channel.end == 2000000
        assert TimeSeriesChannel.UNIT == "uV"  # unit is always uV
        assert channel.rate == 30000.0
        assert channel.type == "CONTINUOUS"
        assert channel.group == "default"
        assert channel.last_annotation == 0
        assert channel.id == "N:channel:test-id-123"
        assert channel.index == -1  # Default when from_dict

    def test_from_dict_with_channel_type_key(self, sample_channel_dict_with_channel_type):
        """Test from_dict with 'channelType' key (API format)."""
        channel = TimeSeriesChannel.from_dict(sample_channel_dict_with_channel_type)

        assert channel.type == "CONTINUOUS"

    def test_from_dict_with_properties_override(self):
        """Test from_dict with properties parameter override."""
        channel_dict = {
            "name": "Channel 1",
            "start": 1000000,
            "end": 2000000,
            "unit": "uV",
            "rate": 30000.0,
            "type": "CONTINUOUS",
            "group": "default",
        }
        custom_props = [{"custom": "property"}]

        channel = TimeSeriesChannel.from_dict(channel_dict, properties=custom_props)

        assert channel.properties == custom_props

    def test_from_dict_without_optional_fields(self):
        """Test from_dict with minimal required fields."""
        minimal_dict = {
            "name": "Minimal Channel",
            "start": 0,
            "end": 1000,
            "unit": "uV",
            "rate": 1000.0,
            "type": "CONTINUOUS",
            "group": "default",
        }

        channel = TimeSeriesChannel.from_dict(minimal_dict)

        assert channel.name == "Minimal Channel"
        assert channel.last_annotation == 0
        assert channel.id is None

    def test_from_dict_start_end_converted_to_int(self):
        """Test that from_dict converts start/end to int."""
        data = {
            "name": "Test",
            "start": "1000000",
            "end": "2000000",
            "unit": "uV",
            "rate": 1000.0,
            "type": "CONTINUOUS",
            "group": "default",
        }

        channel = TimeSeriesChannel.from_dict(data)

        assert channel.start == 1000000
        assert channel.end == 2000000


class TestTimeSeriesChannelEquality:
    """Tests for TimeSeriesChannel custom equality comparison."""

    def test_equal_channels(self):
        """Test that channels with same name, type, and similar rate are equal."""
        channel1 = TimeSeriesChannel(index=0, name="Test Channel", rate=30000.0, start=1000000, end=2000000)
        channel2 = TimeSeriesChannel(
            index=1,  # Different index
            name="Test Channel",
            rate=30000.0,
            start=3000000,  # Different start
            end=4000000,  # Different end
        )

        assert channel1 == channel2

    def test_equal_channels_case_insensitive_name(self):
        """Test that name comparison is case-insensitive."""
        channel1 = TimeSeriesChannel(index=0, name="Test Channel", rate=30000.0, start=0, end=1000)
        channel2 = TimeSeriesChannel(index=0, name="TEST CHANNEL", rate=30000.0, start=0, end=1000)

        assert channel1 == channel2

    def test_equal_channels_case_insensitive_type(self):
        """Test that type comparison is case-insensitive."""
        channel1 = TimeSeriesChannel(index=0, name="Test", rate=30000.0, start=0, end=1000, type="CONTINUOUS")
        channel2 = TimeSeriesChannel(index=0, name="Test", rate=30000.0, start=0, end=1000, type="continuous")

        assert channel1 == channel2

    def test_equal_channels_rate_within_2_percent(self):
        """Test that channels with rate within 2% are equal."""
        channel1 = TimeSeriesChannel(index=0, name="Test", rate=30000.0, start=0, end=1000)
        # 1.5% difference
        channel2 = TimeSeriesChannel(index=0, name="Test", rate=30450.0, start=0, end=1000)

        assert channel1 == channel2

    def test_not_equal_different_name(self):
        """Test that channels with different names are not equal."""
        channel1 = TimeSeriesChannel(index=0, name="Channel A", rate=30000.0, start=0, end=1000)
        channel2 = TimeSeriesChannel(index=0, name="Channel B", rate=30000.0, start=0, end=1000)

        assert channel1 != channel2

    def test_not_equal_different_type(self):
        """Test that channels with different types are not equal."""
        channel1 = TimeSeriesChannel(index=0, name="Test", rate=30000.0, start=0, end=1000, type="CONTINUOUS")
        channel2 = TimeSeriesChannel(index=0, name="Test", rate=30000.0, start=0, end=1000, type="UNIT")

        assert channel1 != channel2

    def test_not_equal_rate_beyond_2_percent(self):
        """Test that channels with rate difference > 2% are not equal."""
        channel1 = TimeSeriesChannel(index=0, name="Test", rate=30000.0, start=0, end=1000)
        # 3% difference
        channel2 = TimeSeriesChannel(index=0, name="Test", rate=30900.0, start=0, end=1000)

        assert channel1 != channel2

    def test_equality_boundary_exactly_2_percent(self):
        """Test equality at exactly 2% rate difference boundary."""
        channel1 = TimeSeriesChannel(index=0, name="Test", rate=30000.0, start=0, end=1000)
        # Just over 2% difference - should NOT be equal (< 0.02 check)
        # The check is: abs(1-(self.rate/other.rate)) < 0.02
        # At 30600: abs(1-(30000/30600)) = abs(1-0.9804) = 0.0196 < 0.02 - EQUAL
        # At 30700: abs(1-(30000/30700)) = abs(1-0.9772) = 0.0228 > 0.02 - NOT EQUAL
        channel2 = TimeSeriesChannel(
            index=0,
            name="Test",
            rate=30700.0,  # Just over 2%
            start=0,
            end=1000,
        )

        assert channel1 != channel2


class TestTimeSeriesChannelRoundTrip:
    """Tests for round-trip serialization/deserialization."""

    def test_as_dict_from_dict_round_trip(self):
        """Test that as_dict() -> from_dict() preserves data."""
        original = TimeSeriesChannel(
            index=5,
            name="Round Trip Channel",
            rate=20000.0,
            start=500000,
            end=1500000,
            type="UNIT",
            group="test_group",
            last_annotation=50,
            properties=[{"key": "value"}],
            id="N:channel:round-trip",
        )

        serialized = original.as_dict()
        restored = TimeSeriesChannel.from_dict(serialized)

        assert restored.name == original.name
        assert restored.rate == original.rate
        assert restored.start == original.start
        assert restored.end == original.end
        assert restored.type == original.type
        assert serialized["unit"] == "uV"  # unit is always uV in serialized output
        assert restored.group == original.group
        assert restored.last_annotation == original.last_annotation
        assert restored.properties == original.properties
        assert restored.id == original.id
        # Index is not serialized, restored should be -1
        assert restored.index == -1
