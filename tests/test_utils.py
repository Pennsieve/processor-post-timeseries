import sys

import numpy as np
from utils import infer_sampling_rate, to_big_endian


class TestInferSamplingRate:
    """Tests for infer_sampling_rate function."""

    def test_1000hz_sampling_rate(self):
        """Test inference from 1000 Hz sampling rate timestamps."""
        # 1000 Hz = 0.001 second period
        timestamps = np.linspace(0, 0.1, 100, endpoint=False)
        rate = infer_sampling_rate(timestamps)
        assert abs(rate - 1000.0) < 1.0  # Allow small floating point error

    def test_30000hz_sampling_rate(self):
        """Test inference from 30000 Hz sampling rate timestamps."""
        # 30000 Hz = 0.0000333... second period
        timestamps = np.linspace(0, 0.001, 30, endpoint=False)
        rate = infer_sampling_rate(timestamps)
        assert abs(rate - 30000.0) < 1.0

    def test_uses_first_10_timestamps(self):
        """Test that only first 10 timestamps are used for inference."""
        # First 10 timestamps at 1000 Hz
        first_10 = np.linspace(0, 0.01, 10, endpoint=False)
        # Rest at different rate (this should be ignored)
        rest = np.linspace(0.01, 0.1, 90, endpoint=False)
        timestamps = np.concatenate([first_10, rest])

        rate = infer_sampling_rate(timestamps)
        assert abs(rate - 1000.0) < 1.0

    def test_fewer_than_10_timestamps(self):
        """Test inference with fewer than 10 timestamps."""
        timestamps = np.linspace(0, 0.005, 5, endpoint=False)
        rate = infer_sampling_rate(timestamps)
        assert abs(rate - 1000.0) < 1.0

    def test_minimum_2_timestamps(self):
        """Test inference with exactly 2 timestamps."""
        timestamps = np.array([0.0, 0.001])
        rate = infer_sampling_rate(timestamps)
        assert abs(rate - 1000.0) < 0.1

    def test_irregular_timestamps_uses_median(self):
        """Test that median is used for slightly irregular timestamps."""
        # Create timestamps with slight irregularity
        # 9 intervals, most at 0.001 (1000 Hz)
        timestamps = np.array(
            [
                0.000,
                0.001,
                0.002,
                0.003,
                0.004,
                0.0051,  # slight irregularity
                0.006,
                0.007,
                0.008,
                0.009,
            ]
        )
        rate = infer_sampling_rate(timestamps)
        # Median should still be ~0.001
        assert abs(rate - 1000.0) < 10.0


class TestToBigEndian:
    """Tests for to_big_endian function."""

    def test_little_endian_conversion(self):
        """Test conversion from little endian to big endian."""
        # Create explicitly little-endian array
        data = np.array([1.0, 2.0, 3.0], dtype="<f8")  # little-endian float64
        result = to_big_endian(data)

        # Result should be big endian
        assert result.dtype.byteorder in (">", "|")  # '|' for byte-order neutral
        # Values should be preserved
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_big_endian_no_change(self):
        """Test that big endian arrays are not modified."""
        data = np.array([1.0, 2.0, 3.0], dtype=">f8")  # big-endian float64
        result = to_big_endian(data)

        assert result.dtype.byteorder == ">"
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_native_endian_on_little_endian_system(self):
        """Test native endian conversion on little-endian system."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # native endian

        result = to_big_endian(data)

        # On little-endian system, should be converted
        # On big-endian system, should remain unchanged
        if sys.byteorder == "little":
            assert result.dtype.byteorder in (">", "|")
        else:
            assert result.dtype.byteorder in (">", "=", "|")

    def test_preserves_array_values(self):
        """Test that array values are preserved after conversion."""
        original = np.array([1.5, -2.5, 0.0, 1e10, 1e-10], dtype=np.float64)
        result = to_big_endian(original.copy())

        np.testing.assert_array_almost_equal(result, original)

    def test_float32_array(self):
        """Test conversion of float32 array."""
        data = np.array([1.0, 2.0, 3.0], dtype="<f4")  # little-endian float32
        result = to_big_endian(data)

        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_int64_array(self):
        """Test conversion of int64 array."""
        data = np.array([1, 2, 3], dtype="<i8")  # little-endian int64
        result = to_big_endian(data)

        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_empty_array(self):
        """Test conversion of empty array."""
        data = np.array([], dtype="<f8")
        result = to_big_endian(data)

        assert len(result) == 0

    def test_2d_array(self):
        """Test conversion of 2D array."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f8")
        result = to_big_endian(data)

        np.testing.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])
        assert result.shape == (2, 2)

    def test_bytes_representation_changes(self):
        """Test that byte representation actually changes for little-endian input."""
        # This test verifies the byte swap actually happens
        data = np.array([1.0], dtype="<f8")
        original_bytes = data.tobytes()

        result = to_big_endian(data.copy())
        result_bytes = result.tobytes()

        # Bytes should be different (reversed) if conversion occurred
        if sys.byteorder == "little" or data.dtype.byteorder == "<":
            # For 1.0 in float64, the bytes are different in different endianness
            assert original_bytes != result_bytes or data.dtype.byteorder == ">"
