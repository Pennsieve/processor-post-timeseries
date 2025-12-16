#!/usr/bin/env python3
"""
Generate a test NWB file with two channels of sine wave timeseries data.

This script creates an NWB file compatible with the processor-post-timeseries
processor. The file contains two channels with sine waves at different frequencies.

Usage:
    python3 generate_test_nwb.py --size 10MB --output test.nwb
    python3 generate_test_nwb.py --size 1GB --output large_test.nwb
    python3 generate_test_nwb.py --size 50GB --output huge_test.nwb

Size format: <number><unit> where unit is B, KB, MB, GB, or TB
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import ElectricalSeries


def parse_size(size_str: str) -> int:
    """Parse a human-readable size string into bytes."""
    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }

    match = re.match(r"^(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)$", size_str.upper().strip())
    if not match:
        raise ValueError(f"Invalid size format: '{size_str}'. Use format like '10MB', '1GB', '50GB'")

    value = float(match.group(1))
    unit = match.group(2)

    return int(value * units[unit])


def calculate_samples_for_size(target_bytes: int, num_channels: int = 2) -> int:
    """
    Calculate the number of samples needed to achieve target file size.

    NWB stores data as 64-bit floats (8 bytes per value).
    With HDF5 overhead, actual file size is slightly larger than raw data.
    We account for ~5% overhead for HDF5 metadata and structure.
    """
    bytes_per_sample = 8 * num_channels  # 8 bytes per float64, per channel
    overhead_factor = 0.95  # Account for HDF5 overhead

    effective_data_bytes = target_bytes * overhead_factor
    num_samples = int(effective_data_bytes / bytes_per_sample)

    return max(num_samples, 1000)  # Minimum 1000 samples


def generate_sine_wave(
    num_samples: int, frequency: float, sampling_rate: float, amplitude: float = 100.0, phase: float = 0.0
) -> np.ndarray:
    """
    Generate a sine wave signal.

    Args:
        num_samples: Number of samples to generate
        frequency: Frequency of the sine wave in Hz
        sampling_rate: Sampling rate in Hz
        amplitude: Peak amplitude of the sine wave (in microvolts)
        phase: Phase offset in radians

    Returns:
        numpy array of float64 values
    """
    t = np.arange(num_samples) / sampling_rate
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


def create_nwb_file(
    output_path: str, target_size_bytes: int, freq1: float = 10.0, freq2: float = 25.0, sampling_rate: float = 1000.0
) -> dict:
    """
    Create an NWB file with two channels of sine wave data.

    Args:
        output_path: Path to save the NWB file
        target_size_bytes: Target file size in bytes
        freq1: Frequency of channel 1 sine wave (Hz)
        freq2: Frequency of channel 2 sine wave (Hz)
        sampling_rate: Sampling rate for both channels (Hz)

    Returns:
        Dictionary with metadata about the created file
    """
    num_channels = 2
    num_samples = calculate_samples_for_size(target_size_bytes, num_channels)
    duration_seconds = num_samples / sampling_rate

    print("Generating NWB file with:")
    print(f"  Target size: {target_size_bytes / (1024**2):.2f} MB")
    print(f"  Samples: {num_samples:,}")
    print(f"  Duration: {duration_seconds:.2f} seconds ({duration_seconds/3600:.2f} hours)")
    print(f"  Sampling rate: {sampling_rate} Hz")
    print(f"  Channel 1 frequency: {freq1} Hz")
    print(f"  Channel 2 frequency: {freq2} Hz")
    print()

    # Generate sine wave data for both channels
    print("Generating sine wave data...")
    channel1_data = generate_sine_wave(num_samples, freq1, sampling_rate, amplitude=100.0)
    channel2_data = generate_sine_wave(num_samples, freq2, sampling_rate, amplitude=150.0, phase=np.pi / 4)

    # Stack into shape (num_samples, num_channels)
    data = np.column_stack([channel1_data, channel2_data])
    print(f"  Data shape: {data.shape}")
    print(f"  Data size: {data.nbytes / (1024**2):.2f} MB")

    # Create NWB file
    print("\nCreating NWB file structure...")
    session_start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    nwbfile = NWBFile(
        session_description="Test NWB file with sine wave timeseries data",
        identifier=f"test_nwb_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        session_start_time=session_start_time,
        experimenter=["Test Generator"],
        lab="Test Lab",
        institution="Test Institution",
        experiment_description="Generated test data with two sine wave channels",
    )

    # Create device
    device = nwbfile.create_device(
        name="TestDevice",
        description="Virtual test device for generating sine wave data",
        manufacturer="Test Manufacturer",
    )

    # Create electrode group
    electrode_group = nwbfile.create_electrode_group(
        name="TestElectrodeGroup",
        description="Test electrode group with two channels",
        location="Test Location",
        device=device,
    )

    # Add electrodes to the electrode table
    # The processor expects 'channel_name' or 'label' column and 'group_name'
    nwbfile.add_electrode_column(name="channel_name", description="Name of the electrode channel")

    nwbfile.add_electrode(
        x=0.0,
        y=0.0,
        z=0.0,
        imp=1000.0,
        location="Test Location 1",
        filtering="None",
        group=electrode_group,
        channel_name=f"SineWave_{freq1}Hz",
    )

    nwbfile.add_electrode(
        x=1.0,
        y=0.0,
        z=0.0,
        imp=1000.0,
        location="Test Location 2",
        filtering="None",
        group=electrode_group,
        channel_name=f"SineWave_{freq2}Hz",
    )

    # Create electrode table region for all electrodes
    electrode_table_region = nwbfile.create_electrode_table_region(region=[0, 1], description="All test electrodes")

    # Create ElectricalSeries with the sine wave data
    electrical_series = ElectricalSeries(
        name="TestElectricalSeries",
        description="Two channels of sine wave data at different frequencies",
        data=data,
        electrodes=electrode_table_region,
        rate=sampling_rate,
        conversion=1e-6,  # Data is in microvolts, conversion to volts
        offset=0.0,
        starting_time=0.0,
    )

    # Add to acquisition
    nwbfile.add_acquisition(electrical_series)

    # Write the file
    print(f"Writing NWB file to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with NWBHDF5IO(str(output_path), mode="w") as io:
        io.write(nwbfile)

    actual_size = output_path.stat().st_size
    print("\nFile created successfully!")
    print(f"  Actual file size: {actual_size / (1024**2):.2f} MB")
    print(f"  Size ratio: {actual_size / target_size_bytes:.2%} of target")

    return {
        "output_path": str(output_path),
        "target_size_bytes": target_size_bytes,
        "actual_size_bytes": actual_size,
        "num_samples": num_samples,
        "num_channels": num_channels,
        "duration_seconds": duration_seconds,
        "sampling_rate": sampling_rate,
        "channel_frequencies": [freq1, freq2],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a test NWB file with sine wave timeseries data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --size 10MB --output test.nwb
    %(prog)s --size 1GB --output large_test.nwb
    %(prog)s --size 50GB --output huge_test.nwb
    %(prog)s --size 100MB --freq1 5 --freq2 50 --rate 2000 --output custom.nwb
        """,
    )

    parser.add_argument("--size", "-s", required=True, help="Target file size (e.g., '10MB', '1GB', '50GB')")

    parser.add_argument("--output", "-o", required=True, help="Output NWB file path")

    parser.add_argument(
        "--freq1", type=float, default=10.0, help="Frequency of channel 1 sine wave in Hz (default: 10.0)"
    )

    parser.add_argument(
        "--freq2", type=float, default=25.0, help="Frequency of channel 2 sine wave in Hz (default: 25.0)"
    )

    parser.add_argument("--rate", type=float, default=1000.0, help="Sampling rate in Hz (default: 1000.0)")

    args = parser.parse_args()

    try:
        target_size = parse_size(args.size)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate frequencies against Nyquist
    nyquist = args.rate / 2
    if args.freq1 >= nyquist or args.freq2 >= nyquist:
        print(f"Error: Frequencies must be less than Nyquist frequency ({nyquist} Hz)", file=sys.stderr)
        print("  Increase --rate or decrease --freq1/--freq2", file=sys.stderr)
        sys.exit(1)

    try:
        result = create_nwb_file(
            output_path=args.output,
            target_size_bytes=target_size,
            freq1=args.freq1,
            freq2=args.freq2,
            sampling_rate=args.rate,
        )

        print("\nFile metadata summary:")
        print(f"  Path: {result['output_path']}")
        print(f"  Channels: {result['num_channels']}")
        print(f"  Samples per channel: {result['num_samples']:,}")
        print(f"  Duration: {result['duration_seconds']:.2f}s ({result['duration_seconds']/3600:.2f}h)")
        print(f"  Sampling rate: {result['sampling_rate']} Hz")
        print(f"  Channel frequencies: {result['channel_frequencies']} Hz")

    except Exception as e:
        print(f"Error creating NWB file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
