# processor-post-timeseries

A processor for converting NWB (Neurodata Without Borders) files into chunked
timeseries data for the Pennsieve platform.

## Overview

This processor reads electrical series data from NWB files and:
1. Extracts channel data with proper scaling (conversion factors, offsets)
2. Writes chunked binary files (gzip-compressed, big-endian float64)
3. Generates channel metadata files (JSON)
4. Optionally uploads the processed data to Pennsieve via the import API

## Architecture

**main.py** - Entry point that orchestrates the processing pipeline.

**reader.py** - `NWBElectricalSeriesReader` reads NWB ElectricalSeries data,
handles timestamps and sampling rates, applies conversion factors and offsets,
and detects contiguous data chunks.

**writer.py** - `TimeSeriesChunkWriter` writes chunked binary data (.bin.gz)
and channel metadata (.metadata.json) in big-endian format.

**importer.py** - Creates import manifests via Pennsieve API
and uploads files to S3 via presigned URLs.

**clients/** - API clients for Pennsieve:
- `AuthenticationClient` - AWS Cognito authentication
- `ImportClient` - Import manifest creation and file upload
- `TimeSeriesClient` - Time series channel management
- `WorkflowClient` - Analytic workflow instance management
- `BaseClient` - Session management with auto-refresh

## Setup

### Prerequisites

- Python 3.10+
- Docker (for local runs)

### Create Virtual Environment

```bash
make venv
source venv/bin/activate
```

### Install Dependencies

```bash
make install
```

## Development

### Run Tests

```bash
make test
```

### Run Tests with Coverage

```bash
make test-cov
```

### Run Linter

```bash
make lint
```

## Running Locally

### 1. Configure Environment

Configure the environment file

Edit `dev.env` with your settings:

```env
ENVIRONMENT=local
INPUT_DIR=/data/input
OUTPUT_DIR=/data/output
CHUNK_SIZE_MB=1
IMPORTER_ENABLED=false
...
```

### 2. Add Input File

Place your `.nwb` file in the `data/input/` directory:

```bash
cp /path/to/your/file.nwb data/input/
```

### 3. Run the Processor

```bash
make run
```

This builds and runs the processor via Docker.
Output files will be written to `data/output/`.

### 4. Clean Up

Remove input/output files:

```bash
make clean
```

## Output Format

The processor generates two types of files per channel:

### Binary Data Files
- Pattern: `channel-{index}_{start_us}_{end_us}.bin.gz`
- Format: Gzip-compressed big-endian float64 values
- Example: `channel-00001_1000000_2000000.bin.gz`

### Metadata Files
- Pattern: `channel-{index}.metadata.json`
- Contains: name, rate, start, end, unit, type, group, properties

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Runtime environment (`local` or `production`) | `local` |
| `INPUT_DIR` | Directory containing NWB files | - |
| `OUTPUT_DIR` | Directory for output files | - |
| `CHUNK_SIZE_MB` | Size of each data chunk in MB | `1` |
| `IMPORTER_ENABLED` | Enable Pennsieve upload | `false` (local) |
| `PENNSIEVE_API_KEY` | Pennsieve API key | - |
| `PENNSIEVE_API_SECRET` | Pennsieve API secret | - |
| `PENNSIEVE_API_HOST` | Pennsieve API endpoint | `https://api.pennsieve.net` |
| `PENNSIEVE_API_HOST2` | Pennsieve API2 endpoint | `https://api2.pennsieve.net` |
| `INTEGRATION_ID` | Workflow instance ID | auto-generated |

## License

See LICENSE file.
