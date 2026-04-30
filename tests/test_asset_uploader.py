from unittest.mock import MagicMock, patch

import botocore.exceptions
import pytest
from asset_uploader import AssetUploader, ChunkUploadResult, relative_key_for_chunk
from clients.packages_assets_client import UploadCredentials


@pytest.fixture
def credentials():
    return UploadCredentials(
        access_key_id="AKIATEST",
        secret_access_key="SECRET",
        session_token="TOKEN",
        expiration="2026-04-30T13:00:00Z",
        bucket="pennsieve-viewer-assets",
        region="us-east-1",
        key_prefix="viewer-assets/O19/D2049/asset-uuid/",
    )


class TestRelativeKeyForChunk:
    def test_strips_directory(self):
        assert relative_key_for_chunk("/data/output/N:channel:abc_0_1000.bin.gz") == "N:channel:abc_0_1000.bin.gz"

    def test_no_directory_passthrough(self):
        assert relative_key_for_chunk("foo.bin.gz") == "foo.bin.gz"


class TestAssetUploaderClientConstruction:
    @patch("asset_uploader.boto3.client")
    def test_build_client_uses_credentials(self, mock_boto, credentials, tmp_path):
        # Trigger _build_client via upload_files (which is the only public path)
        local = tmp_path / "f.bin.gz"
        local.write_bytes(b"x")
        uploader = AssetUploader(credentials, max_workers=1)
        uploader.upload_files([(str(local), "f.bin.gz")])

        mock_boto.assert_called_once_with(
            "s3",
            aws_access_key_id="AKIATEST",
            aws_secret_access_key="SECRET",
            aws_session_token="TOKEN",
            region_name="us-east-1",
        )


class TestAssetUploaderUploadFiles:
    @patch("asset_uploader.boto3.client")
    def test_uploads_each_file_with_prefixed_key(self, mock_boto, credentials, tmp_path):
        s3 = MagicMock()
        mock_boto.return_value = s3

        files = []
        for i in range(3):
            p = tmp_path / f"chunk_{i}.bin.gz"
            p.write_bytes(b"x")
            files.append((str(p), f"chunk_{i}.bin.gz"))

        uploader = AssetUploader(credentials, max_workers=1)
        results = uploader.upload_files(files)

        # Each file's key is the credentials.key_prefix + relative_key
        expected_calls = [
            (
                (
                    str(tmp_path / "chunk_0.bin.gz"),
                    "pennsieve-viewer-assets",
                    "viewer-assets/O19/D2049/asset-uuid/chunk_0.bin.gz",
                ),
            ),
            (
                (
                    str(tmp_path / "chunk_1.bin.gz"),
                    "pennsieve-viewer-assets",
                    "viewer-assets/O19/D2049/asset-uuid/chunk_1.bin.gz",
                ),
            ),
            (
                (
                    str(tmp_path / "chunk_2.bin.gz"),
                    "pennsieve-viewer-assets",
                    "viewer-assets/O19/D2049/asset-uuid/chunk_2.bin.gz",
                ),
            ),
        ]
        # Order may be parallel-shuffled with workers > 1; verify set equality
        assert s3.upload_file.call_count == 3
        actual_args_set = {c.args for c in s3.upload_file.call_args_list}
        expected_args_set = {ec[0] for ec in expected_calls}
        assert actual_args_set == expected_args_set

        # Results preserve input order regardless of upload completion order
        assert [r.relative_key for r in results] == [
            "chunk_0.bin.gz",
            "chunk_1.bin.gz",
            "chunk_2.bin.gz",
        ]
        assert all(r.full_key.startswith("viewer-assets/O19/D2049/asset-uuid/") for r in results)

    @patch("asset_uploader.boto3.client")
    def test_returns_chunk_upload_result_per_file(self, mock_boto, credentials, tmp_path):
        mock_boto.return_value = MagicMock()
        local = tmp_path / "f.bin.gz"
        local.write_bytes(b"x")

        uploader = AssetUploader(credentials, max_workers=1)
        results = uploader.upload_files([(str(local), "rel.bin.gz")])

        assert len(results) == 1
        assert isinstance(results[0], ChunkUploadResult)
        assert results[0].relative_key == "rel.bin.gz"
        assert results[0].full_key == "viewer-assets/O19/D2049/asset-uuid/rel.bin.gz"

    @patch("asset_uploader.boto3.client")
    def test_empty_input_does_not_upload(self, mock_boto, credentials):
        s3 = MagicMock()
        mock_boto.return_value = s3
        uploader = AssetUploader(credentials, max_workers=1)
        assert uploader.upload_files([]) == []
        s3.upload_file.assert_not_called()

    @patch("asset_uploader.boto3.client")
    def test_propagates_client_error(self, mock_boto, credentials, tmp_path):
        s3 = MagicMock()
        s3.upload_file.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "ExpiredToken", "Message": "creds expired"}},
            "PutObject",
        )
        mock_boto.return_value = s3

        local = tmp_path / "f.bin.gz"
        local.write_bytes(b"x")
        uploader = AssetUploader(credentials, max_workers=1)

        with pytest.raises(botocore.exceptions.ClientError):
            uploader.upload_files([(str(local), "f.bin.gz")])
