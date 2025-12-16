import pytest
import os
from unittest.mock import patch

from config import Config, getboolenv


class TestGetBoolEnv:
    """Tests for getboolenv helper function."""

    def test_true_values(self):
        """Test that 'true' and '1' return True."""
        with patch.dict(os.environ, {"TEST_VAR": "true"}):
            assert getboolenv("TEST_VAR") is True

        with patch.dict(os.environ, {"TEST_VAR": "True"}):
            assert getboolenv("TEST_VAR") is True

        with patch.dict(os.environ, {"TEST_VAR": "TRUE"}):
            assert getboolenv("TEST_VAR") is True

        with patch.dict(os.environ, {"TEST_VAR": "1"}):
            assert getboolenv("TEST_VAR") is True

    def test_false_values(self):
        """Test that other values return False."""
        with patch.dict(os.environ, {"TEST_VAR": "false"}):
            assert getboolenv("TEST_VAR") is False

        with patch.dict(os.environ, {"TEST_VAR": "False"}):
            assert getboolenv("TEST_VAR") is False

        with patch.dict(os.environ, {"TEST_VAR": "0"}):
            assert getboolenv("TEST_VAR") is False

        with patch.dict(os.environ, {"TEST_VAR": "no"}):
            assert getboolenv("TEST_VAR") is False

        with patch.dict(os.environ, {"TEST_VAR": ""}):
            assert getboolenv("TEST_VAR") is False

    def test_default_value_true(self):
        """Test default value when set to True."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove TEST_VAR if it exists
            os.environ.pop("TEST_VAR", None)
            assert getboolenv("TEST_VAR", default=True) is True

    def test_default_value_false(self):
        """Test default value when set to False."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TEST_VAR", None)
            assert getboolenv("TEST_VAR", default=False) is False

    def test_missing_var_uses_default(self):
        """Test that missing environment variable uses default."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NONEXISTENT_VAR", None)
            assert getboolenv("NONEXISTENT_VAR") is False
            assert getboolenv("NONEXISTENT_VAR", default=True) is True


class TestConfigLocalEnvironment:
    """Tests for Config class in local environment."""

    def test_local_environment_defaults(self, tmp_path):
        """Test Config initialization with local environment defaults."""
        env_vars = {
            "ENVIRONMENT": "local",
            "INPUT_DIR": str(tmp_path / "input"),
            "OUTPUT_DIR": str(tmp_path / "output"),
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()

            assert config.ENVIRONMENT == "local"
            assert config.INPUT_DIR == str(tmp_path / "input")
            assert config.OUTPUT_DIR == str(tmp_path / "output")
            assert config.CHUNK_SIZE_MB == 1  # Default
            assert config.IMPORTER_ENABLED is False  # Local default

    def test_local_environment_custom_chunk_size(self, tmp_path):
        """Test custom chunk size in local environment."""
        env_vars = {
            "ENVIRONMENT": "local",
            "INPUT_DIR": str(tmp_path),
            "OUTPUT_DIR": str(tmp_path),
            "CHUNK_SIZE_MB": "5",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            assert config.CHUNK_SIZE_MB == 5

    def test_local_environment_api_defaults(self, tmp_path):
        """Test API host defaults in local environment."""
        env_vars = {
            "ENVIRONMENT": "local",
            "INPUT_DIR": str(tmp_path),
            "OUTPUT_DIR": str(tmp_path),
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()

            assert config.API_HOST == "https://api.pennsieve.net"
            assert config.API_HOST2 == "https://api2.pennsieve.net"

    def test_local_environment_custom_api_hosts(self, tmp_path):
        """Test custom API hosts."""
        env_vars = {
            "ENVIRONMENT": "local",
            "INPUT_DIR": str(tmp_path),
            "OUTPUT_DIR": str(tmp_path),
            "PENNSIEVE_API_HOST": "https://custom.api.com",
            "PENNSIEVE_API_HOST2": "https://custom.api2.com",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()

            assert config.API_HOST == "https://custom.api.com"
            assert config.API_HOST2 == "https://custom.api2.com"

    def test_local_environment_api_credentials(self, tmp_path):
        """Test API credentials loading."""
        env_vars = {
            "ENVIRONMENT": "local",
            "INPUT_DIR": str(tmp_path),
            "OUTPUT_DIR": str(tmp_path),
            "PENNSIEVE_API_KEY": "test-api-key",
            "PENNSIEVE_API_SECRET": "test-api-secret",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()

            assert config.API_KEY == "test-api-key"
            assert config.API_SECRET == "test-api-secret"

    def test_local_importer_enabled_override(self, tmp_path):
        """Test IMPORTER_ENABLED can be overridden in local environment."""
        env_vars = {
            "ENVIRONMENT": "local",
            "INPUT_DIR": str(tmp_path),
            "OUTPUT_DIR": str(tmp_path),
            "IMPORTER_ENABLED": "true",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            assert config.IMPORTER_ENABLED is True


class TestConfigProductionEnvironment:
    """Tests for Config class in production environment."""

    def test_production_environment_directories(self, tmp_path):
        """Test Config in production environment uses OUTPUT_DIR for input."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        env_vars = {
            "ENVIRONMENT": "production",
            "OUTPUT_DIR": str(output_dir),
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()

            # In production, INPUT_DIR is set to OUTPUT_DIR
            assert config.INPUT_DIR == str(output_dir)
            # And OUTPUT_DIR is a subdirectory
            assert config.OUTPUT_DIR == str(output_dir / "output")

    def test_production_creates_output_subdirectory(self, tmp_path):
        """Test that production config creates output subdirectory."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        env_vars = {
            "ENVIRONMENT": "production",
            "OUTPUT_DIR": str(base_dir),
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()

            expected_output = base_dir / "output"
            assert os.path.exists(expected_output)
            assert config.OUTPUT_DIR == str(expected_output)

    def test_production_importer_enabled_by_default(self, tmp_path):
        """Test IMPORTER_ENABLED defaults to True in production."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        env_vars = {
            "ENVIRONMENT": "production",
            "OUTPUT_DIR": str(base_dir),
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            assert config.IMPORTER_ENABLED is True


class TestConfigWorkflowInstanceId:
    """Tests for workflow instance ID configuration."""

    def test_workflow_instance_id_from_integration_id(self, tmp_path):
        """Test WORKFLOW_INSTANCE_ID uses INTEGRATION_ID."""
        env_vars = {
            "ENVIRONMENT": "local",
            "INPUT_DIR": str(tmp_path),
            "OUTPUT_DIR": str(tmp_path),
            "INTEGRATION_ID": "workflow-123-456",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            assert config.WORKFLOW_INSTANCE_ID == "workflow-123-456"

    def test_workflow_instance_id_generates_uuid_if_missing(self, tmp_path):
        """Test WORKFLOW_INSTANCE_ID generates UUID if INTEGRATION_ID not set."""
        env_vars = {
            "ENVIRONMENT": "local",
            "INPUT_DIR": str(tmp_path),
            "OUTPUT_DIR": str(tmp_path),
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()

            # Should be a valid UUID format
            import uuid

            try:
                uuid.UUID(config.WORKFLOW_INSTANCE_ID)
            except ValueError:
                pytest.fail("WORKFLOW_INSTANCE_ID is not a valid UUID")


class TestConfigEdgeCases:
    """Edge case tests for Config."""

    def test_missing_input_dir_local(self, tmp_path):
        """Test Config with missing INPUT_DIR in local environment."""
        env_vars = {
            "ENVIRONMENT": "local",
            "OUTPUT_DIR": str(tmp_path),
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            assert config.INPUT_DIR is None

    def test_missing_output_dir_local(self, tmp_path):
        """Test Config with missing OUTPUT_DIR in local environment."""
        env_vars = {
            "ENVIRONMENT": "local",
            "INPUT_DIR": str(tmp_path),
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            assert config.OUTPUT_DIR is None

    def test_missing_api_credentials(self, tmp_path):
        """Test Config with missing API credentials."""
        env_vars = {
            "ENVIRONMENT": "local",
            "INPUT_DIR": str(tmp_path),
            "OUTPUT_DIR": str(tmp_path),
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            assert config.API_KEY is None
            assert config.API_SECRET is None

    def test_non_standard_environment(self, tmp_path):
        """Test Config with non-standard environment name."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        env_vars = {
            "ENVIRONMENT": "staging",
            "OUTPUT_DIR": str(base_dir),
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()

            # Non-local environments should use production logic
            assert config.ENVIRONMENT == "staging"
            assert config.INPUT_DIR == str(base_dir)

    def test_chunk_size_conversion_to_int(self, tmp_path):
        """Test that CHUNK_SIZE_MB is converted to integer."""
        env_vars = {
            "ENVIRONMENT": "local",
            "INPUT_DIR": str(tmp_path),
            "OUTPUT_DIR": str(tmp_path),
            "CHUNK_SIZE_MB": "10",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            assert config.CHUNK_SIZE_MB == 10
            assert isinstance(config.CHUNK_SIZE_MB, int)
