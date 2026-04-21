from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.config import env as env_module


class EnvLoadingTests(unittest.TestCase):
    def test_load_dotenv_if_present_falls_back_without_python_dotenv(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts"
        temp_root.mkdir(parents=True, exist_ok=True)

        temp_dir = Path(
            os.path.abspath(
                os.path.join(
                    temp_root,
                    "env_loading_test",
                )
            )
        )
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            env_path = temp_dir / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "OPENAI_BASE_URL='https://example.com/v1'",
                    ]
                ),
                encoding="utf-8",
            )

            original_api_key = os.environ.pop("OPENAI_API_KEY", None)
            original_base_url = os.environ.pop("OPENAI_BASE_URL", None)
            try:
                with patch.object(env_module, "load_dotenv", None):
                    env_module.load_dotenv_if_present(env_path)

                self.assertEqual(os.environ.get("OPENAI_API_KEY"), "test-key")
                self.assertEqual(os.environ.get("OPENAI_BASE_URL"), "https://example.com/v1")
            finally:
                if original_api_key is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = original_api_key

                if original_base_url is None:
                    os.environ.pop("OPENAI_BASE_URL", None)
                else:
                    os.environ["OPENAI_BASE_URL"] = original_base_url
        finally:
            env_path.unlink(missing_ok=True)
            temp_dir.rmdir()

    def test_load_dotenv_if_present_accepts_project_root_directory(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts"
        temp_root.mkdir(parents=True, exist_ok=True)

        temp_dir = Path(
            os.path.abspath(
                os.path.join(
                    temp_root,
                    "env_loading_root_test",
                )
            )
        )
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            env_path = temp_dir / ".env"
            env_path.write_text("OPENAI_API_KEY=test-key-from-root\n", encoding="utf-8")

            original_api_key = os.environ.pop("OPENAI_API_KEY", None)
            try:
                with patch.object(env_module, "load_dotenv", None):
                    env_module.load_dotenv_if_present(temp_dir)

                self.assertEqual(os.environ.get("OPENAI_API_KEY"), "test-key-from-root")
            finally:
                if original_api_key is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = original_api_key
        finally:
            env_path.unlink(missing_ok=True)
            temp_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
