"""Ollama API integration."""

import subprocess
from pathlib import Path
from typing import Any, Optional

import httpx

from finetune.utils.logging import get_logger

logger = get_logger("ollama")


class OllamaError(Exception):
    """Error communicating with Ollama."""

    pass


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, host: str = "http://localhost:11434"):
        """Initialize Ollama client.

        Args:
            host: Ollama server URL
        """
        self.host = host.rstrip("/")
        self.api_url = f"{self.host}/api"

    def check_connection(self) -> bool:
        """Check if Ollama server is reachable.

        Returns:
            True if connection successful
        """
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=5.0)
            return response.status_code == 200
        except httpx.RequestError:
            return False

    def list_models(self) -> list[dict[str, Any]]:
        """List all models available in Ollama.

        Returns:
            List of model information dictionaries

        Raises:
            OllamaError: If request fails
        """
        try:
            response = httpx.get(f"{self.api_url}/tags", timeout=10.0)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except httpx.RequestError as e:
            raise OllamaError(f"Failed to list models: {e}") from e

    def model_exists(self, name: str) -> bool:
        """Check if a model exists in Ollama.

        Args:
            name: Model name

        Returns:
            True if model exists
        """
        try:
            models = self.list_models()
            return any(m.get("name", "").startswith(name) for m in models)
        except OllamaError:
            return False

    def create_model(
        self,
        name: str,
        modelfile_path: Path,
        stream: bool = True,
    ) -> None:
        """Create a model in Ollama from a Modelfile.

        Args:
            name: Model name
            modelfile_path: Path to Modelfile
            stream: Whether to stream progress

        Raises:
            OllamaError: If creation fails
        """
        modelfile_path = Path(modelfile_path)
        if not modelfile_path.exists():
            raise OllamaError(f"Modelfile not found: {modelfile_path}")

        # Read Modelfile content
        modelfile_content = modelfile_path.read_text()

        # The Modelfile references relative paths, so we need to use ollama CLI
        # which handles this better than the API for local files
        logger.info(f"Creating Ollama model: {name}")

        try:
            # Use ollama CLI for local file handling
            result = subprocess.run(
                ["ollama", "create", name, "-f", str(modelfile_path)],
                cwd=modelfile_path.parent,  # Run from Modelfile directory
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for large models
            )

            if result.returncode != 0:
                raise OllamaError(
                    f"Failed to create model: {result.stderr or result.stdout}"
                )

            logger.info(f"Model created successfully: {name}")

        except FileNotFoundError:
            # ollama CLI not found, fall back to API
            logger.warning("ollama CLI not found, using API (may not work with local files)")
            self._create_model_api(name, modelfile_content)

        except subprocess.TimeoutExpired:
            raise OllamaError("Model creation timed out (>10 minutes)")

    def _create_model_api(self, name: str, modelfile: str) -> None:
        """Create model using the API (fallback).

        Args:
            name: Model name
            modelfile: Modelfile content

        Raises:
            OllamaError: If creation fails
        """
        try:
            response = httpx.post(
                f"{self.api_url}/create",
                json={"name": name, "modelfile": modelfile},
                timeout=600.0,
            )
            response.raise_for_status()
        except httpx.RequestError as e:
            raise OllamaError(f"Failed to create model via API: {e}") from e

    def delete_model(self, name: str) -> None:
        """Delete a model from Ollama.

        Args:
            name: Model name

        Raises:
            OllamaError: If deletion fails
        """
        try:
            response = httpx.delete(
                f"{self.api_url}/delete",
                json={"name": name},
                timeout=30.0,
            )
            response.raise_for_status()
            logger.info(f"Model deleted: {name}")
        except httpx.RequestError as e:
            raise OllamaError(f"Failed to delete model: {e}") from e

    def get_model_info(self, name: str) -> Optional[dict[str, Any]]:
        """Get information about a model.

        Args:
            name: Model name

        Returns:
            Model information dictionary, or None if not found
        """
        try:
            response = httpx.post(
                f"{self.api_url}/show",
                json={"name": name},
                timeout=10.0,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except httpx.RequestError:
            return None

    def pull_model(self, name: str) -> None:
        """Pull a model from the Ollama library.

        Args:
            name: Model name

        Raises:
            OllamaError: If pull fails
        """
        logger.info(f"Pulling model: {name}")
        try:
            # Use streaming to show progress
            with httpx.stream(
                "POST",
                f"{self.api_url}/pull",
                json={"name": name},
                timeout=None,  # No timeout for large downloads
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        # Could parse JSON progress here
                        pass
            logger.info(f"Model pulled: {name}")
        except httpx.RequestError as e:
            raise OllamaError(f"Failed to pull model: {e}") from e
