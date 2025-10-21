"""
Model registry for managing model configurations.

Provides CRUD operations and persistence for model metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from bahar.models.metadata import ModelMetadata


DEFAULT_REGISTRY_PATH: Final[Path] = Path("config/models_registry.json")


class ModelRegistry:
    """
    Registry for managing model metadata with JSON persistence.

    Provides operations to add, remove, search, and manage models.
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        """
        Initialize the model registry.

        Args:
            storage_path: Path to JSON file for persistence. Defaults to config/models_registry.json
        """
        self.storage_path = storage_path or DEFAULT_REGISTRY_PATH
        self.models: dict[str, ModelMetadata] = {}
        self.load()

    def add_model(self, metadata: ModelMetadata) -> None:
        """
        Add a model to the registry.

        Args:
            metadata: Model metadata to add

        Raises:
            ValueError: If model_id already exists
        """
        if metadata.model_id in self.models:
            raise ValueError(f"Model '{metadata.model_id}' already exists in registry")

        self.models[metadata.model_id] = metadata
        self.save()

    def update_model(self, metadata: ModelMetadata) -> None:
        """
        Update an existing model in the registry.

        Args:
            metadata: Updated model metadata

        Raises:
            ValueError: If model_id doesn't exist
        """
        if metadata.model_id not in self.models:
            raise ValueError(f"Model '{metadata.model_id}' not found in registry")

        self.models[metadata.model_id] = metadata
        self.save()

    def remove_model(self, model_id: str) -> None:
        """
        Remove a model from the registry.

        Args:
            model_id: HuggingFace model identifier

        Raises:
            ValueError: If model_id doesn't exist
        """
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found in registry")

        del self.models[model_id]
        self.save()

    def update_model(self, metadata: ModelMetadata) -> None:
        """
        Update an existing model's metadata in the registry.

        Args:
            metadata: Updated model metadata

        Raises:
            ValueError: If model_id doesn't exist
        """
        if metadata.model_id not in self.models:
            raise ValueError(f"Model '{metadata.model_id}' not found in registry")

        self.models[metadata.model_id] = metadata
        self.save()

    def get_model(self, model_id: str) -> ModelMetadata | None:
        """
        Get model metadata by ID.

        Args:
            model_id: HuggingFace model identifier

        Returns:
            Model metadata if found, None otherwise
        """
        return self.models.get(model_id)

    def list_models(
        self,
        task_type: str | None = None,
        language: str | None = None,
        taxonomy: str | None = None,
        tags: list[str] | None = None,
        active_only: bool = True,
    ) -> list[ModelMetadata]:
        """
        List models with optional filtering.

        Args:
            task_type: Filter by task type
            language: Filter by language
            taxonomy: Filter by taxonomy
            tags: Filter by tags (any match)
            active_only: Only return active models

        Returns:
            List of matching model metadata
        """
        results = list(self.models.values())

        if active_only:
            results = [m for m in results if m.is_active]

        if task_type:
            results = [m for m in results if m.task_type == task_type]

        if language:
            results = [
                m for m in results
                if (isinstance(m.language, str) and m.language == language)
                or (isinstance(m.language, list) and language in m.language)
            ]

        if taxonomy:
            results = [m for m in results if m.taxonomy == taxonomy]

        if tags:
            results = [
                m for m in results
                if any(tag in m.tags for tag in tags)
            ]

        return results

    def search_models(self, query: str) -> list[ModelMetadata]:
        """
        Search models by query string.

        Searches in model_id, name, description, and tags.

        Args:
            query: Search query

        Returns:
            List of matching model metadata
        """
        return [
            metadata for metadata in self.models.values()
            if metadata.matches_query(query)
        ]

    def update_usage(self, model_id: str) -> None:
        """
        Update usage statistics for a model.

        Args:
            model_id: HuggingFace model identifier
        """
        if model_id in self.models:
            self.models[model_id].update_usage()
            self.save()

    def get_most_used(self, limit: int = 10) -> list[ModelMetadata]:
        """
        Get most frequently used models.

        Args:
            limit: Maximum number of models to return

        Returns:
            List of most used models
        """
        sorted_models = sorted(
            self.models.values(),
            key=lambda m: m.use_count,
            reverse=True
        )
        return sorted_models[:limit]

    def get_recently_used(self, limit: int = 10) -> list[ModelMetadata]:
        """
        Get recently used models.

        Args:
            limit: Maximum number of models to return

        Returns:
            List of recently used models
        """
        sorted_models = sorted(
            [m for m in self.models.values() if m.last_used],
            key=lambda m: m.last_used,  # type: ignore
            reverse=True
        )
        return sorted_models[:limit]

    def save(self) -> None:
        """Save registry to JSON file."""
        # Ensure directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        data = {
            "version": "1.0",
            "models": [metadata.to_dict() for metadata in self.models.values()]
        }

        # Write to file
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self) -> None:
        """Load registry from JSON file."""
        if not self.storage_path.exists():
            # Initialize with empty registry
            self.models = {}
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load models
            self.models = {
                model_data["model_id"]: ModelMetadata.from_dict(model_data)
                for model_data in data.get("models", [])
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If loading fails, start with empty registry
            print(f"Warning: Failed to load registry from {self.storage_path}: {e}")
            self.models = {}

    def export_to_json(self, path: Path) -> None:
        """
        Export registry to a JSON file.

        Args:
            path: Destination file path
        """
        data = {
            "version": "1.0",
            "models": [metadata.to_dict() for metadata in self.models.values()]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def import_from_json(self, path: Path, merge: bool = False) -> None:
        """
        Import models from a JSON file.

        Args:
            path: Source file path
            merge: If True, merge with existing models. If False, replace all.

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON format is invalid
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        imported_models = {
            model_data["model_id"]: ModelMetadata.from_dict(model_data)
            for model_data in data.get("models", [])
        }

        if merge:
            self.models.update(imported_models)
        else:
            self.models = imported_models

        self.save()

    def __len__(self) -> int:
        """Return number of models in registry."""
        return len(self.models)

    def __contains__(self, model_id: str) -> bool:
        """Check if model exists in registry."""
        return model_id in self.models

    def __repr__(self) -> str:
        return f"ModelRegistry(models={len(self.models)}, path='{self.storage_path}')"

