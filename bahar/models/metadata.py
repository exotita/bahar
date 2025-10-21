"""
Model metadata definitions for the universal model loader.

Stores comprehensive information about loaded models.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


@dataclass
class ModelMetadata:
    """
    Comprehensive metadata for a machine learning model.

    Attributes:
        model_id: HuggingFace model identifier (e.g., "bert-base-uncased")
        name: Human-readable display name
        description: Detailed description of the model
        task_type: Type of task (e.g., "text-classification", "token-classification")
        language: Supported language(s)
        num_labels: Number of output labels
        label_map: Mapping from label index to label name
        taxonomy: Taxonomy type (e.g., "goemotions", "sentiment", "custom")
        added_date: When the model was added to the registry
        last_used: Last time the model was used
        use_count: Number of times the model has been used
        tags: Searchable tags for categorization
        custom_config: Additional custom configuration
        is_active: Whether the model is currently active
        version: Model version (if applicable)
    """

    model_id: str
    name: str
    description: str
    task_type: str
    language: str | list[str]
    num_labels: int
    label_map: dict[int, str]
    taxonomy: str
    added_date: datetime = field(default_factory=datetime.now)
    last_used: datetime | None = None
    use_count: int = 0
    tags: list[str] = field(default_factory=list)
    custom_config: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary format."""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        data["added_date"] = self.added_date.isoformat()
        if self.last_used:
            data["last_used"] = self.last_used.isoformat()
        else:
            data["last_used"] = None
        # Convert label_map keys to strings for JSON serialization
        data["label_map"] = {str(k): v for k, v in self.label_map.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Create metadata from dictionary."""
        # Convert ISO format strings back to datetime
        if isinstance(data.get("added_date"), str):
            data["added_date"] = datetime.fromisoformat(data["added_date"])
        if data.get("last_used") and isinstance(data["last_used"], str):
            data["last_used"] = datetime.fromisoformat(data["last_used"])
        # Convert label_map keys back to integers
        if "label_map" in data:
            data["label_map"] = {int(k): v for k, v in data["label_map"].items()}
        return cls(**data)

    def update_usage(self) -> None:
        """Update usage statistics."""
        self.last_used = datetime.now()
        self.use_count += 1

    def matches_query(self, query: str) -> bool:
        """Check if metadata matches a search query."""
        query_lower = query.lower()
        return (
            query_lower in self.model_id.lower()
            or query_lower in self.name.lower()
            or query_lower in self.description.lower()
            or any(query_lower in tag.lower() for tag in self.tags)
        )

    def __repr__(self) -> str:
        return (
            f"ModelMetadata(model_id='{self.model_id}', "
            f"name='{self.name}', task_type='{self.task_type}', "
            f"num_labels={self.num_labels}, use_count={self.use_count})"
        )

