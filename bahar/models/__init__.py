"""
Universal model loader system for dynamic HuggingFace model integration.

This module provides a complete system for loading, managing, and using
any HuggingFace model dynamically without code changes.
"""

from __future__ import annotations

from bahar.models.adapter import UniversalAdapter
from bahar.models.capabilities import ModelCapabilities, TASK_TYPES, LABEL_TYPES
from bahar.models.inspector import ModelInspector
from bahar.models.loader import UniversalModelLoader
from bahar.models.metadata import ModelMetadata
from bahar.models.registry import ModelRegistry
from bahar.models.result import UniversalResult

__all__ = [
    "ModelMetadata",
    "ModelRegistry",
    "ModelCapabilities",
    "UniversalModelLoader",
    "ModelInspector",
    "UniversalAdapter",
    "UniversalResult",
    "TASK_TYPES",
    "LABEL_TYPES",
]

