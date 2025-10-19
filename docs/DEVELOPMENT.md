# Development Guide

Complete guide for developing and extending Bahar.

## Setup Development Environment

### 1. Clone and Setup

```bash
cd /path/to/projects
git clone <repository> bahar
cd bahar
```

### 2. Python Environment

```bash
# Check Python version
python --version  # Should be 3.12+

# Create virtual environment
uv venv

# Activate
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
# Install all dependencies
uv pip install transformers torch rich

# Install development tools
uv pip install pytest ruff pyright jupyter

# Or sync from pyproject.toml
uv sync
```

### 4. Verify Installation

```bash
# Test imports
python -c "from bahar import EmotionAnalyzer, EnhancedAnalyzer; print('✓ OK')"

# Run basic demo
python main.py
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow the patterns in `.cursorrules`:
- Use type annotations
- Follow naming conventions
- Add docstrings
- Support Rich and plain text output

### 3. Test Changes

```bash
# Run linter
ruff check .

# Run type checker
pyright

# Test imports
python -c "from bahar import *"

# Run your code
python your_script.py
```

### 4. Update Documentation

- Add/update docstrings
- Update relevant files in `docs/`
- Update `CHANGELOG.md`
- Update `README.md` if needed

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add your feature"
```

**Commit Message Format:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `refactor:` - Code refactoring
- `test:` - Tests
- `chore:` - Maintenance

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

## Code Style Guide

### Type Annotations

**Always use:**
```python
from __future__ import annotations

def analyze(text: str, top_k: int = 3) -> Result:
    """Analyze text."""
    ...
```

**Modern syntax:**
```python
# Good
def process(items: list[str]) -> dict[str, int]:
    ...

def get_value() -> str | None:
    ...

# Bad (old syntax)
from typing import Optional, Dict, List

def process(items: List[str]) -> Dict[str, int]:
    ...

def get_value() -> Optional[str]:
    ...
```

### Docstrings

**Format:**
```python
def function_name(param1: str, param2: int = 10) -> Result:
    """
    One-line summary.

    Detailed description if needed.
    Can span multiple lines.

    Args:
        param1: Description of param1
        param2: Description of param2. Defaults to 10.

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is empty
        RuntimeError: When model not loaded

    Example:
        >>> result = function_name("text", 5)
        >>> print(result)
    """
```

### Error Handling

**Provide actionable errors:**
```python
# Good
try:
    import transformers
except ImportError as exc:
    raise RuntimeError(
        "transformers not installed. "
        "Install with: uv pip install transformers"
    ) from exc

# Bad
try:
    import transformers
except:
    print("Error")
```

### File Organization

```python
"""Module docstring."""

from __future__ import annotations

# 1. Standard library
import sys
from typing import Final

# 2. Third-party
from transformers import AutoModel

# 3. Local
from bahar.datasets.goemotions import GOEMOTIONS_EMOTIONS

# 4. Constants
DEFAULT_VALUE: Final[int] = 10

# 5. Classes and functions
class MyClass:
    ...
```

## Adding Features

### Add New Dataset

See `docs/guides/adding-datasets.md` for complete guide.

**Quick steps:**
1. Create `bahar/datasets/your_dataset/`
2. Implement `taxonomy.py`, `classifier.py`, `result.py`
3. Register in `EmotionAnalyzer`
4. Add documentation
5. Test

### Add New Analyzer

**Template:**
```python
# bahar/analyzers/your_analyzer.py
from __future__ import annotations

class YourAnalyzer:
    """Your analyzer description."""

    def __init__(self, param: str = "default") -> None:
        self.param = param
        self._loaded = False

    def load_model(self) -> None:
        """Load resources."""
        # Implementation
        self._loaded = True

    def analyze(self, text: str) -> YourResult:
        """Analyze text."""
        if not self._loaded:
            self.load_model()
        # Implementation
        return YourResult(...)
```

**Export:**
```python
# bahar/analyzers/__init__.py
from bahar.analyzers.your_analyzer import YourAnalyzer

__all__ = [..., "YourAnalyzer"]

# bahar/__init__.py
from bahar.analyzers import YourAnalyzer

__all__ = [..., "YourAnalyzer"]
```

### Add CLI Tool

**Create tool:**
```python
# bahar/cli/your_tool.py
#!/usr/bin/env python3
"""Your tool description."""

from __future__ import annotations

import sys

def main() -> None:
    """Main function."""
    # Implementation
    pass

if __name__ == "__main__":
    main()
```

**Create wrapper:**
```python
# your_tool.py (root)
#!/usr/bin/env python3
"""Wrapper for your tool."""

from bahar.cli.your_tool import main

if __name__ == "__main__":
    main()
```

## Testing

### Unit Tests

**Create test file:**
```python
# tests/test_your_feature.py
import pytest
from bahar import YourAnalyzer

def test_initialization():
    """Test analyzer initializes correctly."""
    analyzer = YourAnalyzer()
    assert analyzer.param == "default"

def test_analysis():
    """Test basic analysis."""
    analyzer = YourAnalyzer()
    result = analyzer.analyze("test text")
    assert result is not None
```

**Run tests:**
```bash
pytest tests/
pytest tests/test_your_feature.py  # Single file
pytest -v  # Verbose
pytest --cov=bahar  # With coverage
```

### Integration Tests

```python
def test_end_to_end():
    """Test complete workflow."""
    from bahar import EmotionAnalyzer

    analyzer = EmotionAnalyzer(dataset="goemotions")
    analyzer.load_model()

    result = analyzer.analyze("I'm happy!", top_k=3)

    assert len(result.get_top_emotions()) == 3
    assert result.get_sentiment_group() in ["positive", "negative", "ambiguous", "neutral"]
```

## Documentation

### Add New Guide

```bash
# Create file
touch docs/guides/your-guide.md

# Write content following existing guides

# Add to docs/README.md
# Update main README.md if needed
```

### Update API Docs

```bash
# Create API doc
touch docs/api/your-module.md

# Document all public classes/functions
# Include examples
# Link from docs/README.md
```

### Documentation Format

**Guide structure:**
```markdown
# Title

Brief introduction.

## Overview

What this guide covers.

## Prerequisites

What you need to know/have.

## Step-by-Step

### 1. First Step

Details...

### 2. Second Step

Details...

## Examples

Practical examples.

## Troubleshooting

Common issues and solutions.

## See Also

Links to related docs.
```

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

# Profile code
profiler = cProfile.Profile()
profiler.enable()

# Your code here
analyzer.analyze("text")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def analyze_batch(texts):
    results = []
    for text in texts:
        result = analyzer.analyze(text)
        results.append(result)
    return results
```

### Optimization Tips

1. **Reuse analyzers** - Load model once
2. **Batch processing** - Use `analyze_batch()`
3. **Lazy loading** - Load only when needed
4. **Caching** - Use `@lru_cache` for pure functions
5. **Generators** - Use for large datasets

## Debugging

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('bahar')
logger.setLevel(logging.DEBUG)
```

### Interactive Debugging

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use built-in
breakpoint()
```

### Rich Inspect

```python
from rich import inspect

# Inspect object
inspect(analyzer, methods=True)
```

## Release Process

### 1. Update Version

```python
# pyproject.toml
version = "0.3.0"

# bahar/__init__.py
__version__ = "0.3.0"
```

### 2. Update CHANGELOG

```markdown
## [0.3.0] - 2025-XX-XX

### Added
- Feature 1
- Feature 2

### Changed
- Change 1

### Fixed
- Fix 1
```

### 3. Test Everything

```bash
# Run all tests
pytest tests/

# Check linting
ruff check .

# Check types
pyright

# Test imports
python -c "from bahar import *"

# Run demos
python main.py
python classify_enhanced.py "test"
```

### 4. Build and Release

```bash
# Build package
python -m build

# Upload to PyPI (when ready)
python -m twine upload dist/*
```

## Troubleshooting

### Import Errors

```bash
# Verify package structure
python -c "import bahar; print(bahar.__file__)"

# Check PYTHONPATH
echo $PYTHONPATH

# Reinstall in development mode
pip install -e .
```

### Type Checking Issues

```bash
# Run pyright with verbose
pyright --verbose

# Check specific file
pyright bahar/your_file.py
```

### Rich Not Working

```bash
# Test Rich
python -m rich

# Check version
python -c "import rich; print(rich.__version__)"

# Fallback to plain text
# Set use_rich=False in code
```

## Best Practices

1. **Write tests first** (TDD when possible)
2. **Document as you code** (docstrings, comments)
3. **Keep functions small** (single responsibility)
4. **Use type hints** (helps catch bugs early)
5. **Handle errors gracefully** (clear messages)
6. **Support both Rich and plain text** (accessibility)
7. **Follow existing patterns** (consistency)
8. **Update CHANGELOG** (track changes)

## Resources

- **Project Structure:** `.cursorrules`
- **Adding Datasets:** `docs/guides/adding-datasets.md`
- **Rich Output:** `docs/guides/rich-output.md`
- **API Patterns:** `.cursorrules` - API Design Patterns section

## Getting Help

1. Check documentation in `docs/`
2. Review existing code for patterns
3. Check `.cursorrules` for guidelines
4. Open an issue on GitHub

---

Happy coding! 🚀

