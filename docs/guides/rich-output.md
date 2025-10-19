# Rich Output Formatting

Bahar uses the [Rich library](https://pypi.org/project/rich/) to provide beautiful, colorful terminal output with tables, panels, and progress bars.

## Overview

Rich is a Python library for rich text and beautiful formatting in the terminal. It makes CLI output more readable and visually appealing.

**Features:**
- Colored text and styling
- Tables with automatic formatting
- Progress bars
- Panels and borders
- Syntax highlighting
- Automatic word wrapping

## Installation

Rich is automatically installed with Bahar:

```bash
uv pip install bahar
# or
uv pip install rich>=14.2.0
```

## Usage

### Automatic Rich Formatting

By default, all Bahar output uses Rich formatting when available:

```python
from bahar import EmotionAnalyzer
from bahar.datasets.goemotions.result import format_emotion_output

analyzer = EmotionAnalyzer(dataset="goemotions")
analyzer.load_model()

result = analyzer.analyze("I'm so happy!", top_k=3)
format_emotion_output(result)  # Uses Rich by default
```

### Disable Rich Formatting

If you prefer plain text output:

```python
format_emotion_output(result, use_rich=False)
```

### Enhanced Analysis with Rich

```python
from bahar import EnhancedAnalyzer
from bahar.analyzers.enhanced_analyzer import format_enhanced_output

analyzer = EnhancedAnalyzer(emotion_dataset="goemotions")
analyzer.load_model()

result = analyzer.analyze("Your text here", top_k=3)
format_enhanced_output(result)  # Beautiful tables and panels
```

## Rich Output Components

### 1. Headers

```python
from bahar.utils.rich_output import print_header

print_header("My Title", "Optional subtitle")
```

Output:
```
╭─────────────────────────────╮
│ My Title                    │
│ Optional subtitle           │
╰─────────────────────────────╯
```

### 2. Sections

```python
from bahar.utils.rich_output import print_section

print_section("Analysis Results")
```

Output:
```
Analysis Results
────────────────────────────────────────────────────────────────────────────────
```

### 3. Status Messages

```python
from bahar.utils.rich_output import (
    print_success,
    print_error,
    print_warning,
    print_info,
)

print_success("Operation completed!")
print_error("Something went wrong")
print_warning("Be careful!")
print_info("Loading model...")
```

Output:
```
✓ Operation completed!
✗ Something went wrong
⚠ Be careful!
ℹ Loading model...
```

### 4. Emotion Tables

```python
from bahar.utils.rich_output import create_emotion_table

table = create_emotion_table(result)
console.print(table)
```

Output:
```
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Emotion       ┃    Score ┃ Confidence                                       ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ joy           │    0.850 │ ████████████████████████████████████████░░░░░░░░ │
│ excitement    │    0.542 │ █████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ optimism      │    0.387 │ ███████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
└───────────────┴──────────┴──────────────────────────────────────────────────┘
```

### 5. Linguistic Tables

```python
from bahar.utils.rich_output import create_linguistic_table

table = create_linguistic_table(features)
console.print(table)
```

Output:
```
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Dimension          ┃ Value         ┃ Confidence                                 ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Formality          │ formal        │ ███████████████████████████░░░░░░░░        │
│ Tone               │ friendly      │ ████████████████████████░░░░░░░░░░░        │
│ Intensity          │ medium        │ ██████████████████░░░░░░░░░░░░░░░░░        │
│ Style              │ assertive     │ ███████████████████████░░░░░░░░░░░░        │
└────────────────────┴───────────────┴────────────────────────────────────────────┘
```

### 6. Summary Panels

```python
from bahar.utils.rich_output import create_summary_panel

panel = create_summary_panel(result)
console.print(panel)
```

Output:
```
╭─────────── Summary ────────────╮
│ Primary Emotion: joy           │
│ Sentiment: positive            │
│ Formality: formal | Tone: kind │
│ Intensity: high | Style: direct│
╰────────────────────────────────╯
```

### 7. Progress Bars

```python
from bahar.utils.rich_output import create_progress_bar

with create_progress_bar("Processing") as progress:
    task = progress.add_task("Analyzing texts", total=100)
    for i in range(100):
        # Do work
        progress.update(task, advance=1)
```

Output:
```
⠋ Processing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  45%
```

## Color Schemes

### Emotion Sentiment Colors

- **Positive**: Green
- **Negative**: Red
- **Ambiguous**: Yellow
- **Neutral**: White

### Confidence Bar Colors

- **High (>0.7)**: Bright Green/Cyan
- **Medium (0.4-0.7)**: Yellow/Cyan
- **Low (<0.4)**: Dim

## Custom Rich Usage

You can use Rich directly in your code:

```python
from bahar.utils.rich_output import console
from rich.table import Table

# Create custom table
table = Table(title="My Data")
table.add_column("Name", style="cyan")
table.add_column("Value", style="green")
table.add_row("Item 1", "100")
table.add_row("Item 2", "200")

console.print(table)
```

## Console Object

The global console object is available for direct use:

```python
from bahar.utils.rich_output import console

# Print with markup
console.print("[bold red]Error:[/bold red] Something went wrong")

# Print with style
console.print("Success!", style="bold green")

# Print JSON
console.print_json('{"key": "value"}')
```

## Fallback Behavior

If Rich is not installed or `use_rich=False` is specified, Bahar automatically falls back to plain text output:

```python
# With Rich (colorful, formatted)
format_emotion_output(result, use_rich=True)

# Without Rich (plain text)
format_emotion_output(result, use_rich=False)
```

## Examples

### Basic Demo with Rich

```bash
python main.py
```

Output includes:
- Colored headers
- Formatted tables
- Status indicators
- Visual progress bars

### Enhanced Demo with Rich

```bash
python classify_enhanced.py "Your text here"
```

Output includes:
- Comprehensive panels
- Multiple tables
- Summary panels
- Color-coded sentiment

## Best Practices

1. **Use Rich by Default**: Let users enjoy beautiful output
2. **Provide Fallback**: Always support `use_rich=False`
3. **Consistent Colors**: Use the same color scheme throughout
4. **Clear Tables**: Keep tables readable with appropriate widths
5. **Visual Hierarchy**: Use panels and sections to organize output

## Disabling Rich Globally

To disable Rich for all output:

```python
# Set environment variable
import os
os.environ['BAHAR_NO_RICH'] = '1'

# Or modify code
from bahar.datasets.goemotions.result import format_emotion_output

# All calls will use plain text
format_emotion_output(result, use_rich=False)
```

## Performance

Rich adds minimal overhead:
- Formatting: <1ms per output
- Tables: <5ms for typical tables
- No impact on analysis speed

## Troubleshooting

### Rich Not Working

If Rich output doesn't appear:

1. Check installation:
   ```bash
   python -c "import rich; print(rich.__version__)"
   ```

2. Verify terminal support:
   ```bash
   python -m rich
   ```

3. Use plain text fallback:
   ```python
   format_emotion_output(result, use_rich=False)
   ```

### Colors Not Showing

- Check terminal color support
- Try a modern terminal (iTerm2, Windows Terminal, etc.)
- Set `TERM=xterm-256color`

### Unicode Issues

If you see broken characters:
- Ensure UTF-8 encoding
- Use a terminal with Unicode support
- Fallback to plain text if needed

## See Also

- [Rich Documentation](https://rich.readthedocs.io/)
- [Rich GitHub](https://github.com/Textualize/rich)
- [Bahar Utils API](../api/utils.md)

## References

- Rich library: https://pypi.org/project/rich/
- Terminal colors: https://en.wikipedia.org/wiki/ANSI_escape_code

