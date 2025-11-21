# Orange Checker - CLAUDE.md

## Project Overview

OrangeChecker is an automated browser automation tool that:

- Logs into Orange.fr (French telecom provider)
- Detects and solves CAPTCHAs using a PyTorch ResNet18 ML model
- Handles cookie banners and form submissions
- Supports both headless and headed browser modes

## Architecture

- **main.py**: Single-file implementation with OrangeChecker class
- **model.pth**: Pre-trained PyTorch model (ResNet18) for CAPTCHA image classification
- **Dependencies**: Playwright (browser), PyTorch (ML), PIL (image processing), concurrent.futures (threading)

## Key Components

- `OrangeChecker`: Main class managing browser session and CAPTCHA solving
- `_load_model()`: Loads checkpoint with model weights and metadata
- `_solve_captcha()`: Multi-step CAPTCHA solving with retry logic
- `_predict()`: Image classification using the ML model
- `check_page()`: Main automation flow (navigate → detect CAPTCHA → solve → login)

## Important Notes

- Model loading is protected by `_model_lock` for thread safety
- CAPTCHA detection checks for specific Orange.fr domain and query parameters
- Vietnamese comments are intentional (preserve as-is)
- Device detection: automatically uses CUDA if available, falls back to CPU
- Image cache prevents reprocessing same images

## Common Commands

- Run: `python main.py` (or use `.venv\Scripts\python` if venv is active)
- Headless mode: `OrangeChecker(headless=True)`
- Custom email: `OrangeChecker(email="user@orange.fr")`

## Development Workflow

1. Always load model before calling `_predict()` - it handles lazy loading
2. Browser setup via context manager: `with OrangeChecker() as checker:`
3. Image paths must exist before preprocessing
4. Playwright locators have 60-second default timeout

## Testing Considerations

- Requires valid Orange.fr domain (not suitable for unit tests without mocking)
- CAPTCHA solving success depends on model accuracy
- Browser automation timing is environment-dependent
- Image cache should be cleared between CAPTCHA attempts

## Code Style

### Clarity Over Abstraction

- Prefer explicit, readable code over abstraction helpers
- Don't create macros or utility functions for operations that are clearer when written out directly
- Example: `(a << 16) | b` is better understood than `U16to32(a, b)` - you can see the bit manipulation
- If code is understandable as-is, don't hide it behind a helper function

### AI-Generated Code Standards

- AI-generated code (especially from Gemini/ChatGPT) tends to be verbose but correct
- Use it as a starting point but simplify if the operation is clear enough to understand directly
- Don't let AI abstractions add unnecessary layers of indirection
- Review all AI code for: portability, type safety, and whether abstractions actually improve readability

### General Guidelines

- Type hints used throughout (Python 3.10+ union syntax)
- Docstrings minimal - comments used for JavaScript injections
- Exception handling is broad but intentional for browser automation robustness
- Print statements used for debugging/logging
- Keep code simple and direct - if you understand what it does, don't wrap it in a function

### Example from this codebase

In `_get_captcha_images()`, we directly iterate and collect buttons without unnecessary helper methods:

```python
buttons = self.page.locator(".captcha_btn__1Pngd").all()
image_data = []
for idx, button in enumerate(buttons):
    try:
        image_data.append({"index": idx, "button": button})
    except Exception:
        continue
```

Rather than creating separate methods like `get_button()`, `validate_button()`, etc., we keep the logic inline where you can see exactly what's happening. The dictionary structure is explicit and clear.
