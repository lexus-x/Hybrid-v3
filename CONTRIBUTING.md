# Contributing to Hybrid-v3

Thank you for your interest in contributing to **Hybrid-v3**! We welcome bug reports, feature requests, documentation improvements, and pull requests.

Following these guidelines helps ensure a smooth contribution process for everyone.

---

## 🛠️ Local Development Setup

To set up a local development environment, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/lexus-x/Hybrid-v3.git
   cd Hybrid-v3
   ```

2. **Create a Virtual Environment:**
   We recommend using a virtual environment (`venv` or `conda`):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   Install the required libraries:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov  # Install test dependencies
   ```

---

## 🧪 Running Unit Tests

Before submitting any code changes, please ensure all unit tests pass. We use `pytest` for testing:

```bash
# Add src/ to your PYTHONPATH and run pytest
PYTHONPATH=src pytest tests/
```

Make sure any new feature or helper has corresponding unit tests under the `tests/` directory.

---

## 🎨 Code Style Guidelines

To keep the codebase clean and maintainable:
- Follow **PEP 8** coding standards for Python.
- Use meaningful variable and function names.
- Keep functions modular and write docstrings for classes and main functions.
- Avoid committing raw credentials or keys.

---

## 📬 Submitting a Pull Request

1. Fork the repository and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes with clear and descriptive commit messages.
3. Push your branch to GitHub:
   ```bash
   git push origin feature/your-feature-name
   ```
4. Open a Pull Request (PR) describing:
   - What changes were made.
   - Why they are needed.
   - How you verified and tested the changes.
