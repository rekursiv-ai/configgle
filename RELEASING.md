# Releasing to PyPI

1. Increment version in `pyproject.toml` and `git push` it.

2. Build and upload:
   ```bash
   uv sync --group publish && uv run python -m build && uv run twine upload dist/*
   ```

3. Create GitHub release:
   ```bash
   gh release create v1.1.14 --title "v1.1.14" --notes ""
   ```

4. Clean up build artifacts:
   ```bash
   rm -rf dist/
   ```

5. Verify successful pypi push: https://pypi.org/simple/configgle/
   (or https://pypi.org/project/configgle/)
   ```bash
   pip index versions configgle
   ```
