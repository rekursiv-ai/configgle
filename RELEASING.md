# Releasing to PyPI

1. Increment version in `pyproject.toml` and `git push` it.

2. Build and smoke-test the wheel:
   ```bash
   uv sync --group publish
   rm -rf dist
   uv build --out-dir dist
   wheel="$(ls dist/configgle-*.whl | head -n 1)"
   uv run --isolated --with "$wheel" python -c "import configgle; print(configgle.__file__)"
   ```

3. Upload:
   ```bash
   uv run twine upload dist/*
   ```

4. Create GitHub release:
   ```bash
   gh release create v1.1.14 --title "v1.1.14" --notes ""
   ```

5. Clean up build artifacts:
   ```bash
   rm -rf dist/
   ```

6. Verify successful pypi push: https://pypi.org/simple/configgle/
   (or https://pypi.org/project/configgle/)
   ```bash
   pip index versions configgle
   ```
