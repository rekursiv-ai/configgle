"""Export-root pytest marker rollups for configgle.

Binds the resource-marker hook at the repository root so it reaches tests
living OUTSIDE the ``configgle`` package -- ``bin/`` in particular. A conftest
hook applies to its own directory and below, so a package-level conftest alone
would leave a ``bin`` test's resource marker with no timeout, no CI skip
policy, and no error to say so.
"""

from configgle.lib.testing.resource_markers import pytest_collection_modifyitems


__all__ = ["pytest_collection_modifyitems"]
