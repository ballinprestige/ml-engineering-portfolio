"""Test fixtures.

The package is imported as an installed distribution (`pip install -e .`) — no sys.path hacks.
The service fails fast without a registered model, so ensure one exists (the release step) before
the service tests run.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def _ensure_registered_model():
    from service import registry, train

    if not registry.has_model():
        train.train_and_register()
