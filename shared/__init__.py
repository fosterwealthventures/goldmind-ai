# shared/__init__.py
"""
Shared indicator pack used by both API and Compute services.

Re-exports:
- list_indicators(): metadata describing available indicators for the UI
- run_insights(...): runs the selected indicators and returns a unified result
"""

import os

# Let both services see the same version tag by default
__version__ = os.getenv("APP_VERSION", "v1.2.1-insights")

# Re-export the main entry points from the aggregator
from .aggregator import list_indicators, run_insights  # noqa: F401

__all__ = ["list_indicators", "run_insights", "__version__"]
