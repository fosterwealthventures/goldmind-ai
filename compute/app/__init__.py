# compute/engine/__init__.py
from .server import app, create_app  # exporting app is enough

"""
Engine package.

Import submodules explicitly, e.g.:
    from engine.financial_data_framework import FinancialDataFramework
    from engine.lstm_temporal_analysis import LSTMTemporalAnalysis
"""

__all__ = []  # keep empty; no eager re-exports to avoid side effects
