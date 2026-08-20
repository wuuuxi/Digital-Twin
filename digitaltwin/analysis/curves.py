"""Public curve-analysis API.

The implementation remains in :mod:`curve_analysis` during the migration;
this module provides the domain-oriented import boundary used by examples.
"""

from .curve_analysis import CurveAnalyzer

__all__ = ["CurveAnalyzer"]
