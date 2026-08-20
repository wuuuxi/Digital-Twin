"""Public table/result-analysis API.

Functions are re-exported from the existing pure analysis module so callers do
not need to depend on the historical filename.
"""

from .result_analysis import *  # noqa: F401,F403
