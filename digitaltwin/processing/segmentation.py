"""Public segmentation façade.

Segmentation remains implemented by ``DataAligner`` in this migration phase;
the façade establishes the target responsibility boundary without copying the
algorithm.
"""

from digitaltwin.analysis.alignment import DataAligner

__all__ = ["DataAligner"]
