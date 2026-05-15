"""Trace data model and (forthcoming) recorder / replayer / I/O.

This subpackage holds the trace schema (`schema.py`) and, in upcoming Phase 0
commits, the Parquet I/O (`io.py`), the recorder (`recorder.py`), and the
replayer (`replayer.py`). The subpackage root re-exports nothing; consumers
import from the leaf module (e.g. `from traceaudit.trace.schema import Trace`).
"""

__all__: list[str] = []
