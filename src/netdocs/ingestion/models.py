"""
Chunk dataclass — the common output type of all parsers/chunkers.

Keeping this in a separate module breaks the potential circular-import
chain between parsers and the chunker.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A text chunk ready for embedding and insertion into the vector store.

    Attributes:
        text:        The raw text content of the chunk.
        doc_id:      Stable, deterministic identifier for this chunk.
                     Format: ``<source_file_stem>__<index>``.
        source_file: Path to the originating file (relative to ``data/raw``).
        doc_type:    One of ``design_doc``, ``runbook``, ``ticket``, ``config``.
        metadata:    Arbitrary key-value pairs specific to the doc type.
                     All values must be JSON-serialisable scalars (str/int/float/bool).
    """

    text: str
    doc_id: str
    source_file: str
    doc_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
