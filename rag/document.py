from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DocumentChunk:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
