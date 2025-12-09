from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class ChunkVerification:
    """
    Verification result for a single text chunk.
    """
    text: str
    is_consistent: bool
    consistency_score: float = 1.0
    modifier_score: float = 1.0
    flags: List[str] = field(default_factory=list)  # e.g. ["CONTRADICTION", "BAD_BINDING"]
    explanation: str = ""
    
@dataclass
class VerificationResult:
    """
    Aggregate verification result for a user query and retrieved context.
    """
    query: str
    verified_chunks: List[ChunkVerification]
    overall_confidence: float
    global_warnings: List[str] = field(default_factory=list)
    
    @property
    def has_high_confidence(self) -> bool:
        return self.overall_confidence > 0.8
    
    def get_clean_text(self) -> str:
        """Returns verified chunks joined by newlines."""
        valid_chunks = [c.text for c in self.verified_chunks if c.is_consistent]
        return "\n\n".join(valid_chunks)
