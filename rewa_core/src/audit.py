"""
FR-9: Audit Logging

Complete audit trail for every decision:
- Witness IDs
- Hull metrics
- State (Valid/Ambiguous/Impossible)
- Policy ID
- Whether Mode B was applied
- Final μ*

Key requirement: Any decision is replayable offline.
"""

import numpy as np
import json
import hashlib
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import uuid


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


@dataclass
class AuditEntry:
    """A single auditable decision entry."""
    # Unique identifiers
    entry_id: str
    session_id: str
    timestamp: str

    # Evidence
    witness_ids: List[str]
    witness_texts: List[str]
    witness_embeddings_hash: str

    # Geometry metrics
    hemisphere_exists: bool
    hull_center: List[float]
    hull_angular_radius: float
    hull_volume_proxy: float

    # Entropy
    entropy: float
    state: str  # valid/ambiguous/impossible

    # Policy
    policy_id: Optional[str]
    policy_score: Optional[float]
    mode_b_applied: bool

    # Decision
    selected_meaning: List[float]
    refusal_type: Optional[str]
    refusal_reason: Optional[str]

    # Verbalization (if applicable)
    generated_text: Optional[str]
    verbalization_drift: Optional[float]
    verbalization_verified: Optional[bool]

    # Metadata
    computation_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), cls=NumpyEncoder, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEntry':
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'AuditEntry':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class AuditLogger:
    """
    Comprehensive audit logging for Rewa decisions.

    Guarantees:
    1. Every decision is logged with full context
    2. Logs are immutable (append-only)
    3. Decisions are reproducible from log data
    """

    def __init__(
        self,
        log_dir: str = "logs",
        session_id: Optional[str] = None,
        enable_file_logging: bool = True,
        enable_memory_logging: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.enable_file_logging = enable_file_logging
        self.enable_memory_logging = enable_memory_logging

        # In-memory log for session
        self.entries: List[AuditEntry] = []

        # Create log directory
        if enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / f"session_{self.session_id}.jsonl"

    def log(
        self,
        witnesses: List[Any],  # List of Witness objects
        hemisphere_result: Any,  # HemisphereResult
        hull_result: Any,  # SphericalHullResult
        entropy_result: Any,  # EntropyResult
        policy_id: Optional[str],
        policy_score: Optional[float],
        mode_b_result: Optional[Any],  # ModeBResult
        refusal_decision: Optional[Any],  # RefusalDecision
        verbalization_result: Optional[Any],  # VerbalizationResult
        computation_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEntry:
        """
        Log a complete decision.

        Args:
            witnesses: Evidence witnesses
            hemisphere_result: Result of hemisphere check
            hull_result: Result of hull computation
            entropy_result: Result of entropy estimation
            policy_id: Applied policy ID (if any)
            policy_score: Policy score (if computed)
            mode_b_result: Result of Mode B (if applied)
            refusal_decision: Refusal decision (if any)
            verbalization_result: Verbalization check (if performed)
            computation_time_ms: Total computation time
            metadata: Additional metadata

        Returns:
            Created AuditEntry
        """
        # Generate entry ID
        entry_id = str(uuid.uuid4())[:16]

        # Hash witness embeddings for reproducibility check
        if witnesses:
            embeddings_concat = np.concatenate([w.embedding for w in witnesses])
            embeddings_hash = hashlib.sha256(
                embeddings_concat.tobytes()
            ).hexdigest()[:16]
        else:
            embeddings_hash = "empty"

        # Extract selected meaning
        selected_meaning = []
        if mode_b_result and hasattr(mode_b_result, 'selected_meaning'):
            selected_meaning = mode_b_result.selected_meaning.tolist()
        elif hull_result and hasattr(hull_result, 'center'):
            selected_meaning = hull_result.center.tolist()

        entry = AuditEntry(
            entry_id=entry_id,
            session_id=self.session_id,
            timestamp=datetime.now().isoformat(),

            # Evidence
            witness_ids=[w.id for w in witnesses] if witnesses else [],
            witness_texts=[w.text for w in witnesses] if witnesses else [],
            witness_embeddings_hash=embeddings_hash,

            # Geometry
            hemisphere_exists=hemisphere_result.exists if hemisphere_result else True,
            hull_center=hull_result.center.tolist() if hull_result and hasattr(hull_result, 'center') and hull_result.center is not None else [],
            hull_angular_radius=hull_result.angular_radius if hull_result else 0.0,
            hull_volume_proxy=hull_result.volume_proxy if hull_result else 0.0,

            # Entropy
            entropy=entropy_result.entropy if entropy_result else 0.0,
            state=entropy_result.state.value if entropy_result else "unknown",

            # Policy
            policy_id=policy_id,
            policy_score=policy_score,
            mode_b_applied=mode_b_result is not None,

            # Decision
            selected_meaning=selected_meaning,
            refusal_type=refusal_decision.type.value if refusal_decision and refusal_decision.is_refusal() else None,
            refusal_reason=refusal_decision.reason if refusal_decision and refusal_decision.is_refusal() else None,

            # Verbalization
            generated_text=None,  # Populated separately if needed
            verbalization_drift=verbalization_result.drift_distance if verbalization_result else None,
            verbalization_verified=verbalization_result.status.value == "verified" if verbalization_result else None,

            # Metadata
            computation_time_ms=computation_time_ms,
            metadata=metadata or {}
        )

        # Store in memory
        if self.enable_memory_logging:
            self.entries.append(entry)

        # Write to file
        if self.enable_file_logging:
            self._append_to_file(entry)

        return entry

    def _append_to_file(self, entry: AuditEntry):
        """Append entry to log file (JSONL format)."""
        with open(self.log_file, 'a') as f:
            f.write(entry.to_json().replace('\n', ' ') + '\n')

    def get_entry(self, entry_id: str) -> Optional[AuditEntry]:
        """Get entry by ID from memory."""
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    def get_session_entries(self) -> List[AuditEntry]:
        """Get all entries from current session."""
        return self.entries.copy()

    def load_session(self, session_id: str) -> List[AuditEntry]:
        """Load entries from a saved session."""
        log_file = self.log_dir / f"session_{session_id}.jsonl"
        if not log_file.exists():
            return []

        entries = []
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(AuditEntry.from_json(line))

        return entries

    def verify_reproducibility(
        self,
        entry: AuditEntry,
        current_embeddings_hash: str
    ) -> Dict[str, Any]:
        """
        Verify that a decision can be reproduced.

        Args:
            entry: Original audit entry
            current_embeddings_hash: Hash of current witness embeddings

        Returns:
            Verification result
        """
        return {
            "entry_id": entry.entry_id,
            "embeddings_match": entry.witness_embeddings_hash == current_embeddings_hash,
            "original_hash": entry.witness_embeddings_hash,
            "current_hash": current_embeddings_hash,
            "reproducible": entry.witness_embeddings_hash == current_embeddings_hash
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics over logged entries."""
        if not self.entries:
            return {"total_entries": 0}

        states = [e.state for e in self.entries]
        refusals = [e.refusal_type for e in self.entries if e.refusal_type]
        mode_b_count = sum(1 for e in self.entries if e.mode_b_applied)

        return {
            "total_entries": len(self.entries),
            "session_id": self.session_id,
            "state_distribution": {
                state: states.count(state) for state in set(states)
            },
            "refusal_distribution": {
                ref: refusals.count(ref) for ref in set(refusals)
            } if refusals else {},
            "mode_b_rate": mode_b_count / len(self.entries),
            "mean_entropy": np.mean([e.entropy for e in self.entries]),
            "mean_computation_time_ms": np.mean([e.computation_time_ms for e in self.entries])
        }

    def export_for_replay(self, output_path: str):
        """Export session data for offline replay."""
        export_data = {
            "session_id": self.session_id,
            "export_timestamp": datetime.now().isoformat(),
            "entries": [e.to_dict() for e in self.entries],
            "statistics": self.get_statistics()
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, cls=NumpyEncoder, indent=2)

    def generate_compliance_report(self) -> str:
        """Generate compliance report from audit log."""
        stats = self.get_statistics()

        report = f"""
REWA CORE COMPLIANCE REPORT
===========================
Session ID: {self.session_id}
Generated: {datetime.now().isoformat()}

SUMMARY
-------
Total Decisions: {stats['total_entries']}
Mode B Applications: {stats.get('mode_b_rate', 0):.1%}

DECISION STATES
---------------
"""
        for state, count in stats.get('state_distribution', {}).items():
            pct = count / stats['total_entries'] * 100
            report += f"  {state}: {count} ({pct:.1f}%)\n"

        report += "\nREFUSALS\n--------\n"
        if stats.get('refusal_distribution'):
            for ref_type, count in stats['refusal_distribution'].items():
                report += f"  {ref_type}: {count}\n"
        else:
            report += "  No refusals recorded\n"

        report += f"""
METRICS
-------
Mean Entropy: {stats.get('mean_entropy', 0):.4f}
Mean Computation Time: {stats.get('mean_computation_time_ms', 0):.2f}ms

AUDIT TRAIL
-----------
All {stats['total_entries']} decisions are logged with:
- Full witness context
- Geometry metrics
- Policy applications
- Reproducibility hashes

COMPLIANCE STATUS: {'COMPLIANT' if stats['total_entries'] > 0 else 'NO DATA'}
"""
        return report
