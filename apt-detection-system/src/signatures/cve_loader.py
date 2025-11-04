"""
CVE Signature Loader

Loads and manages CVE attack signatures from JSON database.
"""

import json
from typing import Dict, List, Set
from pathlib import Path


class CVESignature:
    """Represents a single CVE signature with multi-stage patterns"""

    def __init__(self, cve_id: str, data: dict):
        """
        Initialize CVE signature

        Args:
            cve_id: CVE identifier (e.g., "CVE-2021-44228")
            data: Signature data dictionary
        """
        self.cve_id = cve_id
        self.name = data.get('name', '')
        self.severity = data.get('severity', 'UNKNOWN')
        self.description = data.get('description', '')
        self.stages = data.get('stages', [])

    def get_stage_patterns(self, stage: int) -> List[str]:
        """
        Get attack patterns for a specific stage

        Args:
            stage: Stage number (1-indexed)

        Returns:
            List of attack pattern strings
        """
        for stage_data in self.stages:
            if stage_data['stage'] == stage:
                return stage_data['patterns']
        return []

    def get_stage_technique(self, stage: int) -> str:
        """
        Get MITRE ATT&CK technique ID for a stage

        Args:
            stage: Stage number

        Returns:
            MITRE technique ID (e.g., "T1190")
        """
        for stage_data in self.stages:
            if stage_data['stage'] == stage:
                return stage_data['mitre_technique']
        return ''

    def get_all_patterns(self) -> Set[str]:
        """Get all patterns across all stages"""
        patterns = set()
        for stage_data in self.stages:
            patterns.update(stage_data['patterns'])
        return patterns

    def get_num_stages(self) -> int:
        """Get total number of attack stages"""
        return len(self.stages)

    def __repr__(self) -> str:
        return f"CVESignature({self.cve_id}: {self.name}, {len(self.stages)} stages)"


class CVELoader:
    """Loads and manages CVE signatures"""

    def __init__(self, signature_path: str):
        """
        Initialize CVE loader

        Args:
            signature_path: Path to CVE signatures JSON file
        """
        self.signature_path = Path(signature_path)
        self.signatures: Dict[str, CVESignature] = {}
        self._load_signatures()

    def _load_signatures(self):
        """Load signatures from JSON file"""
        if not self.signature_path.exists():
            raise FileNotFoundError(f"Signature file not found: {self.signature_path}")

        with open(self.signature_path, 'r') as f:
            data = json.load(f)

        for cve_id, cve_data in data.items():
            self.signatures[cve_id] = CVESignature(cve_id, cve_data)

        print(f"Loaded {len(self.signatures)} CVE signatures")

    def get_signature(self, cve_id: str) -> CVESignature:
        """Get signature by CVE ID"""
        return self.signatures.get(cve_id)

    def get_all_signatures(self) -> List[CVESignature]:
        """Get all loaded signatures"""
        return list(self.signatures.values())

    def get_all_patterns(self) -> Set[str]:
        """Get all unique patterns across all CVEs"""
        patterns = set()
        for sig in self.signatures.values():
            patterns.update(sig.get_all_patterns())
        return patterns

    def find_cve_by_pattern(self, pattern: str) -> List[str]:
        """
        Find CVEs that contain a specific pattern

        Args:
            pattern: Attack pattern to search for

        Returns:
            List of CVE IDs that contain the pattern
        """
        matching_cves = []
        for cve_id, sig in self.signatures.items():
            if pattern in sig.get_all_patterns():
                matching_cves.append(cve_id)
        return matching_cves

    def get_patterns_by_stage(self) -> Dict[int, Set[str]]:
        """
        Get patterns grouped by stage number

        Returns:
            Dictionary mapping stage number to set of patterns
        """
        stage_patterns = {}
        for sig in self.signatures.values():
            for stage_data in sig.stages:
                stage = stage_data['stage']
                if stage not in stage_patterns:
                    stage_patterns[stage] = set()
                stage_patterns[stage].update(stage_data['patterns'])
        return stage_patterns

    def get_patterns_by_technique(self) -> Dict[str, Set[str]]:
        """
        Get patterns grouped by MITRE ATT&CK technique

        Returns:
            Dictionary mapping technique ID to set of patterns
        """
        technique_patterns = {}
        for sig in self.signatures.values():
            for stage_data in sig.stages:
                technique = stage_data['mitre_technique']
                if technique not in technique_patterns:
                    technique_patterns[technique] = set()
                technique_patterns[technique].update(stage_data['patterns'])
        return technique_patterns

    def get_stats(self) -> dict:
        """Get statistics about loaded signatures"""
        total_patterns = sum(len(sig.get_all_patterns()) for sig in self.signatures.values())
        total_stages = sum(sig.get_num_stages() for sig in self.signatures.values())

        return {
            'num_cves': len(self.signatures),
            'total_patterns': total_patterns,
            'total_stages': total_stages,
            'avg_patterns_per_cve': total_patterns / len(self.signatures) if self.signatures else 0,
            'avg_stages_per_cve': total_stages / len(self.signatures) if self.signatures else 0
        }

    def __len__(self) -> int:
        return len(self.signatures)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"CVELoader({stats['num_cves']} CVEs, {stats['total_patterns']} patterns)"


if __name__ == "__main__":
    # Test CVE loader
    loader = CVELoader("../../data/signatures/cve_signatures.json")

    print(f"\n{loader}")
    print(f"\nStatistics: {loader.get_stats()}")

    # Test specific CVE
    log4shell = loader.get_signature("CVE-2021-44228")
    if log4shell:
        print(f"\n{log4shell}")
        print(f"Stage 1 patterns: {log4shell.get_stage_patterns(1)[:3]}...")
        print(f"Total patterns: {len(log4shell.get_all_patterns())}")

    # Test pattern search
    pattern = "${jndi:ldap://"
    cves = loader.find_cve_by_pattern(pattern)
    print(f"\nCVEs containing '{pattern}': {cves}")

    # Patterns by technique
    by_technique = loader.get_patterns_by_technique()
    print(f"\nTechniques covered: {list(by_technique.keys())}")
