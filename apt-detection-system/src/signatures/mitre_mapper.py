"""
MITRE ATT&CK Technique Mapper

Maps CVE attack stages to MITRE ATT&CK framework techniques.
"""

from typing import Dict, List


class MITRETechnique:
    """Represents a MITRE ATT&CK technique"""

    def __init__(self, technique_id: str, name: str, tactic: str, description: str = ""):
        """
        Initialize MITRE technique

        Args:
            technique_id: Technique ID (e.g., "T1190")
            name: Technique name
            tactic: MITRE tactic (e.g., "Initial Access")
            description: Technique description
        """
        self.technique_id = technique_id
        self.name = name
        self.tactic = tactic
        self.description = description

    def __repr__(self) -> str:
        return f"{self.technique_id}: {self.name} ({self.tactic})"


class MITREMapper:
    """Maps attack patterns to MITRE ATT&CK framework"""

    # Common MITRE techniques used in CVE exploitation
    TECHNIQUES = {
        'T1190': MITRETechnique(
            'T1190',
            'Exploit Public-Facing Application',
            'Initial Access',
            'Exploiting internet-facing applications to gain initial access'
        ),
        'T1059': MITRETechnique(
            'T1059',
            'Command and Scripting Interpreter',
            'Execution',
            'Execute commands via shell, PowerShell, Python, etc.'
        ),
        'T1053': MITRETechnique(
            'T1053',
            'Scheduled Task/Job',
            'Persistence',
            'Create scheduled tasks for persistence'
        ),
        'T1041': MITRETechnique(
            'T1041',
            'Exfiltration Over C2 Channel',
            'Exfiltration',
            'Exfiltrate data over existing C2 channel'
        ),
        'T1210': MITRETechnique(
            'T1210',
            'Exploitation of Remote Services',
            'Lateral Movement',
            'Exploit remote services to move laterally'
        ),
        'T1486': MITRETechnique(
            'T1486',
            'Data Encrypted for Impact',
            'Impact',
            'Encrypt data to disrupt operations (ransomware)'
        ),
        'T1021.001': MITRETechnique(
            'T1021.001',
            'Remote Desktop Protocol',
            'Lateral Movement',
            'Use RDP for lateral movement'
        ),
        'T1068': MITRETechnique(
            'T1068',
            'Exploitation for Privilege Escalation',
            'Privilege Escalation',
            'Exploit vulnerabilities to escalate privileges'
        ),
        'T1003.006': MITRETechnique(
            'T1003.006',
            'DCSync',
            'Credential Access',
            'Replicate directory data to obtain credentials'
        ),
        'T1078.002': MITRETechnique(
            'T1078.002',
            'Domain Accounts',
            'Persistence',
            'Use domain accounts to maintain persistence'
        ),
        'T1505.003': MITRETechnique(
            'T1505.003',
            'Web Shell',
            'Persistence',
            'Install web shell for persistent access'
        ),
        'T1003.001': MITRETechnique(
            'T1003.001',
            'LSASS Memory',
            'Credential Access',
            'Dump LSASS process memory to obtain credentials'
        ),
        'T1552.001': MITRETechnique(
            'T1552.001',
            'Credentials In Files',
            'Credential Access',
            'Extract credentials from files'
        ),
        'T1078': MITRETechnique(
            'T1078',
            'Valid Accounts',
            'Initial Access',
            'Use stolen valid credentials for access'
        )
    }

    def __init__(self):
        """Initialize MITRE mapper"""
        pass

    def get_technique(self, technique_id: str) -> MITRETechnique:
        """
        Get MITRE technique by ID

        Args:
            technique_id: Technique ID (e.g., "T1190")

        Returns:
            MITRETechnique object or None
        """
        return self.TECHNIQUES.get(technique_id)

    def get_tactic(self, technique_id: str) -> str:
        """
        Get tactic name for a technique

        Args:
            technique_id: Technique ID

        Returns:
            Tactic name (e.g., "Initial Access")
        """
        technique = self.get_technique(technique_id)
        return technique.tactic if technique else 'Unknown'

    def get_attack_chain_summary(self, technique_ids: List[str]) -> str:
        """
        Generate human-readable attack chain summary

        Args:
            technique_ids: List of technique IDs in order

        Returns:
            Summary string
        """
        if not technique_ids:
            return "No attack chain detected"

        tactics = []
        for tid in technique_ids:
            technique = self.get_technique(tid)
            if technique:
                tactics.append(f"{technique.tactic} ({technique.name})")
            else:
                tactics.append(f"Unknown ({tid})")

        return " → ".join(tactics)

    def classify_severity(self, technique_ids: List[str]) -> str:
        """
        Classify attack severity based on technique progression

        Args:
            technique_ids: List of technique IDs

        Returns:
            Severity level: LOW, MEDIUM, HIGH, CRITICAL
        """
        if not technique_ids:
            return 'LOW'

        num_stages = len(technique_ids)

        # Check for high-impact techniques
        high_impact = {'T1486', 'T1003.006', 'T1003.001', 'T1041'}
        has_high_impact = any(tid in high_impact for tid in technique_ids)

        # Check for full kill chain (Initial Access → Execution → Persistence)
        tactics = [self.get_tactic(tid) for tid in technique_ids]
        has_initial_access = 'Initial Access' in tactics
        has_execution = 'Execution' in tactics
        has_persistence = 'Persistence' in tactics

        if num_stages >= 3 and has_initial_access and has_execution and has_persistence:
            return 'CRITICAL'
        elif num_stages >= 2 and (has_high_impact or has_persistence):
            return 'HIGH'
        elif num_stages >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'

    def get_recommended_response(self, technique_ids: List[str]) -> str:
        """
        Get recommended response action based on techniques

        Args:
            technique_ids: List of technique IDs

        Returns:
            Recommended response action
        """
        severity = self.classify_severity(technique_ids)

        responses = {
            'CRITICAL': 'IMMEDIATE RESPONSE REQUIRED: Isolate affected systems, engage incident response team, preserve forensics',
            'HIGH': 'URGENT: Investigate immediately, consider isolating affected systems, alert security team',
            'MEDIUM': 'Investigate within 4 hours, monitor for escalation, review logs for related activity',
            'LOW': 'Monitor and investigate within 24 hours, may be false positive or reconnaissance'
        }

        return responses.get(severity, 'Unknown severity')

    def get_all_techniques(self) -> Dict[str, MITRETechnique]:
        """Get all available techniques"""
        return self.TECHNIQUES.copy()


if __name__ == "__main__":
    # Test MITRE mapper
    mapper = MITREMapper()

    # Test single technique
    technique = mapper.get_technique('T1190')
    print(f"Technique: {technique}")

    # Test attack chain
    chain = ['T1190', 'T1059', 'T1053', 'T1041']
    print(f"\nAttack Chain: {chain}")
    print(f"Summary: {mapper.get_attack_chain_summary(chain)}")
    print(f"Severity: {mapper.classify_severity(chain)}")
    print(f"Response: {mapper.get_recommended_response(chain)}")

    # Test different scenarios
    print("\n" + "="*60)
    print("Scenario 1: Single reconnaissance")
    chain1 = ['T1190']
    print(f"Severity: {mapper.classify_severity(chain1)}")

    print("\nScenario 2: Initial access + execution")
    chain2 = ['T1190', 'T1059']
    print(f"Severity: {mapper.classify_severity(chain2)}")

    print("\nScenario 3: Full APT campaign")
    chain3 = ['T1190', 'T1059', 'T1053', 'T1003.001', 'T1041']
    print(f"Severity: {mapper.classify_severity(chain3)}")
    print(f"Chain: {mapper.get_attack_chain_summary(chain3)}")
