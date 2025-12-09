"""
Rule Loader

Loads domain rules from various sources:
- YAML files
- Python dictionaries
- Code-defined rules
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from rewa.models import Rule, RuleConstraint, ValueConstraint, ComparisonOp


def _parse_comparison_op(op_str: str) -> ComparisonOp:
    """Parse comparison operator from string."""
    op_map = {
        "eq": ComparisonOp.EQ,
        "=": ComparisonOp.EQ,
        "==": ComparisonOp.EQ,
        "neq": ComparisonOp.NEQ,
        "!=": ComparisonOp.NEQ,
        "<>": ComparisonOp.NEQ,
        "gt": ComparisonOp.GT,
        ">": ComparisonOp.GT,
        "gte": ComparisonOp.GTE,
        ">=": ComparisonOp.GTE,
        "lt": ComparisonOp.LT,
        "<": ComparisonOp.LT,
        "lte": ComparisonOp.LTE,
        "<=": ComparisonOp.LTE,
        "in": ComparisonOp.IN,
        "not_in": ComparisonOp.NOT_IN,
        "contains": ComparisonOp.CONTAINS,
        "matches": ComparisonOp.MATCHES,
    }
    return op_map.get(op_str.lower(), ComparisonOp.EQ)


def _parse_constraint(constraint_dict: Dict[str, Any]) -> RuleConstraint:
    """Parse a constraint from dictionary."""
    path = constraint_dict["path"]
    op = _parse_comparison_op(constraint_dict.get("op", "eq"))
    value = constraint_dict.get("value")

    # Handle boolean strings
    if isinstance(value, str):
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False

    return RuleConstraint(
        property_path=path,
        constraint=ValueConstraint(op=op, value=value)
    )


def _parse_rule(rule_dict: Dict[str, Any]) -> Rule:
    """Parse a rule from dictionary."""
    rule_id = rule_dict["id"]
    name = rule_dict.get("name", rule_id)
    description = rule_dict.get("description", "")

    # Parse preconditions
    preconditions = []
    for precond in rule_dict.get("preconditions", []):
        preconditions.append(_parse_constraint(precond))

    # Parse postconditions
    postconditions = []
    for postcond in rule_dict.get("postconditions", []):
        postconditions.append(_parse_constraint(postcond))

    hardness = rule_dict.get("hardness", "hard")
    domain_tags = set(rule_dict.get("domains", []))
    explanation = rule_dict.get("explanation", "")

    return Rule(
        id=rule_id,
        name=name,
        description=description,
        preconditions=preconditions,
        postconditions=postconditions,
        hardness=hardness,
        domain_tags=domain_tags,
        explanation_template=explanation,
    )


def load_rules_from_yaml(path: str) -> List[Rule]:
    """
    Load rules from a YAML file.

    Args:
        path: Path to YAML file

    Returns:
        List of parsed rules
    """
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    rules = []
    for rule_dict in data.get("rules", []):
        rule = _parse_rule(rule_dict)
        rules.append(rule)

    return rules


def load_rules_from_directory(directory: str) -> List[Rule]:
    """
    Load all rules from YAML files in a directory.

    Args:
        directory: Path to directory containing YAML files

    Returns:
        List of all parsed rules
    """
    rules = []
    dir_path = Path(directory)

    for yaml_file in dir_path.glob("*.yaml"):
        file_rules = load_rules_from_yaml(str(yaml_file))
        rules.extend(file_rules)

    for yml_file in dir_path.glob("*.yml"):
        file_rules = load_rules_from_yaml(str(yml_file))
        rules.extend(file_rules)

    return rules


class RuleLoader:
    """
    Manages loading and caching of domain rules.
    """

    def __init__(self, rule_directories: Optional[List[str]] = None):
        """
        Initialize the rule loader.

        Args:
            rule_directories: List of directories to search for rules
        """
        self.rule_directories = rule_directories or []
        self._cache: Dict[str, List[Rule]] = {}
        self._all_rules: Optional[List[Rule]] = None

    def add_directory(self, directory: str) -> None:
        """Add a directory to search for rules."""
        self.rule_directories.append(directory)
        self._all_rules = None  # Invalidate cache

    def load_all(self) -> List[Rule]:
        """Load all rules from all directories."""
        if self._all_rules is not None:
            return self._all_rules

        all_rules = []
        for directory in self.rule_directories:
            if os.path.isdir(directory):
                rules = load_rules_from_directory(directory)
                all_rules.extend(rules)

        self._all_rules = all_rules
        return all_rules

    def load_for_domain(self, domain: str) -> List[Rule]:
        """Load rules for a specific domain."""
        if domain in self._cache:
            return self._cache[domain]

        all_rules = self.load_all()
        domain_rules = [r for r in all_rules if domain in r.domain_tags]

        self._cache[domain] = domain_rules
        return domain_rules

    def load_for_domains(self, domains: Set[str]) -> List[Rule]:
        """Load rules for multiple domains."""
        all_domain_rules = []
        seen_ids = set()

        for domain in domains:
            for rule in self.load_for_domain(domain):
                if rule.id not in seen_ids:
                    all_domain_rules.append(rule)
                    seen_ids.add(rule.id)

        return all_domain_rules

    def reload(self) -> None:
        """Force reload all rules."""
        self._cache.clear()
        self._all_rules = None
        self.load_all()


# Built-in default rules that can be used without YAML files
DEFAULT_RULES = [
    # Medical domain rules
    Rule(
        id="med_cancer_treatment_side_effects",
        name="Cancer Treatment Side Effects",
        description="Effective cancer treatments have side effects",
        preconditions=[
            RuleConstraint(
                property_path="fact.cures",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value="cancer")
            )
        ],
        postconditions=[
            RuleConstraint(
                property_path="fact.has_side_effects",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value=True)
            )
        ],
        hardness="hard",
        domain_tags={"medical", "drug", "treatment"},
        explanation_template="Any treatment that effectively cures cancer will have side effects. This is a medical reality.",
    ),

    Rule(
        id="med_fda_approval_required",
        name="FDA Approval for Sale",
        description="Drugs sold in US must be FDA approved",
        preconditions=[
            RuleConstraint(
                property_path="entity.type",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value="Drug")
            ),
            RuleConstraint(
                property_path="fact.sold_in_us",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value=True)
            )
        ],
        postconditions=[
            RuleConstraint(
                property_path="fact.fda_approved",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value=True)
            )
        ],
        hardness="hard",
        domain_tags={"medical", "drug", "regulatory"},
        explanation_template="Drugs sold in the United States must be FDA approved.",
    ),

    # Safety domain rules
    Rule(
        id="safety_toys_not_dangerous",
        name="Toys Cannot Be Dangerous",
        description="Items classified as toys must be safe",
        preconditions=[
            RuleConstraint(
                property_path="fact.is_toy",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value=True)
            )
        ],
        postconditions=[
            RuleConstraint(
                property_path="fact.dangerous",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value=False)
            )
        ],
        hardness="hard",
        domain_tags={"safety", "toy"},
        explanation_template="Toys are by definition not dangerous. A dangerous item cannot be classified as a toy.",
    ),

    Rule(
        id="safety_weapons_not_toys",
        name="Weapons Are Not Toys",
        description="Real weapons cannot be toys",
        preconditions=[
            RuleConstraint(
                property_path="entity.type",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value="Weapon")
            ),
            RuleConstraint(
                property_path="fact.is_real",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value=True)
            )
        ],
        postconditions=[
            RuleConstraint(
                property_path="fact.is_toy",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value=False)
            )
        ],
        hardness="hard",
        domain_tags={"safety", "weapon"},
        explanation_template="Real weapons cannot be classified as toys.",
    ),

    Rule(
        id="safety_weapons_are_dangerous",
        name="Weapons Are Dangerous",
        description="Weapons are inherently dangerous",
        preconditions=[
            RuleConstraint(
                property_path="entity.type",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value="Weapon")
            ),
            RuleConstraint(
                property_path="fact.is_real",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value=True)
            )
        ],
        postconditions=[
            RuleConstraint(
                property_path="fact.dangerous",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value=True)
            )
        ],
        hardness="hard",
        domain_tags={"safety", "weapon"},
        explanation_template="Real weapons are inherently dangerous.",
    ),

    # Physics domain rules
    Rule(
        id="physics_no_perpetual_motion",
        name="No Perpetual Motion",
        description="Perpetual motion is impossible",
        preconditions=[
            RuleConstraint(
                property_path="fact.perpetual_motion",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value=True)
            )
        ],
        postconditions=[
            RuleConstraint(
                property_path="fact.physically_possible",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value=False)
            )
        ],
        hardness="hard",
        domain_tags={"physics"},
        explanation_template="Perpetual motion violates the laws of thermodynamics.",
    ),

    Rule(
        id="physics_conservation_energy",
        name="Energy Conservation",
        description="Energy cannot be created from nothing",
        preconditions=[
            RuleConstraint(
                property_path="fact.creates_energy_from_nothing",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value=True)
            )
        ],
        postconditions=[
            RuleConstraint(
                property_path="fact.physically_possible",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value=False)
            )
        ],
        hardness="hard",
        domain_tags={"physics"},
        explanation_template="Creating energy from nothing violates conservation of energy.",
    ),

    # Logical domain rules
    Rule(
        id="logic_no_contradictions",
        name="No Self-Contradictions",
        description="An entity cannot have contradictory properties",
        preconditions=[
            RuleConstraint(
                property_path="fact.property_a",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value=True)
            )
        ],
        postconditions=[
            RuleConstraint(
                property_path="fact.not_property_a",
                constraint=ValueConstraint(op=ComparisonOp.EQ, value=False)
            )
        ],
        hardness="hard",
        domain_tags={"logic"},
        explanation_template="An entity cannot simultaneously have and not have a property.",
    ),
]


def get_default_rules() -> List[Rule]:
    """Get the built-in default rules."""
    return DEFAULT_RULES.copy()
