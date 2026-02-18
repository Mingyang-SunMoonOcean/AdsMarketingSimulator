"""Industry baseline logic: proportional rules and legacy human intervention."""

from .rule_engine import apply_proportional_rule
from .legacy_human import apply_human_intervener

__all__ = ["apply_proportional_rule", "apply_human_intervener"]
