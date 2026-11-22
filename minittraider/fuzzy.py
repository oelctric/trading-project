"""Reasoning core: a small Sugeno-style fuzzy inference engine.

Hard-coded rules are too rigid (IF RSI < 30 THEN buy treats RSI 29.9 and 30.1
as different worlds). Here every condition holds to a DEGREE in [0, 1], rules
fire partially, and the output is the firing-strength-weighted average of the
rule consequents. A market can be 'somewhat oversold' (0.6) and 'slightly
trending' (0.3) at the same time, and both truths pull on the answer.
"""

from __future__ import annotations


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def tri(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership: 0 at a, peaks at b, back to 0 at c."""
    if x <= a or x >= c:
        return 0.0
    return (x - a) / (b - a) if x < b else (c - x) / (c - b)


def ramp_up(x: float, a: float, b: float) -> float:
    """0 below a, rising to 1 at b and beyond."""
    if b == a:
        return 1.0 if x >= b else 0.0
    return clamp((x - a) / (b - a))


def ramp_down(x: float, a: float, b: float) -> float:
    """1 below a, falling to 0 at b and beyond."""
    return 1.0 - ramp_up(x, a, b)


def AND(*degrees: float) -> float:
    return min(degrees) if degrees else 0.0


def OR(*degrees: float) -> float:
    return max(degrees) if degrees else 0.0


def NOT(degree: float) -> float:
    return 1.0 - degree


class Rule:
    """IF <antecedent(state) -> degree> THEN output = consequent, weighted."""

    def __init__(self, name: str, antecedent, consequent: float, weight: float = 1.0):
        self.name = name
        self.antecedent = antecedent      # callable: state dict -> degree in [0,1]
        self.consequent = consequent      # crisp output this rule argues for
        self.weight = weight


class FuzzySystem:
    """Weighted-average (zero-order Sugeno) inference over a rule set."""

    def __init__(self, rules: list[Rule], default: float = 0.0):
        self.rules = rules
        self.default = default

    def infer(self, state: dict) -> tuple[float, list[tuple[str, float]]]:
        """Returns (crisp output, [(rule name, firing strength)]) for explainability."""
        num = den = 0.0
        fired = []
        for r in self.rules:
            strength = clamp(r.antecedent(state)) * r.weight
            if strength > 0.01:
                fired.append((r.name, round(strength, 2)))
            num += strength * r.consequent
            den += strength
        out = num / den if den > 1e-9 else self.default
        fired.sort(key=lambda x: -x[1])
        return out, fired
