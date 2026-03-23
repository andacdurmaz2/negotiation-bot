from dataclasses import dataclass


@dataclass
class NegotiationResult:
    """One row of results for a single negotiation session."""

    domain: str
    opponent: str
    repetition: int
    our_role: str  # "initiator" | "responder"
    agreement: bool
    our_utility: float
    opp_utility: float
    social_welfare: float
    nash_product: float
    pareto_distance: float
    n_steps: int
    duration_s: float
    advantage: float
