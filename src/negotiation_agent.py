import random
from negmas.sao import SAONegotiator, SAOState
from negmas import Outcome, ResponseType


class Group35_Negotiator(SAONegotiator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # BOA Parameters for your report
        self.concession_parameter = 0.2
        self.min_utility = 0.6
        self.max_utility = 1.0

        self._sorted_outcomes: list[tuple[float, Outcome]] = []
        self._proposed: set = set()

    def on_preferences_changed(self, changes):
        if self.ufun and self.ufun.reserved_value is not None:
            reserved = float(self.ufun.reserved_value)
        # Only use it if it's a sensible value
        if reserved != float("-inf") and reserved != float("inf"):
            self.min_utility = reserved

        # Sort the outcomes by utility for efficient proposal generation
        outcomes = list(self.nmi.discrete_outcomes())
        self._sorted_outcomes = sorted(
            ((float(self.ufun(o)), o) for o in outcomes),
            key=lambda x: x[0],
            reverse=True,  # best outcomes first
        )

    def get_aspiration_level(self, relative_time: float) -> float:
        """Calculates target utility using a Boulware curve (Time-based)."""

        if relative_time is None or relative_time != relative_time:  # nan check
            relative_time = 0.0

        concession_rate = relative_time ** (1.0 / self.concession_parameter)
        return self.min_utility + (self.max_utility - self.min_utility) * (
            1.0 - concession_rate
        )

    def respond(
        self, state: SAOState, source: str | None = None, **kwargs
    ) -> ResponseType:
        """Acceptance Strategy: Aspiration-based Acceptance (ACasp)."""
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.ufun(offer))
        aspiration_level = self.get_aspiration_level(state.relative_time)

        if offer_utility >= aspiration_level:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def propose(
        self, state: SAOState, dest: str | None = None, **kwargs
    ) -> Outcome | None:
        """Bidding Strategy: Propose an outcome that meets our target utility."""

        if not self._sorted_outcomes:
            return None

        aspiration_level = self.get_aspiration_level(state.relative_time)

        for util, outcome in self._sorted_outcomes:
            if util >= aspiration_level and outcome not in self._proposed:
                self._proposed.add(outcome)
                return outcome

        for util, outcome in self._sorted_outcomes:
            if outcome not in self._proposed:
                self._proposed.add(outcome)
                return outcome

        return self._sorted_outcomes[0][1]
