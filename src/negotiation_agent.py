import random
from collections import defaultdict
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

        self._opponent_counts: list[dict] = []
        self._opponent_total: int = 0

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

        # Opponent modeling setup
        self._opponent_counts = [defaultdict(int) for _ in self.nmi.issues]

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

        self._update_opponent_model(offer)

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

        candidates = [
            (util, outcome)
            for util, outcome in self._sorted_outcomes
            if util >= aspiration_level and outcome not in self._proposed
        ]

        if candidates:
            # Pick the candidate with highest estimated opponent utility
            best = max(candidates, key=lambda x: self._estimate_opponent_utility(x[1]))
            self._proposed.add(best[1])
            return best[1]

        for util, outcome in self._sorted_outcomes:
            if outcome not in self._proposed:
                self._proposed.add(outcome)
                return outcome

        return self._sorted_outcomes[0][1]

    def _update_opponent_model(self, offer: Outcome) -> None:
        """Called every time we see an opponent offer. Updates frequency counts."""
        for i, value in enumerate(offer):
            self._opponent_counts[i][value] += 1
        self._opponent_total += 1

    def _estimate_opponent_utility(self, outcome: Outcome) -> float:
        """
        Estimates how much the opponent values this outcome based on
        how often they proposed each value in each issue.

        Issues where the opponent always proposes the same value get
        higher weight — those are the issues that matter to them.
        """
        if self._opponent_total == 0:
            return 0.0

        score = 0.0
        for i, value in enumerate(outcome):
            # Frequency of this value in issue i across all opponent offers
            freq = self._opponent_counts[i][value] / self._opponent_total
            score += freq

        # Normalise by number of issues so result stays in [0, 1]
        return score / len(self.nmi.issues)
