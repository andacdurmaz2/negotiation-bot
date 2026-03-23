# old_negotiator.py
import random
from negmas.sao import SAONegotiator, SAOState
from negmas import Outcome, ResponseType


class OldNegotiator(SAONegotiator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.concession_parameter = 0.2
        self.min_utility = 0.6
        self.max_utility = 1.0

    def on_preferences_changed(self, changes):
        if self.ufun and self.ufun.reserved_value is not None:
            reserved = float(self.ufun.reserved_value)
        if reserved != float("-inf") and reserved != float("inf"):
            self.min_utility = reserved

    def get_aspiration_level(self, relative_time):
        if relative_time is None or relative_time != relative_time:
            relative_time = 0.0
        concession_rate = relative_time ** (1.0 / self.concession_parameter)
        return self.min_utility + (self.max_utility - self.min_utility) * (
            1.0 - concession_rate
        )

    def respond(self, state, source=None, **kwargs):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        if float(self.ufun(offer)) >= self.get_aspiration_level(state.relative_time):
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state, dest=None, **kwargs):
        aspiration_level = self.get_aspiration_level(state.relative_time)
        acceptable_outcomes = []
        best_outcome = None
        best_util = -1.0
        for outcome in self.nmi.discrete_outcomes():
            util = float(self.ufun(outcome))
            if util >= aspiration_level:
                acceptable_outcomes.append(outcome)
            if util > best_util:
                best_util = util
                best_outcome = outcome
        if acceptable_outcomes:
            return random.choice(acceptable_outcomes)
        return best_outcome
