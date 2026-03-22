import random
from negmas.sao import SAONegotiator, SAOResponse, SAOState
from negmas import Outcome

class GroupN_Negotiator(SAONegotiator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.concession_parameter = 0.2 
        self.min_utility = 0.0 
        self.max_utility = 1.0

    def on_preferences_changed(self, changes):
        if self.ufi.reserved_value is not None:
            self.min_utility = self.ufi.reserved_value
        else:
            self.min_utility = 0.6

    def get_aspiration_level(self, relative_time: float) -> float:
        concession_rate = relative_time ** (1.0 / self.concession_parameter)
        return self.min_utility + (self.max_utility - self.min_utility) * (1.0 - concession_rate)

    def respond(self, state: SAOState, offer: Outcome) -> SAOResponse:
        offer_utility = self.ufi(offer)
        current_time = state.relative_time
        aspiration_level = self.get_aspiration_level(current_time)

        if offer_utility >= aspiration_level:
            return SAOResponse.ACCEPT_OFFER
        
        return SAOResponse.REJECT_OFFER

    def propose(self, state: SAOState) -> Outcome | None:
        current_time = state.relative_time
        aspiration_level = self.get_aspiration_level(current_time)

        acceptable_outcomes = []
        for outcome in self.nmi.discrete_outcomes():
            if self.ufi(outcome) >= aspiration_level:
                acceptable_outcomes.append(outcome)

        if acceptable_outcomes:
            return random.choice(acceptable_outcomes)
            
        return self.ufi.extreme_outcomes()[1]