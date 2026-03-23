# CAI Negotiation Practical Assignment

## Project Overview

This project involves the design and implementation of an automated negotiating agent using the NegMAS framework. Our agent is designed to participate in bilateral negotiations using the Stacked Alternating Offers Protocol (SAOP). The agent aims to reach mutually beneficial agreements in multi-issue domains characterized by linear additive utility functions.

## Requirements

- **Python:** 3.10 – 3.14 (3.13 recommended).
- **NegMAS:** Version 0.15.1.post1
- **Installation:** `pip install negmas`

## Agent Overview

Our agent (`Group35_Negotiator`) follows the **BOA (Bidding / Opponent model / Acceptance)** framework, separating the negotiation strategy into three independent components.

### Acceptance Strategy — ACasp

The agent accepts an offer if and only if its utility meets or exceeds the current aspiration level at that point in time. This is the standard Aspiration-based Acceptance (ACasp) strategy:

```python
if offer_utility >= aspiration_level:
    return ACCEPT
```

### Bidding Strategy — Time-based with Opponent Modeling

The agent uses a **Boulware concession curve** to determine its aspiration level over time:

```
aspiration(t) = min_utility + (max_utility - min_utility) * (1 - t^(1/β))
```

where `β = 0.2` produces a Boulware shape — the agent stays firm near its best utility for most of the negotiation and only concedes sharply near the deadline.

At each round, the agent collects all unproposed outcomes above the current aspiration level and picks the one with the **highest estimated opponent utility** — steering proposals toward outcomes the opponent is likely to accept without sacrificing its own threshold.

If all outcomes above the aspiration threshold have already been proposed, the agent repeats its best outcome above the threshold rather than falling below it.

### Opponent Model — Frequency-based

The agent maintains a frequency count of the values the opponent has proposed for each issue. The estimated opponent utility of an outcome is:

```
opp_utility(outcome) = (1/n_issues) * Σ count(issue_i, value_i) / total_offers
```

Issues where the opponent consistently proposes the same value are inferred to be high-priority for them. This model updates every round via `respond()` and is used in `propose()` to select the most mutually beneficial candidate above the aspiration threshold.

### Key Design Decisions

| Component              | Choice                                  | Rationale                                  |
| ---------------------- | --------------------------------------- | ------------------------------------------ |
| Concession curve       | Boulware (β=0.2)                        | Stays firm, only concedes late             |
| Min utility floor      | `max(reserved_value, 0.5)`              | Prevents accepting near-zero utility deals |
| Outcome precomputation | Sorted list in `on_preferences_changed` | Avoids recomputing utilities every round   |
| No repeated proposals  | `_proposed` set                         | Forces visible concession over time        |
| Opponent model         | Frequency-based                         | Lightweight, no prior knowledge needed     |
