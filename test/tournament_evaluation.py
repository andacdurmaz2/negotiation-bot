"""
tournament_runner.py — Group 35 Negotiation Agent: Tournament Evaluation
=========================================================================
Runs structured tournaments across multiple domains and opponents and
computes per-session negotiation-quality metrics.

Metrics computed:
    our_utility     — utility achieved by Group35_Negotiator
    opp_utility     — utility achieved by the opponent
    social_welfare  — our_utility + opp_utility
    nash_product    — sqrt((uA - rvA) * (uB - rvB))
    pareto_distance — Euclidean distance to nearest Pareto-frontier point
    advantage       — our_utility - opp_utility
    agreement       — whether a deal was reached
    n_steps         — negotiation rounds used

Usage (edit CONFIG at the bottom, then run):
    python tournament_runner.py
"""

import math
import time
import random
import sys
from pathlib import Path
from typing import Type

from negmas.sao.mechanism import SAOMechanism
from negmas.sao import SAONegotiator
from negmas.inout import pareto_frontier

from negmas.sao.negotiators import (
    BoulwareTBNegotiator,  # time-based, slow concession  (e < 1)
    LinearTBNegotiator,  # time-based, linear concession (e = 1)
    ConcederTBNegotiator,  # time-based, fast concession  (e > 1)
    MiCRONegotiator,  # Pareto-frontier-aware, minimal concession
    ToughNegotiator,  # hardheaded: proposes only its single best outcome
    NaiveTitForTatNegotiator,  # reactive: mirrors opponent concession rate
    CABNegotiator,  # concede-and-build hybrid
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.negotiation_agent import Group35_Negotiator
from domains import build_domains
from models import NegotiationResult
from results_io import save_results, print_summary

# ---------------------------------------------------------------------------
# Opponent set — one per strategy family
#
#   Family              Agent              Rationale
#   ──────────────────  ─────────────────  ──────────────────────────────────
#   Time-based slow     Boulware           Canonical tough TB; same family as ours
#   Time-based moderate Linear             Baseline e=1; spans the TB spectrum
#   Time-based fast     Conceder           Tests whether we exploit fast concessions
#   Pareto-aware        MiCRO              Exposes our deadline/floor weakness
#   Hardheaded          Tough              Pure deadlock stress test
#   Reactive/adaptive   NaiveTitForTat     Only opponent that responds to our moves
#   Concede-and-build   CAB                Hybrid; concedes fast on low-weight issues
#   Self-play           Self               Internal consistency / symmetry check
# ---------------------------------------------------------------------------
OPPONENTS: dict[str, Type[SAONegotiator]] = {
    "Boulware": BoulwareTBNegotiator,
    "Linear": LinearTBNegotiator,
    "Conceder": ConcederTBNegotiator,
    "MiCRO": MiCRONegotiator,
    "Tough": ToughNegotiator,
    "NaiveTitForTat": NaiveTitForTatNegotiator,
    "CAB": CABNegotiator,
    "Self": Group35_Negotiator,
}


# ---------------------------------------------------------------------------
# Core session runner
# ---------------------------------------------------------------------------


def _pareto_distance(uA: float, uB: float, frontier: list[tuple[float, ...]]) -> float:
    if not frontier:
        return float("nan")
    return min(math.sqrt((uA - p[0]) ** 2 + (uB - p[1]) ** 2) for p in frontier)


def _run_session(
    our_cls: Type[SAONegotiator],
    opp_cls: Type[SAONegotiator],
    ufun_ours,
    ufun_opp,
    os_,
    n_steps: int,
    our_role: str,
    seed: int,
) -> dict:
    random.seed(seed)
    session = SAOMechanism(outcome_space=os_, n_steps=n_steps)

    if our_role == "initiator":
        session.add(our_cls(), ufun=ufun_ours)
        session.add(opp_cls(), ufun=ufun_opp)
    else:
        session.add(opp_cls(), ufun=ufun_opp)
        session.add(our_cls(), ufun=ufun_ours)

    t0 = time.perf_counter()
    result = session.run()
    elapsed = time.perf_counter() - t0

    if result.agreement:
        u_ours = float(ufun_ours(result.agreement))
        u_opp = float(ufun_opp(result.agreement))
    else:
        u_ours = float(ufun_ours.reserved_value)
        u_opp = float(ufun_opp.reserved_value)

    return {
        "agreement": result.agreement is not None,
        "our_utility": u_ours,
        "opp_utility": u_opp,
        "n_steps": result.step,
        "duration_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------


def run_tournament(
    domains: dict,
    opponents: dict[str, Type[SAONegotiator]],
    n_repetitions: int = 5,
    n_steps: int = 100,
    roles: tuple[str, ...] = ("initiator", "responder"),
    verbose: bool = True,
) -> list[NegotiationResult]:
    """
    Run the full tournament. Returns a flat list of NegotiationResult objects,
    one per (domain x opponent x role x repetition) combination.
    """
    records: list[NegotiationResult] = []
    total = len(domains) * len(opponents) * len(roles) * n_repetitions
    done = 0

    for domain_name, scenario in domains.items():
        os_ = scenario.outcome_space
        ufun_ours = scenario.ufuns[0]
        ufun_opp = scenario.ufuns[1]

        # Pre-compute Pareto frontier once per domain (reused across all sessions)
        outcomes = list(os_.enumerate_or_sample(max_cardinality=5000))
        frontier, _ = pareto_frontier([ufun_ours, ufun_opp], outcomes=outcomes)

        for opp_name, opp_cls in opponents.items():
            for role in roles:
                for rep in range(n_repetitions):
                    seed = hash((domain_name, opp_name, role, rep)) % (2**31)

                    raw = _run_session(
                        our_cls=Group35_Negotiator,
                        opp_cls=opp_cls,
                        ufun_ours=ufun_ours,
                        ufun_opp=ufun_opp,
                        os_=os_,
                        n_steps=n_steps,
                        our_role=role,
                        seed=seed,
                    )

                    uA = raw["our_utility"]
                    uB = raw["opp_utility"]
                    rvA = float(ufun_ours.reserved_value)
                    rvB = float(ufun_opp.reserved_value)
                    nash = math.sqrt(max(0.0, uA - rvA) * max(0.0, uB - rvB))
                    pdist = _pareto_distance(uA, uB, list(frontier))

                    records.append(
                        NegotiationResult(
                            domain=domain_name,
                            opponent=opp_name,
                            repetition=rep,
                            our_role=role,
                            agreement=raw["agreement"],
                            our_utility=round(uA, 4),
                            opp_utility=round(uB, 4),
                            social_welfare=round(uA + uB, 4),
                            nash_product=round(nash, 4),
                            pareto_distance=round(pdist, 4),
                            n_steps=raw["n_steps"],
                            duration_s=round(raw["duration_s"], 4),
                            advantage=round(uA - uB, 4),
                        )
                    )

                    done += 1
                    if verbose and (done % 10 == 0 or done == total):
                        print(
                            f"  [{done:>4}/{total}] {100*done/total:5.1f}%  "
                            f"{domain_name:<18} vs {opp_name:<16} {role}"
                        )

    return records


# ---------------------------------------------------------------------------
# Entry point — edit CONFIG to adjust the run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CONFIG = {
        "n_repetitions": 5,  # sessions per (domain x opponent x role)
        "n_steps": 100,  # max rounds per negotiation
        "out_dir": Path("results"),
    }

    t0 = time.perf_counter()
    records = run_tournament(
        domains=build_domains(),
        opponents=OPPONENTS,
        n_repetitions=CONFIG["n_repetitions"],
        n_steps=CONFIG["n_steps"],
    )
    elapsed = time.perf_counter() - t0

    print(f"\nDone in {elapsed:.1f}s  ({len(records)} sessions)")
    df = save_results(records, CONFIG["out_dir"])
    print_summary(df)
