"""
plot_representative_sessions.py — Plot one representative negotiation per strategy family.

Selection logic:
  For each (opponent_family, domain) pair, find the session whose our_utility
  is closest to the family mean — i.e. the most typical, not the best or worst.

Produces one interactive HTML file per session in results/plots/.
Saves one PNG per session in results/plots/.
utility space, Pareto/Nash/Kalai markers, agreement annotation).

Usage:
    python plot_representative_sessions.py
"""

import random
import sys
import warnings
from pathlib import Path

import pandas as pd
import plotly.io as pio

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.negotiation_agent import Group35_Negotiator
from domains import build_domains
from negmas.sao.mechanism import SAOMechanism
from negmas.sao.negotiators import (
    BoulwareTBNegotiator,
    ConcederTBNegotiator,
    MiCRONegotiator,
    ToughNegotiator,
    NaiveTitForTatNegotiator,
)

# ---------------------------------------------------------------------------
# One entry per strategy family:
#   (label, opponent_class, opponent_name_in_csv, domain, story)
#
# Domain is chosen to make the story most legible:
#   - Trade for time-based (moderate conflict, clean 3-issue space)
#   - JobOffer for adaptive/Pareto (integrative, so interesting utility paths)
#   - SupplyChain for self-play (compatible prefs → agreement possible)
# ---------------------------------------------------------------------------
FAMILIES = [
    (
        "01_time_based_slow",
        BoulwareTBNegotiator,
        "Boulware",
        "Trade",
        "Time-based slow (Boulware): both agents hold firm then converge near deadline",
    ),
    (
        "02_time_based_fast",
        ConcederTBNegotiator,
        "Conceder",
        "Trade",
        "Time-based fast (Conceder): opponent concedes quickly, we exploit it",
    ),
    (
        "03_pareto_tough",
        MiCRONegotiator,
        "MiCRO",
        "JobOffer",
        "Pareto-aware (MiCRO): neither agent concedes enough — deadlock",
    ),
    (
        "04_hardheaded",
        ToughNegotiator,
        "Tough",
        "Trade",
        "Hardheaded (Tough): opponent never moves — guaranteed deadlock",
    ),
    (
        "05_adaptive",
        NaiveTitForTatNegotiator,
        "NaiveTitForTat",
        "JobOffer",
        "Adaptive (NaiveTitForTat): opponent mirrors our concession rate",
    ),
]


def _find_representative_seed(
    results_csv: Path,
    opp_name: str,
    domain_name: str,
) -> tuple[int, str]:
    """
    Return (repetition, role) for the session closest to the family mean utility.
    Falls back to rep=0, role='initiator' if CSV not found.
    """
    if not results_csv.exists():
        return 0, "initiator"

    df = pd.read_csv(results_csv)
    grp = df[(df["opponent"] == opp_name) & (df["domain"] == domain_name)]
    if grp.empty:
        return 0, "initiator"

    mean_u = grp["our_utility"].mean()
    idx = (grp["our_utility"] - mean_u).abs().idxmin()
    row = grp.loc[idx]
    return int(row["repetition"]), str(row["our_role"])


def _run_and_plot(
    label: str,
    opp_cls,
    opp_name: str,
    domain_name: str,
    story: str,
    domains: dict,
    results_csv: Path,
    out_dir: Path,
) -> None:
    scenario = domains[domain_name]
    os_ = scenario.outcome_space
    ufun_ours = scenario.ufuns[0]
    ufun_opp = scenario.ufuns[1]

    rep, role = _find_representative_seed(results_csv, opp_name, domain_name)
    seed = hash((domain_name, opp_name, role, rep)) % (2**31)

    random.seed(seed)
    session = SAOMechanism(outcome_space=os_, n_steps=100)

    if role == "initiator":
        session.add(Group35_Negotiator(name="Group35"), ufun=ufun_ours)
        session.add(opp_cls(name=opp_name), ufun=ufun_opp)
    else:
        session.add(opp_cls(name=opp_name), ufun=ufun_opp)
        session.add(Group35_Negotiator(name="Group35"), ufun=ufun_ours)

    result = session.run()

    agreed = result.agreement is not None
    our_util = (
        float(ufun_ours(result.agreement))
        if agreed
        else float(ufun_ours.reserved_value)
    )
    status = f"AGREED  u={our_util:.3f}" if agreed else "NO DEAL"

    print(
        f"  {label}  |  {opp_name:<16} vs {domain_name:<18}  "
        f"rep={rep} {role:<10}  {status}"
    )

    fig = session.plot(show=False)
    title = (
        f"{label.replace('_',' ').title()} — {opp_name} vs Group35 on {domain_name}<br>"
    )
    title += f"<sup>{story} | rep={rep} role={role} | {status}</sup>"
    fig.update_layout(title_text=title, title_font_size=13)

    png_path = out_dir / f"{label}.png"
    pio.write_image(fig, str(png_path), format="png", width=1400, height=900, scale=2)


def main() -> None:
    results_csv = Path("results/raw_results.csv")
    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = build_domains()

    print(f"Generating {len(FAMILIES)} representative session plots → {out_dir}/\n")
    for label, opp_cls, opp_name, domain_name, story in FAMILIES:
        _run_and_plot(
            label,
            opp_cls,
            opp_name,
            domain_name,
            story,
            domains,
            results_csv,
            out_dir,
        )

    print(f"\nDone. Open any file in results/plots/ in a browser.")
    print("Files:")
    for f in sorted(out_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
