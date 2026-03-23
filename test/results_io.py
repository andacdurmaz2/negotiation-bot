"""
Provides two public functions:
    save_results(records, out_dir)  -> pd.DataFrame
    print_summary(df)               -> None

Outputs written by save_results:
    raw_results.csv         — one row per negotiation session
    summary_by_opponent.csv — mean / std / min / max per opponent
    summary_by_domain.csv   — mean / std / min / max per domain
    summary_by_role.csv     — mean / std per role (initiator / responder)
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from models import NegotiationResult

# Metrics aggregated in every summary table
_METRICS = [
    "our_utility",
    "opp_utility",
    "social_welfare",
    "nash_product",
    "pareto_distance",
    "agreement",
    "advantage",
]


def save_results(
    records: list[NegotiationResult],
    out_dir: Path = Path("results"),
) -> pd.DataFrame:
    """
    Persist tournament results to CSV and return the raw DataFrame.

    Args:
        records: flat list returned by run_tournament().
        out_dir: directory to write files into (created if absent).

    Returns:
        pd.DataFrame with one row per session (same as raw_results.csv).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in records])

    df.to_csv(out_dir / "raw_results.csv", index=False)

    (
        df.groupby("opponent")[_METRICS]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .to_csv(out_dir / "summary_by_opponent.csv")
    )

    (
        df.groupby("domain")[_METRICS]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .to_csv(out_dir / "summary_by_domain.csv")
    )

    (
        df.groupby("our_role")[_METRICS]
        .agg(["mean", "std"])
        .round(4)
        .to_csv(out_dir / "summary_by_role.csv")
    )

    print(f"Results saved to {out_dir}/")
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a concise human-readable tournament summary to stdout."""
    print("\n" + "=" * 68)
    print("TOURNAMENT SUMMARY — Group35_Negotiator")
    print("=" * 68)

    ov = df[
        [
            "our_utility",
            "social_welfare",
            "nash_product",
            "pareto_distance",
            "agreement",
            "advantage",
        ]
    ].mean()
    print(f"\nOverall  (n={len(df)} sessions)")
    print(f"  Our utility      {ov['our_utility']:.3f}")
    print(f"  Social welfare   {ov['social_welfare']:.3f}")
    print(f"  Nash product     {ov['nash_product']:.3f}")
    print(f"  Pareto distance  {ov['pareto_distance']:.3f}")
    print(f"  Agreement rate   {ov['agreement']:.1%}")
    print(f"  Advantage        {ov['advantage']:.3f}")

    print("\nBy opponent")
    for opp, g in df.groupby("opponent"):
        print(
            f"  {opp:<16} util={g['our_utility'].mean():.3f}  "
            f"agree={g['agreement'].mean():.0%}  "
            f"nash={g['nash_product'].mean():.3f}"
        )

    print("\nBy domain")
    for dom, g in df.groupby("domain"):
        print(
            f"  {dom:<20} util={g['our_utility'].mean():.3f}  "
            f"agree={g['agreement'].mean():.0%}  "
            f"pareto_d={g['pareto_distance'].mean():.3f}"
        )

    print("\nBy role")
    for role, g in df.groupby("our_role"):
        print(
            f"  {role:<12} util={g['our_utility'].mean():.3f}  "
            f"agree={g['agreement'].mean():.0%}"
        )
    print("=" * 68)
