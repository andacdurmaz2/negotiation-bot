"""
results_io.py — Saving and summarising tournament results for Group 35.

Provides two public functions:
    save_results(records, out_dir)  -> pd.DataFrame
    print_summary(df)               -> None

Derived columns added before any aggregation:
    utility_under_agreement  — our_utility only for sessions that reached a deal
                               (NaN otherwise); isolates negotiation quality from
                               failure-to-agree penalty
    egalitarian_sw           — min(our_utility, opp_utility); rewards balance
    utilitarian_sw           — alias for social_welfare (uA + uB); kept explicit

Outputs written by save_results:
    raw_results.csv              — one row per session, all derived columns included
    summary_by_opponent.csv      — mean / std / min / max per opponent
    summary_by_domain.csv        — mean / std / min / max per domain
    summary_by_role.csv          — mean / std per role (initiator / responder)
    variance_by_scenario.csv     — std / cv (coeff. of variation) per
                                   (domain × opponent) cell across repetitions
"""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from models import NegotiationResult

# ---------------------------------------------------------------------------
# Metrics included in every standard summary table
# ---------------------------------------------------------------------------
_METRICS = [
    "our_utility",
    "utility_under_agreement",  # only agreed sessions — NaN on failures
    "opp_utility",
    "utilitarian_sw",  # uA + uB  (= social_welfare)
    "egalitarian_sw",  # min(uA, uB)
    "nash_product",
    "pareto_distance",
    "agreement",
    "advantage",
]


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to the raw DataFrame in-place. Returns df."""
    # Utility conditional on agreement (NaN when no deal — keeps mean honest)
    df["utility_under_agreement"] = df["our_utility"].where(df["agreement"])

    # Explicit welfare variants
    df["utilitarian_sw"] = df["our_utility"] + df["opp_utility"]  # = social_welfare
    df["egalitarian_sw"] = df[["our_utility", "opp_utility"]].min(axis=1)

    return df


def _variance_by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (domain × opponent) cell: std and coefficient of variation (CV = std/mean)
    for our_utility, utility_under_agreement, and nash_product.
    High CV flags unstable behaviour across repetitions.
    """
    target_cols = ["our_utility", "utility_under_agreement", "nash_product"]
    grp = df.groupby(["domain", "opponent"])[target_cols]

    std = grp.std().round(4)
    mean = grp.mean().replace(0, float("nan"))  # avoid div/0
    cv = (std / mean).round(4)

    std.columns = [f"{c}_std" for c in std.columns]
    cv.columns = [f"{c}_cv" for c in cv.columns]

    return pd.concat([std, cv], axis=1).sort_index()


def save_results(
    records: list[NegotiationResult],
    out_dir: Path = Path("results"),
) -> pd.DataFrame:
    """
    Persist tournament results to CSV and return the enriched DataFrame.

    Args:
        records: flat list returned by run_tournament().
        out_dir: directory to write files into (created if absent).

    Returns:
        Enriched pd.DataFrame (raw_results.csv + derived columns).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _enrich(pd.DataFrame([asdict(r) for r in records]))

    # --- raw ---
    df.to_csv(out_dir / "raw_results.csv", index=False)

    # --- standard summaries ---
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

    # --- variance across repetitions ---
    _variance_by_scenario(df).to_csv(out_dir / "variance_by_scenario.csv")

    print(
        f"Results saved to {out_dir}/  "
        f"({len(df)} sessions, {df['agreement'].sum()} agreements)"
    )
    return df


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def print_summary(df: pd.DataFrame) -> None:
    """Print a concise human-readable tournament summary to stdout."""
    agreed = df[df["agreement"]]

    print("\n" + "=" * 72)
    print("TOURNAMENT SUMMARY — Group35_Negotiator")
    print("=" * 72)

    ov = df[
        [
            "our_utility",
            "utility_under_agreement",
            "utilitarian_sw",
            "egalitarian_sw",
            "nash_product",
            "pareto_distance",
            "agreement",
            "advantage",
        ]
    ].mean()

    print(f"\nOverall  (n={len(df)} sessions, {len(agreed)} agreements)")
    print(
        f"  Our utility              {ov['our_utility']:.3f}   "
        f"(all sessions incl. failures)"
    )
    print(
        f"  Utility under agreement  {ov['utility_under_agreement']:.3f}   "
        f"(agreed sessions only)"
    )
    print(f"  Utilitarian SW  (uA+uB)  {ov['utilitarian_sw']:.3f}")
    print(f"  Egalitarian SW  min(u,u)  {ov['egalitarian_sw']:.3f}")
    print(f"  Nash product             {ov['nash_product']:.3f}")
    print(f"  Pareto distance          {ov['pareto_distance']:.3f}")
    print(f"  Agreement rate           {ov['agreement']:.1%}")
    print(f"  Advantage (uA - uB)      {ov['advantage']:.3f}")

    print("\nBy opponent  (util | util_agreed | util_SW | egal_SW | agree | nash)")
    for opp, g in df.groupby("opponent"):
        print(
            f"  {opp:<16}"
            f"  u={g['our_utility'].mean():.3f}"
            f"  u|agree={g['utility_under_agreement'].mean():.3f}"
            f"  USW={g['utilitarian_sw'].mean():.3f}"
            f"  ESW={g['egalitarian_sw'].mean():.3f}"
            f"  agree={g['agreement'].mean():.0%}"
            f"  nash={g['nash_product'].mean():.3f}"
        )

    print("\nBy domain  (util | util_agreed | egal_SW | agree | pareto_d | std)")
    for dom, g in df.groupby("domain"):
        print(
            f"  {dom:<20}"
            f"  u={g['our_utility'].mean():.3f}"
            f"  u|agree={g['utility_under_agreement'].mean():.3f}"
            f"  ESW={g['egalitarian_sw'].mean():.3f}"
            f"  agree={g['agreement'].mean():.0%}"
            f"  pd={g['pareto_distance'].mean():.3f}"
            f"  std={g['our_utility'].std():.3f}"
        )

    print("\nVariance highlights  (domain × opponent cells, cv = std/mean)")
    var_df = _variance_by_scenario(df).reset_index()
    # Show top-5 most variable cells — exclude deterministic cells (cv == 0)
    top = var_df[var_df["our_utility_cv"] > 0].nlargest(5, "our_utility_cv")[
        ["domain", "opponent", "our_utility_std", "our_utility_cv"]
    ]
    if top.empty:
        print("  All cells have zero variance (deterministic opponents).")
    else:
        for _, row in top.iterrows():
            print(
                f"  {row['domain']:<20} vs {row['opponent']:<16}"
                f"  std={row['our_utility_std']:.3f}  cv={row['our_utility_cv']:.3f}"
            )

    print("\nBy role")
    for role, g in df.groupby("our_role"):
        print(
            f"  {role:<12}"
            f"  u={g['our_utility'].mean():.3f}"
            f"  u|agree={g['utility_under_agreement'].mean():.3f}"
            f"  agree={g['agreement'].mean():.0%}"
        )
    print("=" * 72)
