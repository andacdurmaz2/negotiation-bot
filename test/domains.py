"""
Each domain is a negmas Scenario with two LinearAdditiveUtilityFunctions
(one per party) over a discrete CartesianOutcomeSpace.

Domains are chosen to cover a range of structural properties:
    - Trade          : 3 issues, moderate conflict (zero-sum price)
    - JobOffer       : 4 issues, integrative (compatible preferences exist)
    - RealEstate     : 2 issues, high conflict (near zero-sum)
    - SoftwareLicense: 5 issues, complex space
    - SupplyChain    : 3 issues, high social-welfare potential
    - HighConflict   : 2 issues, near zero-sum, hard to agree

"""

from negmas import make_issue, Scenario
from negmas.preferences import LinearAdditiveUtilityFunction as LAU
from negmas.preferences.value_fun import TableFun
from negmas.outcomes import make_os


def _ufun(values: list[dict], weights: tuple, os_, rv: float) -> LAU:
    """Build a LinearAdditiveUtilityFunction from a list of value→score dicts."""
    return LAU(
        values=tuple(TableFun(d) for d in values),
        weights=weights,
        outcome_space=os_,
        reserved_value=rv,
    )


def build_domains() -> dict[str, Scenario]:
    """Return all benchmark domains as a dict keyed by name."""
    domains: dict[str, Scenario] = {}

    # ------------------------------------------------------------------
    # Trade — 3 issues, moderate conflict
    # Buyer wants low price + fast delivery; Seller wants the opposite.
    # ------------------------------------------------------------------
    issues = [
        make_issue(["low", "medium", "high"], name="price"),
        make_issue(["basic", "standard", "premium"], name="quality"),
        make_issue([3, 7, 14], name="delivery_days"),
    ]
    os_ = make_os(issues)
    domains["Trade"] = Scenario(
        outcome_space=os_,
        ufuns=(
            _ufun(
                [
                    {"low": 1.0, "medium": 0.5, "high": 0.0},
                    {"basic": 0.0, "standard": 0.5, "premium": 1.0},
                    {3: 1.0, 7: 0.5, 14: 0.0},
                ],
                (0.5, 0.3, 0.2),
                os_,
                rv=0.1,
            ),
            _ufun(
                [
                    {"low": 0.0, "medium": 0.5, "high": 1.0},
                    {"basic": 1.0, "standard": 0.5, "premium": 0.0},
                    {3: 0.0, 7: 0.5, 14: 1.0},
                ],
                (0.5, 0.3, 0.2),
                os_,
                rv=0.1,
            ),
        ),
        name="Trade",
    )

    # ------------------------------------------------------------------
    # JobOffer — 4 issues, integrative
    # Employer cares most about start date; Employee cares about salary.
    # Compatible preferences create mutually beneficial outcomes.
    # ------------------------------------------------------------------
    issues = [
        make_issue([40_000, 55_000, 70_000, 85_000], name="salary"),
        make_issue(["asap", "1month", "3months"], name="start_date"),
        make_issue([15, 20, 25], name="vacation_days"),
        make_issue(["office", "hybrid", "remote"], name="work_mode"),
    ]
    os_ = make_os(issues)
    domains["JobOffer"] = Scenario(
        outcome_space=os_,
        ufuns=(
            # Employee: high salary, delayed start, more vacation, remote
            _ufun(
                [
                    {40_000: 0.0, 55_000: 0.4, 70_000: 0.8, 85_000: 1.0},
                    {"asap": 0.2, "1month": 0.6, "3months": 1.0},
                    {15: 0.0, 20: 0.5, 25: 1.0},
                    {"office": 0.0, "hybrid": 0.5, "remote": 1.0},
                ],
                (0.5, 0.2, 0.15, 0.15),
                os_,
                rv=0.15,
            ),
            # Employer: low salary, ASAP start, fewer vacation days, office
            _ufun(
                [
                    {40_000: 1.0, 55_000: 0.7, 70_000: 0.3, 85_000: 0.0},
                    {"asap": 1.0, "1month": 0.5, "3months": 0.1},
                    {15: 1.0, 20: 0.5, 25: 0.0},
                    {"office": 1.0, "hybrid": 0.5, "remote": 0.0},
                ],
                (0.35, 0.4, 0.15, 0.1),
                os_,
                rv=0.15,
            ),
        ),
        name="JobOffer",
    )

    # ------------------------------------------------------------------
    # RealEstate — 2 issues, high conflict (near zero-sum on price)
    # ------------------------------------------------------------------
    issues = [
        make_issue([200_000, 250_000, 300_000, 350_000, 400_000], name="price"),
        make_issue(["none", "minor", "major"], name="repairs"),
    ]
    os_ = make_os(issues)
    domains["RealEstate"] = Scenario(
        outcome_space=os_,
        ufuns=(
            # Buyer: low price, seller covers major repairs
            _ufun(
                [
                    {
                        200_000: 1.0,
                        250_000: 0.75,
                        300_000: 0.5,
                        350_000: 0.25,
                        400_000: 0.0,
                    },
                    {"none": 0.0, "minor": 0.5, "major": 1.0},
                ],
                (0.7, 0.3),
                os_,
                rv=0.05,
            ),
            # Seller: high price, no repairs
            _ufun(
                [
                    {
                        200_000: 0.0,
                        250_000: 0.25,
                        300_000: 0.5,
                        350_000: 0.75,
                        400_000: 1.0,
                    },
                    {"none": 1.0, "minor": 0.5, "major": 0.0},
                ],
                (0.7, 0.3),
                os_,
                rv=0.05,
            ),
        ),
        name="RealEstate",
    )

    # ------------------------------------------------------------------
    # SoftwareLicense — 5 issues, larger outcome space
    # ------------------------------------------------------------------
    issues = [
        make_issue([1_000, 5_000, 10_000, 20_000], name="annual_fee"),
        make_issue([1, 3, 5, 10], name="user_seats"),
        make_issue(["basic", "standard", "enterprise"], name="support_tier"),
        make_issue([1, 2, 3], name="contract_years"),
        make_issue(["no", "yes"], name="source_code_access"),
    ]
    os_ = make_os(issues)
    domains["SoftwareLicense"] = Scenario(
        outcome_space=os_,
        ufuns=(
            # Client: low fee, many seats, enterprise support, short contract, source access
            _ufun(
                [
                    {1_000: 1.0, 5_000: 0.6, 10_000: 0.3, 20_000: 0.0},
                    {1: 0.0, 3: 0.3, 5: 0.6, 10: 1.0},
                    {"basic": 0.0, "standard": 0.5, "enterprise": 1.0},
                    {1: 1.0, 2: 0.5, 3: 0.0},
                    {"no": 0.0, "yes": 1.0},
                ],
                (0.35, 0.25, 0.2, 0.1, 0.1),
                os_,
                rv=0.1,
            ),
            # Vendor: high fee, fewer seats, basic support, long contract, no source
            _ufun(
                [
                    {1_000: 0.0, 5_000: 0.3, 10_000: 0.7, 20_000: 1.0},
                    {1: 1.0, 3: 0.7, 5: 0.4, 10: 0.0},
                    {"basic": 1.0, "standard": 0.5, "enterprise": 0.0},
                    {1: 0.0, 2: 0.5, 3: 1.0},
                    {"no": 1.0, "yes": 0.0},
                ],
                (0.4, 0.2, 0.2, 0.1, 0.1),
                os_,
                rv=0.1,
            ),
        ),
        name="SoftwareLicense",
    )

    # ------------------------------------------------------------------
    # SupplyChain — 3 issues, high social-welfare potential
    # Compatible on order quantity (both want large orders);
    # opposed on lead time and shipping.
    # ------------------------------------------------------------------
    issues = [
        make_issue([100, 500, 1000, 2000], name="order_quantity"),
        make_issue([7, 14, 21, 30], name="lead_time_days"),
        make_issue(["standard", "express", "overnight"], name="shipping"),
    ]
    os_ = make_os(issues)
    domains["SupplyChain"] = Scenario(
        outcome_space=os_,
        ufuns=(
            # Buyer: large orders, fast delivery, express shipping
            _ufun(
                [
                    {100: 0.0, 500: 0.3, 1000: 0.7, 2000: 1.0},
                    {7: 1.0, 14: 0.6, 21: 0.3, 30: 0.0},
                    {"standard": 0.0, "express": 0.6, "overnight": 1.0},
                ],
                (0.4, 0.35, 0.25),
                os_,
                rv=0.08,
            ),
            # Supplier: large orders, longer lead time, standard shipping
            _ufun(
                [
                    {100: 0.0, 500: 0.4, 1000: 0.75, 2000: 1.0},
                    {7: 0.0, 14: 0.4, 21: 0.7, 30: 1.0},
                    {"standard": 1.0, "express": 0.4, "overnight": 0.0},
                ],
                (0.5, 0.3, 0.2),
                os_,
                rv=0.08,
            ),
        ),
        name="SupplyChain",
    )

    # ------------------------------------------------------------------
    # HighConflict — 2 issues, near zero-sum, hard to agree
    # ------------------------------------------------------------------
    issues = [
        make_issue(list(range(0, 110, 10)), name="share_pct"),
        make_issue(["A", "B", "C"], name="risk_allocation"),
    ]
    os_ = make_os(issues)
    domains["HighConflict"] = Scenario(
        outcome_space=os_,
        ufuns=(
            _ufun(
                [
                    {i: i / 100 for i in range(0, 110, 10)},
                    {"A": 1.0, "B": 0.5, "C": 0.0},
                ],
                (0.8, 0.2),
                os_,
                rv=0.2,
            ),
            _ufun(
                [
                    {i: 1 - i / 100 for i in range(0, 110, 10)},
                    {"A": 0.0, "B": 0.5, "C": 1.0},
                ],
                (0.8, 0.2),
                os_,
                rv=0.2,
            ),
        ),
        name="HighConflict",
    )

    return domains
