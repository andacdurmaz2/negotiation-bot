import sys
import random
from pathlib import Path
import matplotlib.pyplot as plt

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from negmas.sao import SAOMechanism
from negmas.outcomes import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction

from src.negotiation_agent import Group35_Negotiator
from src.negotiator_v1 import OldNegotiator


def make_domain(seed=42):
    random.seed(seed)
    n_issues, n_values = 5, 7
    issues = [make_issue(n_values, name=f"issue_{i}") for i in range(n_issues)]

    def random_ufun():
        weights = [random.random() for _ in range(n_issues)]
        total = sum(weights)
        weights = [w / total for w in weights]
        vals = []
        for _ in issues:
            raw = [random.random() for _ in range(n_values)]
            max_v = max(raw)
            vals.append({j: raw[j] / max_v for j in range(n_values)})
        return LinearAdditiveUtilityFunction(
            values=vals, weights=weights, issues=issues, reserved_value=0.1
        )

    return issues, random_ufun(), random_ufun()


def run_test(n_domains=10):
    print(
        f"{'Seed':>5}  {'Agreement':>10}  {'New(A)':>8}  {'Old(B)':>8}  {'Welfare':>8}  {'Step':>5}"
    )
    print("-" * 58)

    new_wins, old_wins, draws, agreements = 0, 0, 0, 0

    for seed in range(n_domains):
        issues, pref_a, pref_b = make_domain(seed=seed)
        runner = SAOMechanism(issues=issues, n_steps=100)
        runner.add(Group35_Negotiator(name="Old"), ufun=pref_b)
        runner.add(Group35_Negotiator(name="New"), ufun=pref_a)
        state = runner.run()

        if state.agreement:
            ua = float(pref_a(state.agreement))
            ub = float(pref_b(state.agreement))
            agreements += 1
            if ua > ub:
                new_wins += 1
            elif ub > ua:
                old_wins += 1
            else:
                draws += 1
            print(
                f"{seed:>5}  {'YES':>10}  {ua:>8.3f}  {ub:>8.3f}  {ua+ub:>8.3f}  {state.step:>5}"
            )
        else:
            print(f"{seed:>5}  {'NO':>10}  {'—':>8}  {'—':>8}  {'—':>8}  {'—':>5}")

    print("-" * 58)
    print(f"Agreements  : {agreements}/{n_domains}")
    print(f"New wins    : {new_wins}  |  Old wins: {old_wins}  |  Draws: {draws}")

    runner.plot()
    plt.show()


if __name__ == "__main__":
    run_test(n_domains=10)
