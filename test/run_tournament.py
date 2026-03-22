import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from negmas.sao import SAOMechanism
from negmas.outcomes import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction
from src.negotiation_agent import Group35_Negotiator


def run_self_test():
    # 1. Define Issues (10 distinct options per issue, represented as integers 0-9)
    issues = [
        make_issue(["Volvo", "Ford", "Ferrari"], name="Brand"),
        make_issue(["red", "yellow", "green"], name="Color"),
    ]

    # 2. # Define Value Functions for both agents
    values_a = [
        {"Volvo": 1.0, "Ford": 0.6, "Ferrari": 0.2},
        {"red": 0.4, "yellow": 0.6, "green": 1.0},
    ]
    values_b = [
        {"Volvo": 0.2, "Ford": 0.6, "Ferrari": 1.0},
        {"red": 1.0, "yellow": 0.5, "green": 0.3},
    ]
    # 3. Define Linear Additive Preferences for both agents
    pref_a = LinearAdditiveUtilityFunction(
        values=values_a,
        weights=[0.8, 0.2],
        issues=issues,
    )
    pref_b = LinearAdditiveUtilityFunction(
        values=values_b,
        weights=[0.8, 0.2],
        issues=issues,
    )

    # 4. Setup the SAO Mechanism
    runner = SAOMechanism(issues=issues, n_steps=100)

    # 5. Add Agents explicitly to the mechanism
    runner.add(Group35_Negotiator(name="Agent_A"), ufun=pref_a)
    runner.add(Group35_Negotiator(name="Agent_B"), ufun=pref_b)

    print("Starting negotiation: Agent A vs Agent B...")

    # 6. Run the negotiation
    state = runner.run()

    # 7. Evaluate Results
    if state.agreement:
        print(f"\n Agreement reached on: {state.agreement}")

        # Manually calculate the final utility to ensure accuracy
        util_a = float(pref_a(state.agreement))
        util_b = float(pref_b(state.agreement))

        print(f"Agent A Utility: {util_a:.2f}")
        print(f"Agent B Utility: {util_b:.2f}")
        print(f"Social Welfare (Sum): {util_a + util_b:.2f}")
    else:
        print("\n No agreement reached before the deadline.")

    # 8. Plot the negotiation trace
    runner.plot()
    plt.show()


if __name__ == "__main__":
    run_self_test()
