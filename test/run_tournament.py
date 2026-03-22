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
from src.negotiation_bot import GroupN_Negotiator 

def run_self_test():
    # 1. Define Issues (10 distinct options per issue, represented as integers 0-9)
    issues = [
        make_issue(10, name="Apples"), 
        make_issue(10, name="Bananas")
    ]

    # 2. Define Value Functions (Maps choices 0-9 to a utility score 0.0-1.0)
    apple_values = {i: i/9.0 for i in range(10)}
    banana_values = {i: i/9.0 for i in range(10)}

    # 3. Define Linear Additive Preferences for both agents
    pref_a = LinearAdditiveUtilityFunction(
        values=[apple_values, banana_values],
        weights=[0.7, 0.3],
        issues=issues
    )
    pref_b = LinearAdditiveUtilityFunction(
        values=[apple_values, banana_values],
        weights=[0.2, 0.8],
        issues=issues
    )

    # 4. Setup the SAO Mechanism
    runner = SAOMechanism(
        issues=issues,
        n_steps=100
    )
    
    # 5. Add Agents explicitly to the mechanism
    runner.add(GroupN_Negotiator(name="Agent_A"), ufun=pref_a)
    runner.add(GroupN_Negotiator(name="Agent_B"), ufun=pref_b)

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